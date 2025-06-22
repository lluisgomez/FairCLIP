import os
import pickle
import json
import torch
import open_clip
import numpy as np
from datasets import load_dataset
from scipy.stats import entropy
from tqdm import tqdm
import argparse



# Function to compute normalized entropy
def normalized_entropy(counts):
    counts = np.array(counts, dtype=np.float32)
    probs = counts / counts.sum()
    ent = entropy(probs[probs > 0], base=np.e)
    norm_ent = ent / np.log(len(probs))
    return norm_ent

# Function to compute image embeddings (only once)
def compute_image_embeddings(dataset, model, preprocess, device):
    embeddings = []
    for sample in tqdm(dataset, desc="Computing image embeddings"):
        image = sample['image']
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(image_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        embeddings.append(img_emb.squeeze())
    return torch.stack(embeddings)

# Function to compute text embeddings (only once)
def compute_text_embeddings(words, word_categories, model, tokenizer, device):
    text_embeddings = {}
    for word in tqdm(words, desc="Computing text embeddings"):
        if word_categories[word] == 'ADJECTIVE':
            caption = f'a photo of a/an {word} person'
        elif word_categories[word] == 'NOUN':
            caption = f'a photo of a/an {word}'
        text_input = tokenizer([caption]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(text_input)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
        text_embeddings[word] = text_emb
    return text_embeddings

# Function to compute bias using normalized entropy
def evaluate_normalized_entropy(image_embeddings, text_embeddings, dataset, k=100):
    results_race = {}
    results_gender = {}
    similarities = {}

    total_steps = np.sum([len(words) for _, words in taxonomy.items()])
    progress_bar = tqdm(total=total_steps, desc="Computing text-image similarities")

    for category, words in taxonomy.items():
        word_entropies_race = []
        word_entropies_gender = []

        for word in words:
            text_features = text_embeddings[word]
            sims = image_embeddings @ text_features.T
            similarities[word] = sims
            topk_indices = torch.argsort(sims, dim=0)[-k:]

            race_counts = {label: 0 for label in dataset.features['race'].names}
            gender_counts = {label: 0 for label in dataset.features['gender'].names}
            for idx in topk_indices:
                sample = dataset[int(idx)]
                race_label = dataset.features['race'].int2str(sample['race'])
                gender_label = dataset.features['gender'].int2str(sample['gender'])
                race_counts[race_label] += 1
                gender_counts[gender_label] += 1

            race_entropy = normalized_entropy(list(race_counts.values()))
            word_entropies_race.append(race_entropy)
            gender_entropy = normalized_entropy(list(gender_counts.values()))
            word_entropies_gender.append(gender_entropy)
            progress_bar.update(1)

        results_race[category] = np.mean(word_entropies_race)
        results_gender[category] = np.mean(word_entropies_gender)

    return results_race, results_gender, similarities

# Function to compute C-ASC metric
def evaluate_CASC(similarities, dataset, taxonomy):
    """
    For each category and for each word in that category, compute the C-ASC metric 
    for every demographic field and for every label within that field.

    The C-ASC metric for a given word c and demographic label L in field F is defined as:
      C-ASC(c, L) = ( avg_sim(c, images with label L) - avg_sim(c, images without label L) )
                     / std(sim(c, all images) )

    Returns a nested dictionary of the form:
    { 
       category: { 
           word: {
              field: { label: casc_value, ... },
              ...
           },
           ...
       },
       ...
    }
    """
    results_casc = {}

    # We'll evaluate over two sensitive fields: 'race' and 'gender'
    sensitive_fields = ['race', 'gender']

    total_steps = np.sum([len(dataset.features[field].names) for field in sensitive_fields])
    progress_bar = tqdm(total=total_steps, desc="Split FairFace images by sensitive fields labels:")

    # For each field and label, split images into two groups:
    # G: images that have the label and G_bar: images that don't
    splits_indices = {'race': {}, 'gender': {}}
    for field in sensitive_fields:
        field_labels = dataset.features[field].names
        for label in field_labels:
            indices_label = []
            indices_not_label = []
            for idx in range(len(dataset)):
                sample = dataset[int(idx)]
                sample_label = dataset.features[field].int2str(sample[field])
                if sample_label == label:
                    indices_label.append(idx)
                else:
                    indices_not_label.append(idx)
            splits_indices[field][label]=(indices_label, indices_not_label)
            progress_bar.update(1)

    progress_bar.close()
    total_steps = np.sum([len(words) for _, words in taxonomy.items()])
    progress_bar = tqdm(total=total_steps, desc="Computing C-ASC metrics")

    for category, words in taxonomy.items():
        results_casc[category] = {}
        for word in words:
            results_casc[category][word] = {}
            text_features = text_embeddings[word]
            # Compute cosine similarities between the word and all image embeddings
            sims = similarities[word]
 
            # For each sensitive field, compute C-ASC for every possible label
            for field in sensitive_fields:
                results_casc[category][word][field] = {}
                field_labels = dataset.features[field].names
                for label in field_labels:
                    indices_label     = splits_indices[field][label][0]
                    indices_not_label = splits_indices[field][label][1]

                    avg_label = torch.mean(sims[indices_label], dim=0)
                    avg_not_label = torch.mean(sims[indices_not_label], dim=0)
                    std_all = torch.std(sims, dim=0)
                    casc = (avg_label - avg_not_label) / std_all
                    results_casc[category][word][field][label] = casc.cpu().item()
            progress_bar.update(1)
    return results_casc

def print_extreme_CASC(results_casc):
    """
    For each category and each sensitive field, find the word/label combination
    with the highest absolute C-ASC score (most extreme bias) and print it.
    """
    for category, words_dict in results_casc.items():
        for field in ['race', 'gender']:
            extreme_word = None
            extreme_label = None
            extreme_value = -np.inf
            for word, fields in words_dict.items():
                for label, casc in fields.get(field, {}).items():
                    if casc is None or np.isnan(casc):
                        continue
                    if casc > extreme_value:
                        extreme_value = casc
                        extreme_word = word
                        extreme_label = label
            if extreme_word is not None:
                print(f"Category: {category} - Field: {field} -> Most extreme bias: Word '{extreme_word}', Label '{extreme_label}', C-ASC = {extreme_value:.3f}")
            else:
                print(f"Category: {category} - Field: {field} -> No valid C-ASC score computed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_output_dir", type=str, required=False, default=None, help="Path to output directory from training.")

    args = parser.parse_args()

    # Load the So-B-IT taxonomy
    with open('./So-B-IT_taxonomy.json') as f:
        taxonomy = json.load(f)
    with open('./So-B-IT_categories.json') as f:
        word_categories = json.load(f)

    if args.train_output_dir != None:
        if os.path.exists(os.path.join(args.train_output_dir, 'eval_results_So-B-IT.json')):
            print('Loading precomputed results from eval_results_So-B-IT.json')
            with open(os.path.join(args.train_output_dir, 'eval_results_So-B-IT.json')) as f:
                results = json.load(f)
            for category in taxonomy.keys():
                print(f"Category: {category}, Race Bias (Normalized Entropy): {results['race_results'][category]:.3f}")
                print(f"Category: {category}, Gender Bias (Normalized Entropy): {results['gender_results'][category]:.3f}")
            print_extreme_CASC(results['casc_results'])
            quit()

    # Load the 'validation' split of FairFace with the '0.25' config (tighter face crop)
    # in the '1.25' config the crop is expanded by a factor of 1.25 rel. to the face bbox
    dataset = load_dataset('HuggingFaceM4/FairFace', data_dir='0.25', split='validation')

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    if args.train_output_dir == None:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
    else:
        # Read training information
        train_info_filename = os.path.join(args.train_output_dir, "info.pkl")
        train_info = pickle.load(open(train_info_filename, "rb"))
        model_path = os.path.join(args.train_output_dir, 'checkpoints/epoch_latest.pt')
        model, _, preprocess = open_clip.create_model_and_transforms(train_info['scale_config']['model'], pretrained=model_path, load_weights_only=False)
        tokenizer = open_clip.get_tokenizer(train_info['scale_config']['model'])

    model.eval()
    model.to(device)

    # Precompute image and text embeddings
    image_embeddings = compute_image_embeddings(dataset, model, preprocess, device)
    text_embeddings = compute_text_embeddings([word for category in taxonomy.values() for word in category], word_categories, model, tokenizer, device)

    # Evaluate normalized entropy
    race_results, gender_results, similarities = evaluate_normalized_entropy(image_embeddings, text_embeddings, dataset)
    for category in taxonomy.keys():
        print(f"Category: {category}, Race Bias (Normalized Entropy): {race_results[category]:.3f}")
        print(f"Category: {category}, Gender Bias (Normalized Entropy): {gender_results[category]:.3f}")

    # Evaluate C-ASC
    casc_results = evaluate_CASC(similarities, dataset, taxonomy)
    print_extreme_CASC(casc_results)

    if args.train_output_dir != None:
        results = {'race_results': race_results, 'gender_results': gender_results, 'casc_results': casc_results}
        with open(os.path.join(args.train_output_dir, 'eval_results_So-B-IT.json'), 'w') as f:
            json.dump(results,f)
