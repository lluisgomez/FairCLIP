import os
import pickle
import json
import torch
import open_clip
import numpy as np
from tqdm import tqdm
import argparse
import glob
from sklearn.metrics import ndcg_score, average_precision_score
import reranking
from PIL import Image

# Professions used in SISPI
professions = ['accountant', 'actor', 'actuary', 'administrator', 'advisor', 'agent', 'aide', 'analyst', 'announcer', 
'appraiser', 'architect', 'archivist', 'artist', 'assembler', 'assessor', 'assistant', 'astronomer', 'athlete', 'attendant', 
'auditor', 'author', 'babysitter', 'bailiff', 'baker', 'bartender', 'biologist', 'boss', 'breeder', 'broker', 'builder', 
'butcher', 'buyer', 'captain', 'caretaker', 'carpenter', 'cartographer', 'cashier', 'ceo', 'chauffeur', 'chef', 'chemist', 
'chief', 'chiropractor', 'choreographer', 'cleaner', 'clerk', 'client', 'coach', 'collector', 'commissioner', 'conductor', 
'cook', 'correspondent', 'counselor', 'courier', 'crafter', 'curator', 'customer', 'dancer', 'dentist', 'designer', 
'detective', 'developer', 'dietitian', 'director', 'dispatcher', 'doctor', 'driver', 'economist', 'editor', 'educator', 
'electrician', 'employee', 'engineer', 'entertainer', 'environmentalist', 'examiner', 'farmer', 'firefighter', 'fisher', 
'forester', 'fundraiser', 'glazier', 'grader', 'groundskeeper', 'guard', 'hairdresser', 'helper', 'homeowner', 
'horticulturalist', 'host', 'housekeeper', 'hunter', 'hygienist', 'inspector', 'installer', 'instructor', 'interpreter', 
'interviewer', 'investigator', 'jailer', 'janitor', 'journalist', 'judge', 'laborer', 'lawyer', 'lecturer', 'legislator', 
'librarian', 'lifeguard', 'logistician', 'machinist', 'magistrate', 'manager', 'marketer', 'mason', 'mathematician', 
'mechanic', 'messenger', 'midwife', 'miner', 'mortician', 'mover', 'musician', 'nurse', 'nutritionist', 'officer', 
'operator', 'optician', 'owner', 'painter', 'paralegal', 'paramedic', 'pathologist', 'pensioner', 'performer', 'pharmacist', 
'photographer', 'physician', 'physicist', 'pilot', 'planner', 'plasterer', 'plumber', 'policeofficer', 'politician', 
'porter', 'postman', 'practitioner', 'president', 'priest', 'producer', 'professor', 'programmer', 'promoter', 
'proofreader', 'psychiatrist', 'psychologist', 'receptionist', 'repairer', 'reporter', 'representative', 'researcher', 
'roofer', 'sailor', 'salesperson', 'scientist', 'screener', 'secretary', 'sheriff', 'singer', 'sociologist', 'soldier', 
'specialist', 'statistician', 'student', 'supervisor', 'surgeon', 'surveyor', 'tailor', 'teacher', 'technician', 'teller', 
'therapist', 'trainer', 'umpire', 'undertaker', 'usher', 'vendor', 'veterinarian', 'waitstaff', 'warden', 'worker', 
'writer']


def compute_image_embeddings(items, model, preprocess, device):
    embeddings = []
    for path in tqdm(items, desc="Computing image embeddings"):
        image = Image.open(path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(image_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        embeddings.append(img_emb.squeeze())
    return torch.stack(embeddings)

def compute_text_embeddings(professions, model, tokenizer, device):
    text_embeddings = []
    for prof in tqdm(professions, desc="Computing text embeddings"):
        text_input = tokenizer([f"a photo of a {prof}"]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(text_input)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_emb.squeeze())
    return torch.stack(text_embeddings)

def evaluate_sispi(text_embeddings, image_embeddings):
    similarities = text_embeddings @ image_embeddings.T
    rankings = torch.argsort(-similarities, dim=1).cpu().numpy()
    sim = similarities.cpu().numpy()

    ndcg = []
    ap = []
    ndkl_gender = []
    ndkl_ethnic = []
    ndkl_gender_ethnic = []

    for i, profession in enumerate(professions):
        true_relevance = np.zeros((1, len(items)))
        for j, img_path in enumerate(items):
            if os.path.basename(img_path).startswith(profession):
                true_relevance[0, j] = 1.

        scores = sim[i, :].reshape(1, -1)
        ndcg.append(ndcg_score(true_relevance, scores))
        ap.append(average_precision_score(true_relevance.squeeze(), scores.squeeze()))

        items_attribute_gender = []
        items_attribute_ethnic = []
        items_attribute_gender_ethnic = []
        gender_dist = {'female': 0.5, 'male': 0.5}
        ethnic_dist = {'asian': 0.25, 'black': 0.25, 'latin': 0.25, 'white': 0.25}
        gender_ethnic_dist = {
            f'{g}_{e}': 0.125 for g in ['female', 'male'] for e in ['asian', 'black', 'latin', 'white']
        }

        for j in rankings[i]:
            img_path = items[j]
            if os.path.basename(img_path).startswith(profession):
                parts = os.path.basename(img_path).split('_')
                gender = parts[2]
                ethnic = parts[3].split('.')[0]
                gender_ethnic = f"{gender}_{ethnic}"
                items_attribute_gender.append(gender)
                items_attribute_ethnic.append(ethnic)
                items_attribute_gender_ethnic.append(gender_ethnic)

        ndkl_gender.append(reranking.ndkl(items_attribute_gender, gender_dist))
        ndkl_ethnic.append(reranking.ndkl(items_attribute_ethnic, ethnic_dist))
        ndkl_gender_ethnic.append(reranking.ndkl(items_attribute_gender_ethnic, gender_ethnic_dist))

    return {
        "ndcg": np.mean(ndcg),
        "ap": np.mean(ap),
        "ndkl_gender": np.mean(ndkl_gender),
        "ndkl_ethnic": np.mean(ndkl_ethnic),
        "ndkl_gender_ethnic": np.mean(ndkl_gender_ethnic),
    }



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_output_dir", type=str, required=False, default=None, help="Path to output directory from training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing SISPI images.")

    args = parser.parse_args()

    if args.train_output_dir is not None:
        cached_results_path = os.path.join(args.train_output_dir, 'eval_results_SISPI.json')
        if os.path.exists(cached_results_path):
            print('Loading precomputed results from eval_results_SISPI.json')
            with open(cached_results_path, 'r') as f:
                results = json.load(f)
            print("\n--- Cached SISPI Evaluation ---")
            print("NDCG: {:.3f}".format(results["ndcg"]))
            print("AP: {:.3f}".format(results["ap"]))
            print("NDKL Gender: {:.3f}".format(results["ndkl_gender"]))
            print("NDKL Ethnic: {:.3f}".format(results["ndkl_ethnic"]))
            print("NDKL Gender+Ethnic: {:.3f}".format(results["ndkl_gender_ethnic"]))
            quit()

    # Load image paths
    items = sorted(glob.glob(os.path.join(args.data_dir, '*.jpg')))

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
    image_embeddings = compute_image_embeddings(items, model, preprocess, device)
    text_embeddings = compute_text_embeddings(professions, model, tokenizer, device)

    # Evaluation
    results = evaluate_sispi(text_embeddings, image_embeddings)
    print("\n--- SISPI Evaluation ---")
    print("NDCG: {:.3f}".format(results["ndcg"]))
    print("AP: {:.3f}".format(results["ap"]))
    print("NDKL Gender: {:.3f}".format(results["ndkl_gender"]))
    print("NDKL Ethnic: {:.3f}".format(results["ndkl_ethnic"]))
    print("NDKL Gender+Ethnic: {:.3f}".format(results["ndkl_gender_ethnic"]))

    # Save results
    if args.train_output_dir is not None:
        out_path = os.path.join(args.train_output_dir, "eval_results_SISPI.json")
        with open(out_path, "w") as f:
            json.dump(results, f)
