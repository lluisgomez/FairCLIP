import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import open_clip
import re
import numpy as np


# Set device and load the OpenCLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.to(device)
model.eval() 
tokenizer = open_clip.get_tokenizer('ViT-B-32')

class Flickr30kDataLoader:
    def __init__(self, split="test", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None
    
    def load_data(self):
        # Load Flickr30k Dataset
        self.dataset = load_dataset("nlphuji/flickr30k")
        
        # Filter dataset based on the provided split
        self.dataset = self.dataset['test'].filter(lambda example: example['split'] == self.split)
        print(f"Filtered {self.split} set size: {len(self.dataset)}")
        
    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        return {"images": images, "captions": captions}
    
    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return self.dataloader

# Initialize the custom data loader (using the "test" split; adjust as needed)
data_loader_obj = Flickr30kDataLoader(split="test", batch_size=32)
dataloader = data_loader_obj.get_dataloader()

# Containers for storing features
all_image_features = []
all_text_features = []

# Process the dataset in batches
with torch.no_grad():
    for batch in dataloader:
        # Preprocess images: the preprocess function expects a PIL image
        images = batch["images"]
        processed_images = [preprocess(img).unsqueeze(0) for img in images]
        images_tensor = torch.cat(processed_images, dim=0).to(device)
        
        # For each image, select the first caption (if multiple captions are available)
        captions = [caps[0] if len(caps) > 0 else "" for caps in batch["captions"]]
        
        # Tokenize captions
        text_tokens = tokenizer(captions).to(device)
        
        # Compute image and text features
        image_features = model.encode_image(images_tensor)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Append the computed features
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())

# Concatenate features from all batches
all_image_features = torch.cat(all_image_features, dim=0)
all_text_features = torch.cat(all_text_features, dim=0)

# Compute similarity matrix (cosine similarity via dot product)
similarity = all_text_features @ all_image_features.T

# Evaluate retrieval: for each caption, check if the corresponding image is among the top-k
num_samples = similarity.size(0)
r1, r5, r10 = 0, 0, 0

for i in range(num_samples):
    sim_scores = similarity[i]
    ranked_indices = torch.argsort(sim_scores, descending=True)
    if i in ranked_indices[:1]:
        r1 += 1
    if i in ranked_indices[:5]:
        r5 += 1
    if i in ranked_indices[:10]:
        r10 += 1

# Convert counts to percentage recall values
r1 = r1 / num_samples * 100
r5 = r5 / num_samples * 100
r10 = r10 / num_samples * 100

print("Text-to-Image Retrieval Results:")
print(f"R@1: {r1:.2f}%")
print(f"R@5: {r5:.2f}%")
print(f"R@10: {r10:.2f}%")


print('Computing bias metrics...')


# Define gender word lists (from the paper’s table)
masculine_words = ["man", "men", "male", "boy", "gentleman", "father", "brother", "son", "husband", "boyfriend"]
feminine_words = ["woman", "women", "female", "girl", "lady", "mother", "mom", "sister", "daughter", "wife", "girlfriend"]


def get_gender_label(caption):
    """
    Determines the gender label of a caption.
    Returns:
        1 for male (if at least one masculine word is present and no feminine word),
        2 for female (if at least one feminine word is present and no masculine word),
        0 for gender-neutral otherwise.
    """
    cap_lower = caption.lower()
    # Use regex with word boundaries to ensure we match complete words.
    has_masc = any(re.search(r'\b' + re.escape(word) + r'\b', cap_lower) for word in masculine_words)
    has_fem = any(re.search(r'\b' + re.escape(word) + r'\b', cap_lower) for word in feminine_words)
    
    if has_masc and not has_fem:
        return 1  # male
    elif has_fem and not has_masc:
        return 2  # female
    else:
        return 0  # gender-neutral


def neutralize_caption(caption):
    """
    Replaces gender-specific words with gender-neutral alternatives.
    Examples:
        "A man with a red helmet..." -> "A person with a red helmet..."
        "A little girl is..." -> "A little child is..."
    """
    # Replace masculine words (note: using regex word boundaries for exact matching)
    caption = re.sub(r'\b(man|male|boy|gentleman|father|brother|son|husband|boyfriend)\b', 'person', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\b(men)\b', 'people', caption, flags=re.IGNORECASE)
    # Replace feminine words
    caption = re.sub(r'\b(woman|female|girl|lady|mother|mom|sister|daughter|wife|girlfriend)\b', 'person', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\b(women)\b', 'people', caption, flags=re.IGNORECASE)
    return caption

# Custom DataLoader Flickr30k for Bias evals (only one caption per image)
class Flickr30k1CDataLoader:
    def __init__(self, split="test", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None
    
    def load_data(self):
        # Load Flickr30k Dataset
        self.dataset = load_dataset("nlphuji/flickr30k")
        
        # Filter dataset based on the provided split
        self.dataset = self.dataset['test'].filter(lambda example: example['split'] == self.split)
        print(f"Filtered {self.split} set size: {len(self.dataset)}")
        
    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        # Here we take only one caption! (see FairSample (Wang et al., 2021a))
        captions = [item["caption"][0] for item in batch]
        return {"images": images, "captions": captions}
    
    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return self.dataloader

# Evaluation function for retrieval bias
def evaluate(similarity: np.ndarray, gender: np.ndarray):
    """
    Evaluates bias metrics.
    Args:
        similarity: A numpy array of shape [N, N] with cosine similarities.
        gender: A numpy array of shape [N] with gender labels (1: male, 2: female, 0: neutral).
    Returns:
        biases: List of bias values for top-K (K=1,...,10).
    """
    npt = similarity.shape[0]
    male_counts = np.zeros(npt)
    female_counts = np.zeros(npt)

    # Bias metric: Bias@K for K=1 to 10
    biases = []
    for k in range(1, 11):
        for i in range(npt):
            inds = np.argsort(similarity[i])[::-1][:k]
            male_counts[i] = (gender[inds] == 1).sum()
            female_counts[i] = (gender[inds] == 2).sum()
        # Compute bias as the mean normalized difference between male and female counts.
        bias = (male_counts - female_counts) / (male_counts + female_counts + 1e-12)
        bias = bias.mean()
        biases.append(bias)
    return (biases[0], biases[4], biases[9])


# Load data using the custom dataloader (using the "test" split)
data_loader_obj = Flickr30k1CDataLoader(split="test", batch_size=32)
dataloader = data_loader_obj.get_dataloader()

all_image_features = []
all_text_features = []
gender_labels = []  # To store gender labels (derived from the original caption)

with torch.no_grad():
    for batch in dataloader:
        images = batch["images"]
        orig_captions = batch["captions"]  # Original (first) captions
        # Compute gender labels from the original caption
        for cap in orig_captions:
            gender_labels.append(get_gender_label(cap))
        # Pre-process captions to obtain gender-neutral text queries
        neutral_captions = [neutralize_caption(cap) for cap in orig_captions]
        #print(orig_captions, neutral_captions, [get_gender_label(cap) for cap in orig_captions])
        
        # Preprocess images using the provided transform (expects PIL.Image)
        processed_images = [preprocess(img).unsqueeze(0) for img in images]
        images_tensor = torch.cat(processed_images, dim=0).to(device)
        
        # Tokenize the neutralized captions
        text_tokens = tokenizer(neutral_captions).to(device)
        
        # Compute image and text features
        image_features = model.encode_image(images_tensor)
        text_features = model.encode_text(text_tokens)
        
        # Normalize the features for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())

# Concatenate features and gender labels
all_image_features = torch.cat(all_image_features, dim=0)
all_text_features = torch.cat(all_text_features, dim=0)
gender_labels = np.array(gender_labels)

# Compute similarity matrix (cosine similarity via dot product)
similarity = all_text_features @ all_image_features.T
similarity = similarity.numpy()

# Evaluate recall and bias metrics
biases = evaluate(similarity, gender_labels)

print(f"Bias@1: {biases[0]:.4f}%")
print(f"Bias@5: {biases[1]:.4f}%")
print(f"Bias@10: {biases[2]:.4f}%")

