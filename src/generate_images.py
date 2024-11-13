import json
import argparse
import torch
import time
import random
from diffusers import AutoPipelineForText2Image
from PIL import Image
import os

def generate_images(input_files, output_dir, batch_size=32):
    # Load the SDXL-Turbo model from the specified path
    model_path = "/gpfs/projects/ehpc42/sdxlturbo/models/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/"
    pipe = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    # Initialize counters for throughput calculation
    total_images = 0
    total_time = 0.0
        
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file
    for input_file in input_files:
        # Load the JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Prepare batch processing
        img_keys = list(data.keys())
        img_captions_pairs = list(data.values())  # Each value is a list with two captions

        # Create a subdirectory in the output_dir for each JSON file
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)

        # Generate images in batches
        for i in range(0, len(img_keys), batch_size):
            # Get a batch of keys and caption pairs
            batch_keys = img_keys[i:i + batch_size]
            batch_caption_pairs = img_captions_pairs[i:i + batch_size]

            # Generate random seeds for this batch
            seeds = [random.randint(0, int(1e9)) for _ in range(len(batch_keys))]

            # First batch of captions (first caption of each pair)
            first_batch_captions = [pair[0] for pair in batch_caption_pairs]
            start_time = time.time()
            images_1 = pipe(prompt=first_batch_captions, num_inference_steps=1, guidance_scale=0.0, generator=[torch.manual_seed(seed) for seed in seeds], width=320, height=240).images

            # Second batch of captions (second caption of each pair), using the same seeds
            second_batch_captions = [pair[1] for pair in batch_caption_pairs]
            images_2 = pipe(prompt=second_batch_captions, num_inference_steps=1, guidance_scale=0.0, generator=[torch.manual_seed(seed) for seed in seeds], width=320, height=240).images
            batch_time = time.time() - start_time

            # Save images with suffixes "_1" and "_2"
            for img_key, img1, img2 in zip(batch_keys, images_1, images_2):
                img1.save(os.path.join(file_output_dir, f"{img_key}_1.png"))
                img2.save(os.path.join(file_output_dir, f"{img_key}_2.png"))

            # Update counters
            total_images += len(images_1) + len(images_2)
            total_time += batch_time

            # Print throughput every 10 batches
            if (i // batch_size + 1) % 10 == 0:
                throughput = total_images / total_time
                print(f"Processed {total_images} images with throughput: {throughput:.2f} images/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from caption pairs with shared seeds")
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Paths to the JSON input files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    
    generate_images(args.input_files, args.output_dir, args.batch_size)

