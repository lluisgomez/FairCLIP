import json
import argparse
import torch
import time
from diffusers import AutoPipelineForText2Image
from PIL import Image
import os

def generate_images(input_files, output_dir, batch_size=32):
    # Load the SDXL-Turbo model from the specified path
    model_path = "/gpfs/projects/ehpc42/sdxlturbo/models/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/"
    pipe = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    # Initialize counters for throughput calculation
    total_images = 0
    total_time = 0.0

    # Process each JSON file
    for input_file in input_files:
        # Load the JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Prepare batch processing
        img_ids, captions = list(data.keys()), list(data.values())
        os.makedirs(output_dir, exist_ok=True)

        # Generate images in batches
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i + batch_size]
            batch_ids = img_ids[i:i + batch_size]
            
            # Measure time for the current batch
            start_time = time.time()
            with torch.autocast("cuda"):
                images = pipe(prompt=batch_captions, num_inference_steps=1, guidance_scale=0.0).images
            batch_time = time.time() - start_time

            # Save each image in the batch
            for img, img_id in zip(images, batch_ids):
                output_path = os.path.join(output_dir, f"{img_id}.png")
                img.save(output_path)

            # Update counters
            total_images += len(images)
            total_time += batch_time

            # Print throughput every 5 batches
            if (i // batch_size + 1) % 5 == 0:
                throughput = total_images / total_time
                print(f"Processed {total_images} images with throughput: {throughput:.2f} images/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from captions")
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Paths to the JSON input files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    
    generate_images(args.input_files, args.output_dir, args.batch_size)

