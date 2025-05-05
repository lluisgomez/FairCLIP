import os,glob
import logging
import json
import argparse
import torch
import time
import random
import diffusers
from diffusers import FluxPipeline
from PIL import Image
from multiprocessing import Process
import multiprocessing



def generate_images(gpu_id, input_files, output_dir, batch_size=32):
    # Set the device for this process
    torch.cuda.set_device(gpu_id)
    
    # Load the pipeline and send it to the designated GPU
    model_path = "/gpfs/projects/ehpc42/flux-schnell/models/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/"
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(f"cuda:{gpu_id}")

    total_images = 0
    total_time = 0.0
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        img_keys = list(data.keys())
        img_captions = list(data.values())

        file_name = os.path.splitext(os.path.basename(input_file))[0]
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)

        for i in range(0, len(img_keys), batch_size):
            batch_keys = img_keys[i:i + batch_size]
            batch_captions = img_captions[i:i + batch_size]
            seeds = [random.randint(0, int(1e9)) for _ in range(len(batch_keys))]

            start_time = time.time()
            images = pipe(
                prompt=batch_captions,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=[torch.manual_seed(seed) for seed in seeds],
                width=384, height=384,
                max_sequence_length=256
            ).images

            batch_time = time.time() - start_time

            # save images 
            for img_key, img in zip(batch_keys, images):
                img.save(os.path.join(file_output_dir, img_key.replace('json','jpg')), format="JPEG", quality=90)

            # edit captions 
            for img_key,caption in zip(batch_keys,batch_captions):
                with open(os.path.join(file_output_dir, img_key.replace('json','txt')), 'w') as tf:
                    tf.write(caption)
                with open(os.path.join(file_output_dir.replace('edits','tmp'), img_key)) as jf:
                    jdata = json.load(jf)
                jdata['caption'] = caption
                with open(os.path.join(file_output_dir, img_key), 'w') as jf:
                    jdata = json.dump(jdata, jf)


            total_images += len(images)
            total_time += batch_time
            throughput = total_images / total_time
            print(f"[GPU {gpu_id}] Processed {total_images} images at {throughput:.2f} images/sec")

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)

    os.environ["DIFFUSERS_VERBOSITY"] = "critical"
    os.environ["DIFFUSERS_NO_ADVISORY_WARNINGS"] = "1" 
    diffusers.utils.logging.set_verbosity(diffusers.logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Generate images with multi-GPU data parallelism")
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Paths to the JSON input files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    args = parser.parse_args()

    
    # Split the input files evenly among the available GPUs
    gpu_inputs = [[] for _ in range(args.num_gpus)]
    for idx, input_file in enumerate(args.input_files):
        gpu_inputs[idx % args.num_gpus].append(input_file)
    
    processes = []
    for gpu_id in range(args.num_gpus):
        p = Process(target=generate_images, args=(gpu_id, gpu_inputs[gpu_id], args.output_dir, args.batch_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

