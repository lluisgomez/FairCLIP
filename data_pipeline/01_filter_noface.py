import tarfile
import json
import os
import argparse
from io import TextIOWrapper
from tqdm import tqdm
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process shards and filter caption for images with at least one face detected.")
parser.add_argument("--input", type=str, required=True, help="Path to .tar shard file")
parser.add_argument("--output", type=str, required=True, help="Path to output folder where JSON files with results will be saved.")
parser.add_argument("--edits", type=str, required=True, help="Path to folder where selected JSON files with original sample data will be saved.")
args = parser.parse_args()


tar_path = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

shard_name = os.path.splitext(os.path.basename(tar_path))[0]
output_json = os.path.join(output_dir, f"{shard_name}.json")

# Create edits subdirectory for this shard
edit_shard_dir = os.path.join(args.edits, shard_name)
os.makedirs(edit_shard_dir, exist_ok=True)

outputs = {}
log_info = {'total': 0, 'no_face': 0}
start_time = time.time()

with tarfile.open(tar_path, "r") as tar:
    members = [m for m in tar.getmembers() if m.name.endswith('.json')]
    with tqdm(total=len(members), desc=f"Processing {shard_name}", unit="file") as pbar:
        for member in members:
            log_info['total'] += 1
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                text_file = TextIOWrapper(f, encoding='utf-8')
                data = json.load(text_file)

                caption = data['caption'].replace("\n", " ")
                if len(data.get('face_bboxes', [])) == 0:
                    log_info['no_face'] += 1
                    pbar.update(1)
                    continue

                filename = os.path.basename(member.name)
                outputs[filename] = caption

                # Save the original JSON data
                with open(os.path.join(edit_shard_dir, filename), 'w', encoding='utf-8') as out_json:
                    json.dump(data, out_json, indent=4)

            except Exception as e:
                print(f"Error processing {member.name}: {e}")
            pbar.update(1)

# Save output JSON
with open(output_json, 'w', encoding='utf-8') as f_out:
    json.dump(outputs, f_out, indent=4)

# Optionally log summary
with open('captions_processor.log', 'a', encoding='utf-8') as f_log:
    json.dump({shard_name: log_info}, f_log, indent=4)
    f_log.write('\n')

