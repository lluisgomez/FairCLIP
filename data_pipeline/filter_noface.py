import os
import json
import time
import argparse
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process prompts for multiple ollama model instances.")
parser.add_argument("--input", type=str, required=True, help="Path to folder with extracted shard")
parser.add_argument("--output", type=str, required=True, help="Path to output folder where JSON files with results will be saved.")
args = parser.parse_args()


# Utility function to update prograss bar with requests per second
def update_progress(pb, start_time):
    # Update the progress bar
    pb.update(1)
    # Update RPS in progress bar description
    elapsed_time = time.time() - start_time
    rps = pb.n / elapsed_time if elapsed_time > 0 else 0
    pb.set_postfix(RPS=f"{rps:.2f}")


def process_shard_folder(folder_path, output_folder):
    """
    Process an extracted shard folder containing JSON files.
    
    For each JSON file, the function:
      - Reads the file and extracts a caption.
      - Checks if the sample has at least one face
      - Collects and saves relevant captions into a JSON output file.
    """
    shard_name = os.path.basename(folder_path)
    print(f"Processing folder: {shard_name}")

    outputs = {}
    log_info = {'total': 0, 'no_face': 0}
    start_time = time.time()

    # List all JSON files in the folder
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    total_files = len(json_files)

    with tqdm(total=total_files, desc=f"Processing {shard_name}", unit="file", leave=True) as progress_bar:
            for filename in json_files:
                file_path = os.path.join(folder_path, filename)
                log_info['total'] += 1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    update_progress(progress_bar, start_time)
                    continue

                prompt = data['caption'].replace("\n", " ")
                if len(data.get('face_bboxes', [])) == 0:
                    log_info['no_face'] += 1
                    update_progress(progress_bar, start_time)
                    continue

                outputs[filename] = prompt
                update_progress(progress_bar, start_time)

    # Save outputs to a JSON file named after the shard folder
    output_filename = f"{shard_name}.json"
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(outputs, f, indent=4)
    return log_info


# Main execution loop to process all shards in the input folder
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    logs = {}
    log = process_shard_folder(args.input, args.output)
    logs[os.path.basename(args.input)] = log

    # Save a log summary of processing
    with open('captions_processor.log', 'w+', encoding="utf-8") as f:
        json.dump(logs, f, indent=4)
