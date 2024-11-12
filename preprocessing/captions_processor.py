import os
import tarfile
import json
import requests
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process prompts for multiple ollama model instances.")
parser.add_argument("--input", type=str, required=True, help="Path to folder with shard tar files.")
parser.add_argument("--output", type=str, required=True, help="Path to output folder where JSON files with results will be saved.")
parser.add_argument("--num_threads", type=int, default=20, help="Number of concurrent threads.")
parser.add_argument("--ports", type=str, default="11434,11435,11436,11437", help="Comma-separated list of ports for ollama instances.")
parser.add_argument("--model_name", type=str, default="person_tagger", help="Name of the ollama model.")
parser.add_argument("--timeout", type=int, default=10, help="Timeout for each request in seconds.")
args = parser.parse_args()

# Set up the URLs for each model instance using specified ports
ports = args.ports.split(',')
urls = [f"http://localhost:{port}/api/chat" for port in ports]

# Set headers and model-specific data
headers = {"Content-Type": "application/json"}
model_name = args.model_name

# Function to send a single request to a specified URL
def send_request(url, prompt):
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data, timeout=args.timeout)
    response_time = response.elapsed.total_seconds()
    output_content = response.json().get('message', {}).get('content', '')
    return output_content, response_time

# Process each shard directly from the tarfile
def process_shard(file_path, output_folder, num_threads):
    # Print the name of the shard being processed
    shard_name = os.path.basename(file_path)
    print(f"Processing shard: {shard_name}")

    outputs = {}
    start_time = time.time()  # Start timer for RPS calculation

    # Open the tar file and process each .txt file immediately
    with tarfile.open(file_path, "r") as tar:
        # Set leave=True to keep the progress bar after completion for diagnostic visibility
        with tqdm(total=10000, desc=f"Processing {shard_name}", unit="file", leave=True) as progress_bar:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {}
                for member in tar:
                    if member.isfile() and member.name.endswith(".txt"):
                        file = tar.extractfile(member)
                        if file:
                            prompt = file.read().decode("utf-8").strip()
                            url = urls[len(futures) % len(urls)]  # Round-robin URL selection
                            future = executor.submit(send_request, url, prompt)
                            futures[future] = member.name

                # Collect results as they complete
                for future in as_completed(futures):
                    file_name = futures[future]
                    try:
                        output, response_time = future.result(timeout=args.timeout)
                        if '***' in output:
                            outputs[file_name] = output
                        #print(f"Processed {file_name}: Time Taken {response_time:.2f} seconds")
                    except TimeoutError:
                        #print(f"Timeout for {file_name}. Skipping.")
                        outputs[file_name] = "Timeout"  # Mark timeout in the output
                    except Exception as e:
                        #print(f"Error processing {file_name}: {e}")
                        outputs[file_name] = str(e)  # Mark other errors in the output
                    
                    # Update the progress bar only after a request completes
                    progress_bar.update(1)

                    # Update RPS in progress bar description
                    elapsed_time = time.time() - start_time
                    rps = progress_bar.n / elapsed_time if elapsed_time > 0 else 0
                    progress_bar.set_postfix(RPS=f"{rps:.2f}")

            # Cancel any remaining incomplete futures (if any)
            for future in futures:
                if not future.done():
                    future.cancel()
                    print(f"Canceled pending future for {futures[future]}")

    # Save outputs to a JSON file named after the shard file
    output_filename = shard_name.replace(".tar", ".json")
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=4)

# Main execution loop to process all shards in the input folder
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of .tar files in the input folder and sort them alphabetically
    tar_files = sorted([f for f in os.listdir(args.input) if f.endswith(".tar")])

    # Process each shard in alphabetical order
    for filename in tar_files:
        file_path = os.path.join(args.input, filename)
        process_shard(file_path, args.output, args.num_threads)

