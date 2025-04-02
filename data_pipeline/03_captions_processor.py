import os, glob
import random
import json
import requests
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Register the language_detector factory so spaCy knows about it
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Add the language detector component to the pipeline
if "language_detector" not in nlp.pipe_names:
    nlp.add_pipe("language_detector", last=True)

def is_english(sentence):
    """
    Returns True if the sentence is detected as English.
    """
    doc = nlp(sentence)
    # The language detection returns a dict like {'language': 'en', 'score': 0.999...}
    lang_result = doc._.language
    return lang_result["language"] == "en"

def contains_person(sentence):
    """
    Returns True if the sentence contains at least one PERSON named entity.
    """
    doc = nlp(sentence)
    return any(ent.label_ == "PERSON" for ent in doc.ents)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process prompts for multiple ollama model instances.")
parser.add_argument("--input", type=str, required=True, help="Path to folder with JSON files with captions to be processed.")
parser.add_argument("--output", type=str, required=True, help="Path to output folder where JSON files with results will be saved.")
parser.add_argument("--num_threads", type=int, default=20, help="Number of concurrent threads.")
parser.add_argument("--ports", type=str, default="11434,11435,11436,11437", help="Comma-separated list of ports for ollama instances.")
parser.add_argument("--model_name", type=str, default="tulu3:fairclip", help="Name of the ollama model.")
parser.add_argument("--timeout", type=int, default=10, help="Timeout for each request in seconds.")
args = parser.parse_args()

# Set up the URLs for each model instance using specified ports
ports = args.ports.split(',')
urls = [f"http://localhost:{port}/api/chat" for port in ports]

# Set headers and model-specific data
headers = {"Content-Type": "application/json"}
model_name = args.model_name

# Lists of gender/ethnicity to "swap" demographic group in caption
genders = ['male','female']
ethnicities = ['black','white','asian','latino','middle east','indian']

# Utility function to update prograss bar with requests per second
def update_progress(pb, start_time):
    # Update the progress bar
    pb.update(1)
    # Update RPS in progress bar description
    elapsed_time = time.time() - start_time
    rps = pb.n / elapsed_time if elapsed_time > 0 else 0
    pb.set_postfix(RPS=f"{rps:.2f}")

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
    if output_content != '':
        print(prompt)
        print(output_content)
    return output_content, response_time



def process_json(json_path, output_folder, num_threads):
    """
    Process a JSON file with fileIDs as keys and captions as values.
    
    For each sample in the JSON file, the function:
      - Checks if the caption is English, and (optional) if it contains a PERSON entity.
      - Augments the caption by appending a random ethnicity and gender.
      - Submits the prompt to an ollama model via a threaded request.
      - Collects and saves responses into a JSON output file.
    """
    shard_name = os.path.basename(json_path)
    print(f"Processing folder: {shard_name}")

    outputs = {}
    log_info = {'total': 0, 'no_english': 0, 'has_ne': 0, 'llm_processed': 0}
    start_time = time.time()

    # read JSON file
    with open(json_path) as f:
        data = json.load(f)
    total_captions = len(data)

    with tqdm(total=total_captions, desc=f"Processing {shard_name}", unit="file", leave=True) as progress_bar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for filename in data.keys():
                log_info['total'] += 1
                prompt = data[filename].replace("\n", " ")
                if not is_english(prompt):
                    log_info['no_english'] += 1
                    update_progress(progress_bar, start_time)
                    continue
                # TODO handle samples with PERSON entities differently?
                #if contains_person(prompt):
                #    log_info['has_ne'] += 1
                    # update_progress(progress_bar, start_time)
                    # continue

                ethnicity = random.choice(ethnicities)
                gender = random.choice(genders)
                prompt = f'{prompt} @ {ethnicity} {gender}'
                outputs[filename] = prompt
                log_info['llm_processed'] += 1
                url = urls[len(futures) % len(urls)]  # Round-robin URL selection
                future = executor.submit(send_request, url, prompt)
                futures[future] = filename

            # Collect results as they complete
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    output, response_time = future.result(timeout=args.timeout)
                    # If the response is a single line, replace the initial prompt with the response.
                    if len(output.splitlines()) == 1:
                        outputs[file_name] = output
                except TimeoutError:
                    print(f"Timeout for {file_name}. Skipping.")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

                # Update the progress bar only after a request completes
                progress_bar.update(1)
                # Update RPS in progress bar description
                elapsed_time = time.time() - start_time
                rps = progress_bar.n / elapsed_time if elapsed_time > 0 else 0
                progress_bar.set_postfix(RPS=f"{rps:.2f}")

        # Cancel any remaining incomplete futures
        for future in futures:
            if not future.done():
                future.cancel()
                print(f"Canceled pending future for {futures[future]}")

    # Save outputs to a JSON file named after the shard folder
    output_filename = f"{shard_name}"
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(outputs, f, indent=4)
    return log_info


# Main execution loop to process all shards in the input folder
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # List all json files in the input folder (each representing an extracted shard)
    json_files = sorted(glob.glob(args.input+'/*json'))
    logs = {}
    for json_file in json_files:
        log = process_json(json_file, args.output, args.num_threads)
        logs[json_file] = log

    # Save a log summary of processing
    with open('captions_processor.log', 'w', encoding="utf-8") as f:
        json.dump(logs, f, indent=4)
