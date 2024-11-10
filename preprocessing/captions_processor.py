import argparse
import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function to send a single request to a specified URL with a specific prompt
def send_request(url, prompt, model_name):
    data = {
        "model": model_name,
        "messages": [
            {
              "role": "user",
              "content": prompt
            }
        ],
        "stream": False
    }
    response = requests.post(url, headers={'Content-Type': 'application/json'}, json=data)
    response_time = response.elapsed.total_seconds()  # Get response time
    output_content = response.json().get('message', {}).get('content', '')
    return output_content, response_time

# Function to process all prompts with multiple instances and update progress
def process_prompts(prompts_dict, urls, model_name, num_threads):
    start_time = time.time()
    outputs = {}

    # Initialize tqdm progress bar with total number of prompts
    progress_bar = tqdm(total=len(prompts_dict), desc="Processing prompts", unit="prompt")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Round-robin distribute requests across URLs and collect futures
        futures = {
            executor.submit(send_request, urls[i % len(urls)], prompt, model_name): id 
            for i, (id, prompt) in enumerate(prompts_dict.items())
        }

        # Process each completed future
        for future in as_completed(futures):
            id = futures[future]
            output, response_time = future.result()
            outputs[id] = output

            # Update the progress bar
            progress_bar.update(1)
            elapsed_time = time.time() - start_time
            remaining = len(prompts_dict) - progress_bar.n
            avg_rps = progress_bar.n / elapsed_time if elapsed_time > 0 else 0
            progress_bar.set_postfix(remaining=remaining, avg_rps=f"{avg_rps:.2f}")

    # Close the progress bar
    progress_bar.close()

    # Calculate throughput
    total_time = time.time() - start_time
    num_requests = len(prompts_dict)
    avg_response_time = total_time / num_requests
    throughput = num_requests / total_time

    # Print summary
    print(f"\nTotal Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Response Time: {avg_response_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests per second (RPS)")

    return outputs

# Main function to parse arguments and run the processing
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process prompts for multiple ollama model instances.")
    parser.add_argument("--input", type=str, default="prompts.json", help="Path to input JSON file with prompts.")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON file for results.")
    parser.add_argument("--num_threads", type=int, default=20, help="Number of concurrent threads.")
    parser.add_argument("--ports", type=str, default="11434,11435,11436,11437", help="Comma-separated list of ports for ollama instances.")
    parser.add_argument("--model_name", type=str, default="person_tagger", help="Name of the ollama model.")

    args = parser.parse_args()

    # Parse the list of ports
    urls = [f"http://localhost:{port.strip()}/api/chat" for port in args.ports.split(",")]

    # Load prompts from the input JSON file
    with open(args.input, 'r') as f:
        prompts_dict = json.load(f)

    # Process prompts and capture results
    outputs = process_prompts(prompts_dict, urls, args.model_name, args.num_threads)

    # Save the results to the output JSON file
    with open(args.output, 'w') as f:
        json.dump(outputs, f, indent=4)

# Run the main function
if __name__ == "__main__":
    main()

