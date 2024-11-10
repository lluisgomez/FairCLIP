import tarfile
import json
import argparse
import os

def extract_txt_files_from_tar(tar_path):
    prompts_dict = {}

    # Open the tar file
    with tarfile.open(tar_path, 'r') as tar:
        # Iterate through each file in the tar archive
        for member in tar.getmembers():
            # Process only .txt files
            if member.isfile() and member.name.endswith('.txt'):
                # Extract the file content
                file = tar.extractfile(member)
                content = file.read().decode('utf-8')
                
                # Use the filename without the .txt extension as the key
                filename = os.path.splitext(os.path.basename(member.name))[0]
                prompts_dict[filename] = content

    return prompts_dict

def save_to_json(data, output_path):
    # Save the dictionary to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Convert .txt files in a tar archive to JSON format.")
    parser.add_argument("--tar", type=str, required=True, help="Path to the input .tar file.")
    parser.add_argument("--output", type=str, default="prompts.json", help="Path to the output JSON file.")

    args = parser.parse_args()

    # Extract .txt files and convert to JSON format
    prompts_dict = extract_txt_files_from_tar(args.tar)
    save_to_json(prompts_dict, args.output)
    print(f"JSON file '{args.output}' created successfully.")

if __name__ == "__main__":
    main()

