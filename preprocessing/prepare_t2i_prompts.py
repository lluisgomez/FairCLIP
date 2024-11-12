import argparse
import os,json
import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process prompts for multiple ollama model instances.")
parser.add_argument("--input", type=str, required=True, help="Path to folder with JSON files.")
parser.add_argument("--output", type=str, required=True, help="Path to output folder where JSON files with results will be saved.")
args = parser.parse_args()



def process_json(file_path, output_folder):
    with open(file_path) as f:
        data = json.load(f)

    prompts = {}

    for k, v in tqdm.tqdm(data.items(), desc="Processing "+file_path):
        s = v.split('\n')[0]
        if '***' not in s:
            continue
            
        ss = s.split('***')
        s1 = ss[0].strip().lower()
        s2 = ss[1].strip().lower()
        
        if s1 == s2:
            continue
            
        w1 = set(s1.split())
        w2 = set(s2.split())
        
        if len(w1) < 3 or len(w2) < 3:
            continue
        
        if len(w1-w2)>1 or len(w2-w1)>1:
            continue
        
        k1 = k.split('-')[0]+'-0'
        k2 = k.split('-')[0]+'-1'
        prompts[k1] = s1
        prompts[k2] = s2


    output_filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=4)
    #print(len(prompts))


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of .json files in the input folder and sort them alphabetically
    json_files = sorted([f for f in os.listdir(args.input) if f.endswith(".json")])

    # Process each json file in alphabetical order
    for filename in json_files:
        file_path = os.path.join(args.input, filename)
        process_json(file_path, args.output)
