import csv
import random
import os

INPUT_FILE = 'synthetic_prompts_sobit.csv'
CHUNK_SIZE = 10_000
OUTPUT_DIR = 'output_chunks'
START_INDEX = 10_000_000  # Starting number for filenames

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_csv_randomly(input_file, chunk_size, output_dir, start_index):
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header
        all_lines = list(reader)

    print(f"Loaded {len(all_lines):,} lines (excluding header). Shuffling...")
    random.shuffle(all_lines)

    print("Writing chunks (no headers)...")
    for i in range(0, len(all_lines), chunk_size):
        chunk = all_lines[i:i+chunk_size]
        file_number = start_index + (i // chunk_size)
        filename = f"{file_number}.csv"
        out_file = os.path.join(output_dir, filename)
        with open(out_file, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerows(chunk)

    total_files = (len(all_lines) + chunk_size - 1) // chunk_size
    print(f"✅ Done. {len(all_lines):,} lines split into {total_files} files.")

split_csv_randomly(INPUT_FILE, CHUNK_SIZE, OUTPUT_DIR, START_INDEX)
