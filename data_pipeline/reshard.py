import tarfile
import os
import shutil
import glob
import json
from concurrent.futures import ThreadPoolExecutor

def process_tar_with_json(tar_path, output_tar_path, json_file, file_source_dir, tmp_dir):
    """
    Process a tar file based on a JSON dictionary: delete files matching `fileID*`,
    and add files from `/path/to/files/fileID*`.

    :param tar_path: Path to the original tar file.
    :param output_tar_path: Path to save the modified tar file.
    :param json_file: Path to the JSON file containing the fileID dictionary.
    :param file_source_dir: Directory containing files to add based on fileID.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        file_dict = json.load(f)  # Keys are fileID
    
    file_ids = file_dict.keys()  # Extract fileIDs
    
    # create a temporary directory
    temp_dir = os.path.join(tmp_dir, "temp_tar_processing_" + os.path.basename(tar_path).split('.')[0])
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # Clean up if exists
    os.makedirs(temp_dir)
    
    try:
        # extract tar file to the temporary directory
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=temp_dir)
        
        # overwrite image from `/file_source_dir/fileID*`
        for file_id in file_ids:
            imfile = os.path.join(file_source_dir, f"{file_id}.jpg")
            print('adding ', imfile)
            shutil.copy(imfile, temp_dir)
            if not '@' in file_dict[file_id]:
                print('editing txt and json files')
                with open(os.path.join(temp_dir, file_id+'.txt'), 'w') as f:
                    f.write(file_dict[file_id])
                with open(os.path.join(temp_dir, file_id+'.json')) as f:
                    jdata = json.load(f)
                jdata['caption'] = file_dict[file_id]
                with open(os.path.join(temp_dir, file_id+'.json'), 'w') as f:
                    json.dump(fdata, f)

        
        # repackage the contents into a new tar file
        with tarfile.open(output_tar_path, "w") as tar:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, temp_dir)  # Relative path for tar file
                    tar.add(full_path, arcname=arcname)
    finally:
        # clean up temporary directory
        shutil.rmtree(temp_dir)


def process_all_tars(base_tar_dir, output_tar_dir, json_dir, file_source_base_dir, tmp_dir, num_threads=40):
    """
    Process all tar files in parallel using threads.

    :param base_tar_dir: Directory containing the input tar files.
    :param output_tar_dir: Directory to save the output tar files.
    :param json_dir: Directory containing JSON files.
    :param file_source_base_dir: Base directory containing the files to add.
    :param num_threads: Number of threads to use.
    """
    def process_single_tar(index):
        tar_path = os.path.join(base_tar_dir, f"{index:08d}.tar")
        output_tar_path = os.path.join(output_tar_dir, f"{index:08d}.tar")
        json_file = os.path.join(json_dir, f"{index:08d}.json")
        file_source_dir = os.path.join(file_source_base_dir, f"{index:08d}/")
        process_tar_with_json(tar_path, output_tar_path, json_file, file_source_dir, tmp_dir)

    indices = range(336)  # For tar files from 00000000.tar to 00000335.tar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_single_tar, indices)

# Example Usage
base_tar_dir = "/gpfs_scratch/datacomp/small_filtered/shards"
output_tar_dir = "/gpfs_scratch/datacomp/small_hybrid/shards"
tmp_dir = "/gpfs_scratch/datacomp/small_hybrid/tmp"
json_dir = "/gpfs_scratch/datacomp/small_hybrid/prompts/"
file_source_base_dir = "/gpfs_scratch/datacomp/small_hybrid/images/"
process_all_tars(base_tar_dir, output_tar_dir, json_dir, file_source_base_dir, tmp_dir, num_threads=40)
