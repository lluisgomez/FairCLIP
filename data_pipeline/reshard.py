import os
import tarfile
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Create a shard from pre-existing tar, updating those samples found in specified directory.")
parser.add_argument("--input", type=str, required=True, help="Path to existing shard")
parser.add_argument("--output", type=str, required=True, help="Path to output shard.")
parser.add_argument("--edits", type=str, required=True, help="Path to folder with files to update.")
args = parser.parse_args()

def build_modified_files_set(tmp_folder):
    """
    Walk the temporary folder and build a set of file paths (relative to tmp_folder)
    for all files that have been modified.
    """
    modified_files = set()
    for root, dirs, files in os.walk(tmp_folder):
        for name in files:
            # Get the relative path with respect to tmp_folder
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, tmp_folder)
            modified_files.add(rel_path)
    return modified_files

def repack_tar(original_tar_path, tmp_folder, new_tar_path, modified_files):
    """
    Create a new tar archive by copying files from the original tar for files that 
    haven't changed, and using the modified files from tmp_folder when available.
    """
    with tarfile.open(original_tar_path, 'r') as orig_tar, \
         tarfile.open(new_tar_path, 'w') as new_tar:
        
        for member in orig_tar.getmembers():
            # Check if this member was modified by comparing its relative name.
            if member.name in modified_files:
                print(f"Using modified file for: {member.name}")
                file_path = os.path.join(tmp_folder, member.name)
                # Add the file from tmp_folder to the new tar archive.
                new_tar.add(file_path, arcname=member.name)
            else:
                #print(f"Copying original file for: {member.name}")
                # For regular files, extract a file object and add it.
                fileobj = orig_tar.extractfile(member)
                if fileobj is not None:
                    new_tar.addfile(member, fileobj)
                else:
                    # For directories or special files, add the member directly.
                    new_tar.addfile(member)


if __name__ == "__main__":
    # Build a set of modified file paths from the specified folder.
    modified_files = build_modified_files_set(args.edits)
    repack_tar(args.input, args.edits, args.output, modified_files)

