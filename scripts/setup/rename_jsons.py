import os
import re
from pathlib import Path

def rename_json_files(base_path, dry_run=True):
    """
    Rename all JSON files in subfolders of base_path to uniform pattern.
    dry_run=True: only print what would be done.
    dry_run=False: actually rename.
    """
    base = Path(base_path)
    
    # Find all .json files recursively
    json_files = list(base.rglob('*.json'))
    
    renamed_count = 0
    skipped_count = 0
    
    for json_path in json_files:
        # Get parent folder name (e.g., 'sub_5')
        folder_name = json_path.parent.name
        # Extract subject ID from folder name (e.g., 'sub_5' -> '5')
        sub_match = re.search(r'sub[_-]?(\d+)', folder_name, re.IGNORECASE)
        if not sub_match:
            print(f"Skipping {json_path}: could not extract subject ID from folder '{folder_name}'")
            skipped_count += 1
            continue
        subject_id = sub_match.group(1)
        
        # Extract plearning number from filename (any integer)
        stem = json_path.stem  # filename without extension
        numbers = re.findall(r'\d+', stem)
        if not numbers:
            print(f"Skipping {json_path}: no number found in filename")
            skipped_count += 1
            continue
        plearning_num = numbers[-1]  # take last number
        
        # New filename
        new_name = f"subject_{subject_id}_plearning_{plearning_num}.json"
        new_path = json_path.parent / new_name
        
        # Check if target already exists
        if new_path.exists():
            print(f"Skipping {json_path}: target {new_name} already exists in same folder")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"Would rename: {json_path.name} -> {new_name}")
        else:
            json_path.rename(new_path)
            print(f"Renamed: {json_path.name} -> {new_name}")
        renamed_count += 1
    
    print(f"\nSummary: {renamed_count} files would be renamed, {skipped_count} skipped (dry_run={dry_run})")

if __name__ == "__main__":
    # Set your base path where the subject folders live
    base_path = "/Users/maddiemac/Puberty_VR/results"
    
    # First, dry run to see what will happen
    print("=== DRY RUN ===")
    rename_json_files(base_path, dry_run=True)
    
    # If satisfied, change dry_run to False and run again
    print("\n" + "="*50)
    answer = input("Do you want to proceed with actual renaming? (yes/no): ").strip().lower()
    if answer == 'yes':
        rename_json_files(base_path, dry_run=False)
    else:
        print("No files were renamed.")