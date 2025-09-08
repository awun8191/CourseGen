#!/usr/bin/env python3
import os
import hashlib
import sys
from collections import defaultdict

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (IOError, OSError) as e:
        print(f"Error reading {file_path}: {e}")
        return None

def find_duplicates(directory):
    """Find duplicate files based on MD5 hash"""
    file_hashes = defaultdict(list)
    
    print("Scanning files and calculating hashes...")
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_count += 1
            
            if file_count % 100 == 0:
                print(f"Processed {file_count} files...")
            
            md5_hash = calculate_md5(file_path)
            if md5_hash:
                file_hashes[md5_hash].append(file_path)
    
    print(f"Total files processed: {file_count}")
    
    # Find duplicates (hashes with more than one file)
    duplicates = {hash_val: paths for hash_val, paths in file_hashes.items() if len(paths) > 1}
    
    return duplicates

def remove_duplicates(duplicates, dry_run=True):
    """Remove duplicate files, keeping ONE copy (the first one found)"""
    total_duplicates_removed = 0
    total_size_saved = 0
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Processing duplicates...")
    
    for hash_val, file_paths in duplicates.items():
        # Sort paths to have a consistent ordering
        file_paths.sort()
        
        # Keep the first file, remove ALL the rest
        keep_file = file_paths[0]
        files_to_remove = file_paths[1:]  # All except the first one
        
        print(f"\nDuplicate group (Hash: {hash_val[:8]}...):")
        print(f"  ✓ KEEPING: {keep_file}")
        
        for dup_file in files_to_remove:
            try:
                file_size = os.path.getsize(dup_file)
                total_size_saved += file_size
                total_duplicates_removed += 1
                
                print(f"  ✗ {'WOULD REMOVE' if dry_run else 'REMOVING'}: {dup_file} ({file_size:,} bytes)")
                
                if not dry_run:
                    os.remove(dup_file)
                    
            except (OSError, IOError) as e:
                print(f"  ! Error {'checking' if dry_run else 'removing'} {dup_file}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"  Unique file groups found: {len(duplicates)}")
    print(f"  Duplicate files {'that would be' if dry_run else ''} removed: {total_duplicates_removed}")
    print(f"  Original files kept: {len(duplicates)}")
    print(f"  Total space {'that would be' if dry_run else ''} saved: {total_size_saved / (1024*1024):.2f} MB")
    print(f"{'=' * 60}")
    
    return total_duplicates_removed, total_size_saved

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        dry_run = False
        print("🚨 EXECUTING MODE: Will actually delete duplicate files!")
        print("Only ONE copy of each duplicate will be kept.")
        
        response = input("\nAre you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
            
    else:
        dry_run = True
        print("🔍 DRY RUN MODE: Will only show what would be deleted.")
        print("Use --execute to actually delete files.")
    
    current_dir = r"C:\Users\awun8\Documents\SCHOOL"
    print(f"Searching for duplicates in: {current_dir}")
    
    # Find duplicates
    duplicates = find_duplicates(current_dir)
    
    if not duplicates:
        print("✅ No duplicate files found!")
        return
    
    print(f"\n📊 Found {len(duplicates)} groups of duplicate files")
    
    # Show/remove duplicates
    remove_duplicates(duplicates, dry_run)
    
    if dry_run:
        print(f"\n🚀 To actually delete the duplicates, run:")
        print(f"    python3 {sys.argv[0]} --execute")

if __name__ == "__main__":
    main()
