#!/usr/bin/env python3
"""
Multithreaded converter:
- Converts .docx, .txt, .ppt, .pptx, .odt to PDF (via LibreOffice headless)
- Deletes originals after successful conversion
- Deletes unwanted file types
- Preserves images and PDFs
- Cleans empty directories
"""

import os
import sys
import shutil
import tempfile
import threading
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

PRINT_LOCK = threading.Lock()

def log(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)

def run_command(cmd, cwd=None, timeout=300):
    """Run a command (list of args preferred) and return (ok, stdout, stderr)."""
    try:
        # prefer list for safety; if a string sneaks in, run with shell
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, cwd=cwd,
                                    capture_output=True, text=True, timeout=timeout)
        else:
            result = subprocess.run(cmd, shell=False, cwd=cwd,
                                    capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def convert_to_pdf(file_path: Path, soffice_bin: str = "libreoffice", timeout: int = 300) -> bool:
    """
    Convert a single file to PDF using LibreOffice in headless mode.
    Uses a unique temporary user profile to allow safe parallelism.
    """
    file_path = Path(file_path)
    output_dir = file_path.parent

    # Unique LO profile per conversion to avoid profile lock collisions
    tmp_profile = Path(tempfile.mkdtemp(prefix="lo_profile_"))
    user_install = f"file://{tmp_profile}"

    cmd = [
        soffice_bin, "--headless",
        f"-env:UserInstallation={user_install}",
        "--norestore", "--nolockcheck",
        "--convert-to", "pdf",
        "--outdir", str(output_dir),
        str(file_path),
    ]

    ok, stdout, stderr = run_command(cmd, timeout=timeout)

    # Cleanup temp LO profile
    try:
        shutil.rmtree(tmp_profile, ignore_errors=True)
    except Exception:
        pass

    if ok:
        pdf_path = output_dir / f"{file_path.stem}.pdf"
        if pdf_path.exists():
            log(f"✓ Converted: {file_path.name} -> {pdf_path.name}")
            return True
        else:
            log(f"✗ PDF not created for: {file_path.name}")
            if stdout:
                log(f"  LO stdout: {stdout.strip()}")
            if stderr:
                log(f"  LO stderr: {stderr.strip()}")
            return False
    else:
        log(f"✗ Failed to convert: {file_path.name}")
        if stderr:
            log(f"  Error: {stderr.strip()}")
        elif stdout:
            log(f"  Output: {stdout.strip()}")
        return False

def should_delete_file(file_path: Path) -> bool:
    """Check if file should be deleted based on extension."""
    delete_extensions = {
        '.m', '.matcad', '.htm', '.html',
        '.mp3', '.mp4', '.aac', '.ogg',
        '.mat', '.tmp', '.dthumb', '.eml',
        '.ods', '.xlsx', '.xls',
    }
    keep_extensions = {
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
    }
    ext = file_path.suffix.lower()
    if ext in delete_extensions:
        return True
    if ext in keep_extensions:
        return False
    if ext in {'.docx', '.txt', '.ppt', '.pptx', '.odt'}:
        return False
    return False  # unknown: keep

def convert_and_cleanup(file_path: Path, soffice_bin: str, timeout: int):
    """Worker: convert a file; if success, delete original. Returns (success)."""
    success = convert_to_pdf(file_path, soffice_bin=soffice_bin, timeout=timeout)
    if success:
        try:
            file_path.unlink()
            log(f"  Deleted original: {file_path.name}")
        except Exception as e:
            log(f"  Warning: Could not delete original {file_path.name}: {e}")
    return success

def process_directory(base_dir: str, max_workers: int = 3, soffice_bin: str = "libreoffice", timeout: int = 300):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return

    convertible_extensions = {'.docx', '.txt', '.ppt', '.pptx', '.odt'}
    files_to_convert = [
        p for p in base_path.rglob('*')
        if p.is_file()
        and p.suffix.lower() in convertible_extensions
        and not p.name.startswith('~$')
    ]

    converted_count = 0
    failed_conversions = 0
    deleted_count = 0

    print(f"Processing directory: {base_path}")
    print("=" * 60)
    print("\n1. Converting files to PDF (multithreaded)...")
    print("-" * 40)
    log(f"Found {len(files_to_convert)} file(s) to convert. Using {max_workers} worker(s).")

    # Convert in parallel
    if files_to_convert:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(convert_and_cleanup, f, soffice_bin, timeout): f
                for f in files_to_convert
            }
            for fut in as_completed(futures):
                try:
                    ok = fut.result()
                    if ok:
                        converted_count += 1
                    else:
                        failed_conversions += 1
                except Exception as e:
                    failed_conversions += 1
                    log(f"  Unexpected error for {futures[fut].name}: {e}")

    # Second pass: Delete unwanted files (sequential – fast enough, safe)
    print(f"\n2. Deleting unwanted file types...")
    print("-" * 40)
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and should_delete_file(file_path):
            try:
                log(f"Deleting: {file_path.relative_to(base_path)}")
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                log(f"Error deleting {file_path.name}: {e}")

    # Clean up empty directories
    print(f"\n3. Cleaning up empty directories...")
    print("-" * 40)
    empty_dirs_removed = 0
    for dir_path in sorted(base_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if dir_path.is_dir():
            try:
                next(dir_path.iterdir())
            except StopIteration:
                try:
                    dir_path.rmdir()
                    log(f"Removed empty directory: {dir_path.relative_to(base_path)}")
                    empty_dirs_removed += 1
                except Exception as e:
                    log(f"Could not remove directory {dir_path.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Files converted to PDF: {converted_count}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Unwanted files deleted: {deleted_count}")
    print(f"Empty directories removed: {empty_dirs_removed}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Multithreaded office-to-PDF converter & cleaner")
    parser.add_argument("--dir", default="/home/raregazetto/Documents/SCHOOL/COMPILATION/100", help="Base directory to process")
    parser.add_argument("--workers", type=int, default=max(2, min(4, (os.cpu_count() or 4) // 2 + 1)),
                        help="Number of parallel conversions (2–4 recommended)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-file conversion timeout (seconds)")
    parser.add_argument("--soffice", default="libreoffice", help="LibreOffice binary (e.g., 'soffice' or 'libreoffice')")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if not args.yes:
        print("This script will:")
        print("1. Convert .docx, .txt, .ppt, .pptx, .odt files to PDF (in parallel)")
        print("2. Delete original files after successful conversion")
        print("3. Delete unwanted file types (matlab, html, media files, etc.)")
        print("4. Keep images (.jpg, .png, etc.) and PDFs")
        print("5. Clean up empty directories")
        resp = input("\nDo you want to continue? (y/N): ")
        if resp.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    process_directory(args.dir, max_workers=max(1, args.workers),
                      soffice_bin=args.soffice, timeout=args.timeout)

if __name__ == "__main__":
    main()
