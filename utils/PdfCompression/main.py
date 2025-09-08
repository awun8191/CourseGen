#!/usr/bin/env python3
"""
OCR‑safe, mirror‑preserving batch compressor for PDF & PPTX.
Now with **aggressive typed‑PDF mode** and **OCR‑safe mode** for handwritten/image‑heavy notes.

What changed
- Classifies PDFs into {image_heavy | typed | mixed} using `pdfimages -list` ratio.
- Keeps high DPI + lossless filters for image‑heavy/handwritten (better OCR).
- Uses stronger JPEG + lower DPI for typed PDFs (bigger wins on textbooks/“tipped” files).
- Mixed docs land in the middle.
- Logging is 3.13‑safe (datefmt).

Class defaults (tunable via CLI):
  image‑heavy:  res=min(ocr_min_dpi, 300 default), filters=Flate/CCITT, quality=N/A
  typed:       res=110 dpi, filters=DCT, JPEG quality=60 (aggressive)
  mixed:       res=200 dpi, filters=DCT, JPEG quality=80 (balanced)

Usage examples
  # Mirror into ./out, preserve originals, keep OCR‑safe only for image‑heavy
  python compressor.py ./docs --output ./out --ocr-safe

  # Make typed more brutal (even smaller), and tweak thresholds
  python compressor.py ./books --output ./out \
    --typed-resolution 96 --typed-quality 55 --typed-ratio-max 0.30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------- Logging ---------------------------------
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger("compressor")

# -------------------------- Dependency Checks -------------------------

def which_gs() -> Optional[str]:
    for name in ["gs", "gswin64c", "gswin32c", "gswin64c.exe", "gswin32c.exe"]:
        p = shutil.which(name)
        if p:
            return p
    return None


def is_tool_available(name: str) -> bool:
    return shutil.which(name) is not None

GS_BIN = which_gs()
PDFIMAGES_AVAILABLE = is_tool_available("pdfimages")

try:
    from pypdf import PdfWriter, PdfReader
    PYPDF_AVAILABLE = True
except Exception:  # pragma: no cover
    PYPDF_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    PILLOW_AVAILABLE = False

# ----------------------------- Config ---------------------------------

@dataclass
class PDFConfig:
    # Baseline knobs
    resolution: int = 120
    base_quality: int = 85
    image_heavy_quality: int = 90
    image_ratio_threshold: float = 0.50  # >= this => image_heavy
    method: str = "auto"
    timeout_s: int = 600
    safe_image_dense: bool = True
    image_heavy_min_dpi: int = 220
    auto_filter_images: bool = True
    detect_duplicate_images: bool = True
    # OCR safety (applies to image_heavy only)
    ocr_safe: bool = False
    ocr_min_dpi: int = 300
    # New: typed/mixed aggressiveness
    typed_ratio_max: float = 0.25  # <= this => typed
    typed_resolution: int = 110
    typed_quality: int = 60
    mixed_resolution: int = 200
    mixed_quality: int = 80


@dataclass
class PPTXConfig:
    image_quality: int = 80


@dataclass
class RunConfig:
    input_path: Path
    output_folder: Optional[Path] = None
    workers: int = max(1, (os.cpu_count() or 2) // 2)
    min_size_mb: float = 0.5
    min_reduction_pct: float = 1.0
    backup: bool = False
    dry_run: bool = False
    preserve_times: bool = True
    report_json: Optional[Path] = None
    report_csv: Optional[Path] = None


SUPPORTED_EXTS = {".pdf", ".pptx"}

# --------------------------- Utilities --------------------------------

def human(n: int) -> str:
    f = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


def safe_temp_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".tmp")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_stat(src: Path, dst: Path) -> None:
    try:
        shutil.copystat(src, dst)
    except Exception:
        pass


def compute_out_path(src_file: Path, run_cfg: RunConfig) -> Path:
    if run_cfg.output_folder:
        if run_cfg.input_path.is_dir():
            rel = src_file.relative_to(run_cfg.input_path)
        else:
            rel = Path(src_file.name)
        return run_cfg.output_folder / rel
    return src_file

# ------------------------ PDF Diagnose & Classify ---------------------

def diagnose_pdf_image_ratio(input_path: Path) -> Tuple[float, Optional[str]]:
    if not PDFIMAGES_AVAILABLE:
        return 0.0, "pdfimages not available"
    try:
        result = subprocess.run(["pdfimages", "-list", str(input_path)], capture_output=True, text=True, check=True)
        total = 0.0
        size_re = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)([BKMG])\b", re.IGNORECASE)
        for line in result.stdout.splitlines()[2:]:
            m = size_re.search(line)
            if not m:
                continue
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit == "B":
                total += val
            elif unit == "K":
                total += val * 1024
            elif unit == "M":
                total += val * 1024 ** 2
            elif unit == "G":
                total += val * 1024 ** 3
        file_size = input_path.stat().st_size or 1
        return total / file_size, None
    except subprocess.CalledProcessError as e:
        return 0.0, f"pdfimages failed: rc={e.returncode}"
    except Exception as e:  # pragma: no cover
        return 0.0, str(e)


def classify_pdf(input_path: Path, cfg: PDFConfig) -> Tuple[str, float]:
    """Return (class, ratio). class in {"image_heavy","typed","mixed"}."""
    ratio, _ = diagnose_pdf_image_ratio(input_path)
    if ratio >= cfg.image_ratio_threshold:
        return "image_heavy", ratio
    if ratio <= cfg.typed_ratio_max:
        return "typed", ratio
    return "mixed", ratio

# ---------------------------- PDF Compress ----------------------------

def compress_pdf_with_gs(input_path: Path, output_path: Path, cfg: PDFConfig) -> bool:
    if not GS_BIN:
        return False

    doc_class, ratio = classify_pdf(input_path, cfg)

    if doc_class == "image_heavy":
        eff_res = max(cfg.image_heavy_min_dpi if cfg.safe_image_dense else cfg.resolution, cfg.ocr_min_dpi if cfg.ocr_safe else cfg.resolution)
        eff_quality = cfg.image_heavy_quality
        ocr_mode = cfg.ocr_safe
        color_filter_block = [
            "-dAutoFilterColorImages=false",
            "-dAutoFilterGrayImages=false",
            "-dColorImageFilter=/FlateEncode",
            "-dGrayImageFilter=/FlateEncode",
            "-dMonoImageFilter=/CCITTFaxEncode",
            "-dEncodeMonoImages=true",
            "-dMonoImageDownsampleType=/Subsample",
            "-dMonoImageDownsampleThreshold=1.0",
        ]
    elif doc_class == "typed":
        eff_res = cfg.typed_resolution
        eff_quality = cfg.typed_quality
        ocr_mode = False
        color_filter_block = [
            "-dAutoFilterColorImages=false",
            "-dAutoFilterGrayImages=false",
            "-dColorImageFilter=/DCTEncode",
            "-dGrayImageFilter=/DCTEncode",
            f"-dJPEGQuality={eff_quality}",
        ]
    else:  # mixed
        eff_res = cfg.mixed_resolution
        eff_quality = cfg.mixed_quality
        ocr_mode = False
        color_filter_block = [
            "-dAutoFilterColorImages=false",
            "-dAutoFilterGrayImages=false",
            "-dColorImageFilter=/DCTEncode",
            "-dGrayImageFilter=/DCTEncode",
            f"-dJPEGQuality={eff_quality}",
        ]

    logger.info(f"[GS] {input_path.name}: class={doc_class} ratio≈{ratio:.0%} → res={eff_res}dpi, q={eff_quality}, ocr_safe={ocr_mode}")

    cmd = [
        GS_BIN,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/default",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={str(output_path)}",
        str(input_path),
        # Downsampling and resolutions
        "-dDownsampleColorImages=true",
        "-dDownsampleGrayImages=true",
        "-dDownsampleMonoImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={eff_res}",
        f"-dGrayImageResolution={eff_res}",
        f"-dMonoImageResolution={eff_res}",
        # Images & fonts
        f"-dDetectDuplicateImages={'true' if cfg.detect_duplicate_images else 'false'}",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
    ] + color_filter_block

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=cfg.timeout_s)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[GS] Timeout after {cfg.timeout_s}s: {input_path}")
        return False
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "")[-4000:]
        logger.error(f"[GS] Failed {input_path}\nSTDERR: {tail}")
        return False


def compress_pdf_with_pypdf(input_path: Path, output_path: Path) -> bool:
    if not PYPDF_AVAILABLE:
        return False
    try:
        reader = PdfReader(str(input_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                logger.warning(f"[pypdf] Encrypted PDF (skipping): {input_path.name}")
                return False
        writer = PdfWriter()
        for p in reader.pages:
            writer.add_page(p)
        writer.compress_content_streams()
        with open(output_path, "wb") as f:
            writer.write(f)
        return True
    except Exception as e:
        logger.error(f"[pypdf] Failed {input_path}: {e}")
        return False


def compress_pdf(input_path: Path, final_dest: Path, pdf_cfg: PDFConfig, min_reduction_pct: float, dry_run: bool, backup: bool, preserve_times: bool) -> Dict[str, Any]:
    orig_size = input_path.stat().st_size
    tmp_out = safe_temp_path(final_dest)
    ensure_parent(tmp_out)

    method = pdf_cfg.method
    if method == "auto":
        method = "ghostscript" if GS_BIN else ("pypdf" if PYPDF_AVAILABLE else "none")

    ok = False
    if not dry_run:
        if method == "ghostscript":
            ok = compress_pdf_with_gs(input_path, tmp_out, pdf_cfg)
        elif method == "pypdf":
            ok = compress_pdf_with_pypdf(input_path, tmp_out)
        else:
            return {"success": False, "reason": "No available PDF compression backend"}
    else:
        ok = True
        est_size = int(orig_size * 0.7)  # a bit more optimistic in plan mode
        with open(tmp_out, "wb") as f:
            f.truncate(est_size)

    if not ok or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        return {"success": False, "reason": "Compression failed or empty output"}

    new_size = tmp_out.stat().st_size
    reduction = (1 - new_size / max(1, orig_size)) * 100

    action = "kept_original"
    if reduction >= max(0.0, min_reduction_pct):
        action = "compressed"
        if not dry_run:
            if backup:
                bak = final_dest.with_suffix(final_dest.suffix + ".bak")
                try:
                    shutil.copy2(input_path, bak)
                except Exception:
                    logger.warning(f"Could not write backup: {bak}")
            shutil.move(str(tmp_out), str(final_dest))
            if preserve_times:
                copy_stat(input_path, final_dest)
        else:
            tmp_out.unlink(missing_ok=True)
    else:
        if not dry_run:
            tmp_out.unlink(missing_ok=True)
            if final_dest != input_path and not final_dest.exists():
                ensure_parent(final_dest)
                shutil.copy2(input_path, final_dest)
                if preserve_times:
                    copy_stat(input_path, final_dest)
        else:
            tmp_out.unlink(missing_ok=True)
        action = "copied_original" if final_dest != input_path else "kept_original"

    return {
        "success": True,
        "type": "pdf",
        "method": method,
        "action": action,
        "original_size": orig_size,
        "final_size": new_size if action == "compressed" else orig_size,
        "reduction_pct": reduction if action == "compressed" else 0.0,
    }

# --------------------------- PPTX Compress ----------------------------

def compress_pptx(input_path: Path, final_dest: Path, pptx_cfg: PPTXConfig, min_reduction_pct: float, dry_run: bool, backup: bool, preserve_times: bool, ocr_safe: bool = False) -> Dict[str, Any]:
    if not PILLOW_AVAILABLE:
        return {"success": False, "reason": "Pillow not installed", "type": "pptx"}

    orig_size = input_path.stat().st_size
    tmp_out = safe_temp_path(final_dest)
    ensure_parent(tmp_out)

    if dry_run:
        est_size = int(orig_size * 0.85)
        with open(tmp_out, "wb") as f:
            f.truncate(est_size)
    else:
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                shutil.unpack_archive(str(input_path), str(td_path), format="zip")
                media = td_path / "ppt" / "media"
                if media.exists():
                    for img_path in media.iterdir():
                        if not img_path.is_file():
                            continue
                        ext = img_path.suffix.lower()
                        if ext not in {".jpg", ".jpeg", ".png"}:
                            continue
                        try:
                            with Image.open(img_path) as im:
                                if ext in {".jpg", ".jpeg"}:
                                    mode = "RGB" if im.mode in ("RGBA", "P", "LA") else im.mode
                                    im = im.convert(mode)
                                    save_kwargs = dict(format="JPEG", optimize=True, progressive=True)
                                    q = 95 if ocr_safe else pptx_cfg.image_quality
                                    try:
                                        im.save(img_path, quality=q, subsampling=0 if ocr_safe else "keep", **save_kwargs)
                                    except TypeError:
                                        im.save(img_path, quality=q, **save_kwargs)
                                elif ext == ".png":
                                    if ocr_safe:
                                        if im.mode in ("P", "LA"):
                                            im = im.convert("RGBA") if "A" in im.mode else im.convert("RGB")
                                        im.save(img_path, format="PNG", optimize=True, compress_level=9)
                                    else:
                                        has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in im.info)
                                        if not has_alpha and im.mode not in ("P",):
                                            try:
                                                im8 = im.convert("P", palette=Image.ADAPTIVE, colors=256)
                                                im8.save(img_path, format="PNG", optimize=True, compress_level=9)
                                            except Exception:
                                                im.save(img_path, format="PNG", optimize=True, compress_level=9)
                                        else:
                                            im.save(img_path, format="PNG", optimize=True, compress_level=9)
                        except Exception as e:
                            logger.debug(f"PPTX image skipped {img_path.name}: {e}")

                base_no_ext = tmp_out.with_suffix("")
                archive_path = shutil.make_archive(str(base_no_ext), "zip", str(td_path))
                if tmp_out.exists():
                    tmp_out.unlink(missing_ok=True)
                Path(archive_path).rename(tmp_out)
        except Exception as e:
            logger.error(f"PPTX compress failed {input_path.name}: {e}")
            return {"success": False, "reason": str(e), "type": "pptx"}

    if not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        return {"success": False, "reason": "Empty output", "type": "pptx"}

    new_size = tmp_out.stat().st_size
    reduction = (1 - new_size / max(1, orig_size)) * 100

    action = "kept_original"
    if reduction >= max(0.0, min_reduction_pct):
        action = "compressed"
        if not dry_run:
            if backup:
                bak = final_dest.with_suffix(final_dest.suffix + ".bak")
                try:
                    shutil.copy2(input_path, bak)
                except Exception:
                    logger.warning(f"Could not write backup: {bak}")
            shutil.move(str(tmp_out), str(final_dest))
            if preserve_times:
                copy_stat(input_path, final_dest)
        else:
            tmp_out.unlink(missing_ok=True)
    else:
        if not dry_run:
            tmp_out.unlink(missing_ok=True)
            if final_dest != input_path and not final_dest.exists():
                ensure_parent(final_dest)
                shutil.copy2(input_path, final_dest)
                if preserve_times:
                    copy_stat(input_path, final_dest)
        else:
            tmp_out.unlink(missing_ok=True)
        action = "copied_original" if final_dest != input_path else "kept_original"

    return {
        "success": True,
        "type": "pptx",
        "action": action,
        "original_size": orig_size,
        "final_size": new_size if action == "compressed" else orig_size,
        "reduction_pct": reduction if action == "compressed" else 0.0,
    }

# --------------------------- Batch Engine -----------------------------

def gather_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if not p.exists():
            logger.warning(f"Path does not exist, skipping: {p}")
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
        elif p.is_dir():
            for ext in SUPPORTED_EXTS:
                files.extend(p.rglob(f"*{ext}"))
    uniq = sorted(set(files), key=lambda x: str(x).lower())
    return uniq


def process_one(path: Path, run_cfg: RunConfig, pdf_cfg: PDFConfig, pptx_cfg: PPTXConfig) -> Dict[str, Any]:
    try:
        ext = path.suffix.lower()
        if path.stat().st_size < run_cfg.min_size_mb * 1024 * 1024:
            return {"success": True, "action": "skipped_small", "original_size": path.stat().st_size, "final_size": path.stat().st_size, "type": ext[1:], "file": str(path)}

        out_path = compute_out_path(path, run_cfg)

        if ext == ".pdf":
            res = compress_pdf(path, out_path, pdf_cfg, run_cfg.min_reduction_pct, run_cfg.dry_run, run_cfg.backup, run_cfg.preserve_times)
        elif ext == ".pptx":
            res = compress_pptx(path, out_path, pptx_cfg, run_cfg.min_reduction_pct, run_cfg.dry_run, run_cfg.backup, run_cfg.preserve_times, ocr_safe=pdf_cfg.ocr_safe)
        else:
            return {"success": False, "reason": "unsupported", "file": str(path)}

        res.update({"file": str(path), "dest": str(out_path)})
        return res
    except Exception as e:
        return {"success": False, "reason": str(e), "file": str(path)}


def write_reports(results: List[Dict[str, Any]], run_cfg: RunConfig) -> None:
    if run_cfg.report_json:
        ensure_parent(run_cfg.report_json)
        with open(run_cfg.report_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if run_cfg.report_csv:
        ensure_parent(run_cfg.report_csv)
        fields = sorted({k for r in results for k in r.keys()})
        with open(run_cfg.report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)

# ------------------------------ CLI -----------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compress PDF and PPTX with OCR‑safe and mirror‑preserving options.\n"
            "Typed docs get stronger compression; image‑heavy/handwritten stay OCR‑friendly."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("input", type=str, help="File or directory to process recursively.")
    p.add_argument("--output", type=str, default=None, help="Mirror outputs into this folder; originals untouched.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2), help="Parallel workers (default: ~half cores).")
    p.add_argument("--min-size-mb", type=float, default=0.5, help="Skip files smaller than this many MB (default 0.5).")
    p.add_argument("--min-reduction", type=float, default=1.0, help="Only accept output if reduction ≥ this percent.")
    p.add_argument("--backup", action="store_true", help="When operating in place, write a .bak before replacing.")
    p.add_argument("--no-preserve-times", dest="preserve_times", action="store_false", help="Do not preserve timestamps on outputs.")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not write outputs.")
    p.add_argument("--report-json", type=str, default=None, help="Write JSON report to this path.")
    p.add_argument("--report-csv", type=str, default=None, help="Write CSV report to this path.")

    # PDF options
    p.add_argument("--pdf-method", choices=["ghostscript", "pypdf", "auto"], default="auto", help="PDF backend.")
    p.add_argument("--pdf-resolution", type=int, default=120, help="Base target DPI for raster downsampling (GS).")
    p.add_argument("--pdf-quality-base", type=int, default=85, help="JPEG quality when not image‑heavy (GS).")
    p.add_argument("--pdf-quality-image", type=int, default=90, help="JPEG quality when image‑heavy (GS).")
    p.add_argument("--pdf-image-threshold", type=float, default=0.50, help="Image‑heavy threshold ratio (0‑1).")
    p.add_argument("--pdf-timeout", type=int, default=600, help="Per‑file Ghostscript timeout (s).")
    p.add_argument("--pdf-safe-image-dense", action="store_true", default=True, help="Use safer settings when image‑heavy (default on).")
    p.add_argument("--pdf-no-safe-image-dense", dest="pdf_safe_image_dense", action="store_false", help="Disable image‑heavy DPI floor.")
    p.add_argument("--pdf-image-min-dpi", type=int, default=220, help="Minimum DPI when image‑heavy (safer).")
    p.add_argument("--pdf-auto-filter-images", action="store_true", default=True, help="Let GS auto‑pick image filters (default on).")
    p.add_argument("--pdf-no-auto-filter-images", dest="pdf_auto_filter_images", action="store_false", help="Disable auto filter selection.")
    p.add_argument("--pdf-detect-duplicate-images", action="store_true", default=True, help="Enable duplicate image detection (default on).")
    p.add_argument("--pdf-no-detect-duplicate-images", dest="pdf_detect_duplicate_images", action="store_false", help="Disable duplicate detection.")

    # OCR and classification knobs
    p.add_argument("--ocr-safe", action="store_true", help="Keep OCR‑friendly filters for image‑heavy/handwritten.")
    p.add_argument("--ocr-min-dpi", type=int, default=300, help="DPI floor when --ocr-safe (image‑heavy only).")
    p.add_argument("--typed-ratio-max", type=float, default=0.25, help="≤ this image ratio => treat as typed.")
    p.add_argument("--typed-resolution", type=int, default=110, help="DPI for typed class (aggressive).")
    p.add_argument("--typed-quality", type=int, default=60, help="JPEG quality for typed class (aggressive).")
    p.add_argument("--mixed-resolution", type=int, default=200, help="DPI for mixed class.")
    p.add_argument("--mixed-quality", type=int, default=80, help="JPEG quality for mixed class.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return 2

    output_folder = Path(args.output).resolve() if args.output else None
    if output_folder and not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("--- Dependency Check ---")
    if not GS_BIN:
        logger.warning("Ghostscript not found; PDF compression will use pypdf if available.")
    if not PYPDF_AVAILABLE:
        logger.warning("pypdf not available; cannot fallback for PDFs.")
    if not PDFIMAGES_AVAILABLE:
        logger.warning("pdfimages not found; image‑ratio classification disabled; everything treated as mixed.")
    if not PILLOW_AVAILABLE:
        logger.warning("Pillow not found; PPTX compression disabled.")
    logger.info("------------------------")

    pdf_cfg = PDFConfig(
        resolution=args.pdf_resolution,
        base_quality=args.pdf_quality_base,
        image_heavy_quality=args.pdf_quality_image,
        image_ratio_threshold=args.pdf_image_threshold,
        method=args.pdf_method,
        timeout_s=args.pdf_timeout,
        safe_image_dense=bool(args.pdf_safe_image_dense),
        image_heavy_min_dpi=args.pdf_image_min_dpi,
        auto_filter_images=bool(args.pdf_auto_filter_images),
        detect_duplicate_images=bool(args.pdf_detect_duplicate_images),
        ocr_safe=bool(args.ocr_safe),
        ocr_min_dpi=int(args.ocr_min_dpi),
        typed_ratio_max=float(args.typed_ratio_max),
        typed_resolution=int(args.typed_resolution),
        typed_quality=int(args.typed_quality),
        mixed_resolution=int(args.mixed_resolution),
        mixed_quality=int(args.mixed_quality),
    )

    pptx_cfg = PPTXConfig(image_quality=args.pptx_quality)
    run_cfg = RunConfig(
        input_path=input_path,
        output_folder=output_folder,
        workers=max(1, args.workers),
        min_size_mb=max(0.0, args.min_size_mb),
        min_reduction_pct=max(0.0, args.min_reduction),
        backup=bool(args.backup),
        dry_run=bool(args.dry_run),
        preserve_times=bool(args.preserve_times),
        report_json=Path(args.report_json) if args.report_json else None,
        report_csv=Path(args.report_csv) if args.report_csv else None,
    )

    targets = [t for t in gather_files([input_path]) if t.suffix.lower() in SUPPORTED_EXTS]
    logger.info(f"Found {len(targets)} file(s) to process.")

    results: List[Dict[str, Any]] = []
    if not targets:
        write_reports(results, run_cfg)
        logger.info("Nothing to do.")
        return 0

    try:
        with ThreadPoolExecutor(max_workers=run_cfg.workers) as ex:
            fut2path = {ex.submit(process_one, p, run_cfg, pdf_cfg, pptx_cfg): p for p in targets}
            for fut in as_completed(fut2path):
                p = fut2path[fut]
                try:
                    res = fut.result()
                except Exception as e:  # pragma: no cover
                    res = {"success": False, "reason": str(e), "file": str(p)}
                results.append(res)
                if res.get("success"):
                    action = res.get("action", "?")
                    red = res.get("reduction_pct", 0.0)
                    o = res.get("original_size", 0)
                    fsz = res.get("final_size", 0)
                    logger.info(f"{p.name}: {action} (orig {human(o)} → final {human(fsz)}; -{red:.1f}%)")
                else:
                    logger.error(f"{p.name}: failed ({res.get('reason')})")
    except KeyboardInterrupt:
        logger.error("Interrupted by user (SIGINT). Partial results will be reported.")

    tot = len(results)
    compressed = sum(1 for r in results if r.get("action") == "compressed")
    copied = sum(1 for r in results if r.get("action") in {"copied_original", "kept_original"})
    skipped = sum(1 for r in results if r.get("action") == "skipped_small")
    failed = sum(1 for r in results if not r.get("success"))

    logger.info("\n--- Batch Complete ---")
    logger.info(f"Files total: {tot}")
    logger.info(f"Compressed: {compressed}")
    logger.info(f"Copied/Kept: {copied}")
    logger.info(f"Skipped small: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info("----------------------")

    write_reports(results, run_cfg)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
