#!/usr/bin/env python3
import os, sys, cv2, fitz, numpy as np, argparse, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Try optional import
try:
    from realesrgan import RealESRGAN
    _HAS_REALESRGAN = True
except ImportError:
    _HAS_REALESRGAN = False


# ---------- Utility ----------
def log_error(log_path, msg):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def unsharp(image, amount=1.5, sigma=1.0):
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)


def adaptive_bin(image, block=31, C=10):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, C)


def deskew(image):
    coords = np.column_stack(np.where(image < 255))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def clahe_luminance(img, clip=2.0, tiles=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------- Page Classification ----------
def classify_page(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    if edge_density < 0.01 and std_intensity > 40:
        return "image", {"edge_density": edge_density, "contrast": std_intensity}
    elif edge_density > 0.08 and mean_intensity > 120:
        return "printed", {"edge_density": edge_density, "contrast": std_intensity}
    else:
        return "handwriting", {"edge_density": edge_density, "contrast": std_intensity}


# ---------- Upscaler ----------
class Upscaler:
    def __init__(self, use_realesrgan, sr_scale, device):
        self.use_realesrgan = bool(use_realesrgan) and _HAS_REALESRGAN
        self.sr_scale = 4 if int(sr_scale) >= 4 else 2
        self.device = device
        self._sr = None
        if self.use_realesrgan:
            try:
                self._sr = RealESRGAN(device, scale=self.sr_scale)
                self._sr.load_weights(f"RealESRGAN_x{self.sr_scale}.pth", download=True)
            except Exception:
                self.use_realesrgan = False
                self._sr = None

    def upscale(self, bgr, factor: float, *, force_sr: bool = False):
        if force_sr and self.use_realesrgan and self._sr is not None:
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out = self._sr.predict(pil)
            return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        fx = max(1.0, float(factor))
        return cv2.resize(bgr, (int(w * fx), int(h * fx)), interpolation=cv2.INTER_LANCZOS4)


# ---------- Enhancement ----------
def enhance_page(bgr, upscaler, base_scale, allow_deskew=True, sr_images_only=True):
    label, feats = classify_page(bgr)

    if label == "printed":
        img = upscaler.upscale(bgr, max(2.5, base_scale), force_sr=False)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        g = unsharp(g, amount=1.5, sigma=1.0)
        out = adaptive_bin(g)
        if allow_deskew: out = deskew(out)
        return out, label, feats

    elif label == "handwriting":
        img = upscaler.upscale(bgr, max(3.0, base_scale), force_sr=False)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = unsharp(g, amount=0.9, sigma=1.2)
        if allow_deskew: g = deskew(g)
        return g, label, feats

    else:  # image
        img = upscaler.upscale(bgr, max(3.0, base_scale), force_sr=sr_images_only)
        img = clahe_luminance(img)
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        sharp = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.25, sharp, -0.25, 0)
        if allow_deskew: img = deskew(img)
        return img, label, feats


# ---------- PDF Processing ----------
def process_pdf(pdf_path, out_dir, args, upscaler, log_path):
    try:
        doc = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).stem
        pdf_out_dir = Path(out_dir) / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img, label, feats = enhance_page(
                img, upscaler, args.scale,
                allow_deskew=True,
                sr_images_only=args.sr_images_only
            )
            out_path = pdf_out_dir / f"{pdf_name}_page{i+1}.{args.pdf_image_format}"
            if len(img.shape) == 2:
                cv2.imwrite(str(out_path), img)
            else:
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        log_error(log_path, f"Error processing {pdf_path}: {traceback.format_exc()}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="Folder with PDFs")
    ap.add_argument("output_dir", help="Folder for upscaled images")
    ap.add_argument("--scale", type=float, default=3.0, help="Base scaling factor")
    ap.add_argument("--pdf-image-format", type=str, default="png", help="Image format for output pages")
    ap.add_argument("--log-file", type=str, default="errors.log", help="Log file for errors")
    ap.add_argument("--realesrgan", action="store_true", help="Enable Real-ESRGAN")
    ap.add_argument("--sr-scale", type=int, default=4, help="Real-ESRGAN upscale factor")
    ap.add_argument("--sr-images-only", action="store_true", help="Use SR only for image-heavy pages")
    ap.add_argument("--workers", type=int, default=2, help="Threads to use")
    ap.add_argument("--device", type=str, default="cuda", help="Device for Real-ESRGAN (cuda/cpu)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    upscaler = Upscaler(args.realesrgan, args.sr_scale, args.device)

    pdf_files = [str(p) for p in Path(args.input_dir).rglob("*.pdf")]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pdf in pdf_files:
            ex.submit(process_pdf, pdf, args.output_dir, args, upscaler, args.log_file)


if __name__ == "__main__":
    main()
