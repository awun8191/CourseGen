import os
from PIL import Image

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')

def collect_images(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if file.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, file))
    return images

def convert_images_to_pdf(image_paths, output_path):
    if not image_paths:
        print("❌ No images found.")
        return False

    image_list = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            image_list.append(img)
        except Exception as e:
            print(f"⚠️ Failed to load image: {img_path} | {e}")

    if not image_list:
        print("❌ No valid images to convert.")
        return False

    first_image = image_list[0]
    rest_images = image_list[1:]

    print(f"📦 Saving {len(image_list)} images to {output_path}...")
    first_image.save(output_path, save_all=True, append_images=rest_images)
    print("✅ PDF saved successfully!")
    return True

def delete_images(image_paths):
    print("🧹 Deleting original images...")
    for path in image_paths:
        try:
            os.remove(path)
            print(f"🗑️ Deleted {path}")
        except Exception as e:
            print(f"⚠️ Failed to delete {path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compile all images in a folder into a single PDF.")
    parser.add_argument("directory", help="Path to the folder containing images.")
    parser.add_argument("--keep", action="store_true", help="Keep images after PDF conversion (default: False)")
    parser.add_argument("--filename", help="The name of the file you want to save as pdf")
    args = parser.parse_args()

    abs_dir = os.path.abspath(args.directory)
    output_pdf_path = os.path.join(abs_dir, f"{args.filename}.pdf")

    images = collect_images(abs_dir)

    if convert_images_to_pdf(images, output_pdf_path):
        if not args.keep:
            delete_images(images)

if __name__ == "__main__":
    main()
