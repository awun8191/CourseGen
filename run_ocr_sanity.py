from paddleocr import PaddleOCR
import sys
img = sys.argv[1] if len(sys.argv)>1 else "/home/raregazetto/Pictures/Screenshots/Screenshot From 2025-09-01 23-50-12.png"
ocr = PaddleOCR(lang="en", use_textline_orientation=True)
res = ocr.ocr(img, det=True, rec=True, cls=True)
print(res[:1])
