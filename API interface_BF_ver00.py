import os
import json
import base64
import re
import time
from collections import defaultdict
from typing import Dict, List
from difflib import get_close_matches

from dotenv import load_dotenv
from openai import OpenAI

# --- OCR / image libs ---
import cv2
import numpy as np
from PIL import Image  # noqa: F401
import pytesseract

# Point pytesseract to your installed Tesseract OCR exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\gandrade\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# --- [1] Load API key ---
load_dotenv(dotenv_path=r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\OpenAI_key_bryan.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- [2] Paths & constants ---
image_folder  = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\Capture_photos_upload"
output_folder = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\Output_jason_api"
os.makedirs(output_folder, exist_ok=True)

VALID_SUFFIXES = {"0", "1", "3"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VALID_MANUFACTURERS = ["Watts", "Wilkins", "Conbraco", "Apollo"]

# Extract these from BOTH 0 and 1 sequences
FIELD_SOURCES: Dict[str, List[str]] = {
    "Manufacturer": ["0", "1"],
    "Model": ["0", "1"],
    "Serial Number": ["0", "1"],
    "Diameter": ["0", "1"],
}

# --- [3] Group files by QR ---
pattern = re.compile(
    r"^(\d+)\s+"
    r"(\d+(?:-\d+)?)\s+"
    r"(BF)\s*-\s*([013])$",
    re.IGNORECASE
)

grouped = defaultdict(lambda: {"images": {}, "building": "", "asset_type": "BF"})

for fn in os.listdir(image_folder):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in VALID_EXTS:
        continue
    m = pattern.match(base)
    if not m:
        print(f"⚠ Skipping unrecognized filename: {fn}")
        continue
    qr, building, asset_type, seq = m.groups()
    if seq not in VALID_SUFFIXES or asset_type.upper() != "BF":
        continue
    grouped[qr]["building"] = building
    grouped[qr]["images"][seq] = os.path.join(image_folder, fn)

print(f"\nTotal assets found: {len(grouped)}")

# --- [4] Utilities ---

def encode_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".bmp":  "image/bmp",
        ".webp": "image/webp"
    }.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def normalize_manufacturer(value: str) -> str:
    """Force manufacturer to one of the allowed values or return empty string."""
    if not value:
        return ""
    match = get_close_matches(value.strip().title(), VALID_MANUFACTURERS, n=1, cutoff=0.6)
    return match[0] if match else ""

# Diameter normalization:
# Accept forms like: 2", 1/2", 1-1/2", 1 1/4", 1.25", 3 in, 3 inches
DIAMETER_PATTERNS = [
    r'\b\d{1,2}\s*-\s*\d/\d\s*["”]?',      # 1-1/2"
    r'\b\d{1,2}\s+\d/\d\s*["”]?',          # 1 1/2"
    r'\b\d{1,2}/\d\s*["”]?',               # 1/2"
    r'\b\d{1,2}(?:\.\d{1,2})?\s*["”]?',    # 1", 1.25"
    r'\b\d{1,2}(?:\s*(?:in|inch|inches))\b', # 3 in / 3 inch
]

def normalize_diameter(val: str) -> str:
    if not val:
        return ""
    t = val.strip().replace("”", '"').replace("“", '"').replace("''", '"')
    t = t.replace("INCHES", 'in').replace("INCH", 'in').replace("IN.", 'in').replace("IN ", 'in ')
    # Try to find a known pattern
    for pat in DIAMETER_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            s = m.group(0).strip()
            s = s.replace(" ", "")
            s = s.replace("INCHES", "in").replace("INCH", "in").replace("IN.", "in").replace("IN", "in")
            # Ensure canonical: end with "
            if s.lower().endswith("in"):
                s = s[:-2] + '"'
            elif not s.endswith('"'):
                s = s + '"'
            # Clean up dash spacing
            s = s.replace("--", "-").replace("-'", '-"')
            s = s.replace('-"', '-"')  # already correct
            # Allow forms like 1-1/2", 1/2", 2"
            return s
    # Last try: pure number -> add quote
    m2 = re.search(r'\b\d{1,2}(?:\.\d{1,2})?\b', t)
    if m2:
        return m2.group(0) + '"'
    return ""

def ocr_find_diameter(image_path: str) -> str:
    """OCR fallback to locate a diameter-like pattern on the image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
        text = text.replace("\n", " ")
        return normalize_diameter(text)
    except Exception:
        return ""

def ask_model_for_fields(image_path: str, fields: List[str]) -> dict:
    fields_list = "\n".join([f"- {f}" for f in fields])
    prompt = f"""
You will see ONE image. Extract ONLY the requested fields below.
Return a STRICT JSON object with EXACT keys, using empty string if missing/unclear.

For the "Manufacturer" field, only use one of the following values:
{", ".join(VALID_MANUFACTURERS)}
If the image shows another manufacturer, set "Manufacturer" to "".

For the "Diameter" field, return the size in inches with a trailing double quote.
Acceptable forms: 2", 1/2", 1-1/2", 1 1/4", 1.25".
If unclear, return "".

{fields_list}

Do not include any text before or after the JSON.
""".strip()

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=300
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            data = json.loads(raw)

            # Enforce allowed manufacturer set
            if "Manufacturer" in data:
                data["Manufacturer"] = normalize_manufacturer(str(data.get("Manufacturer", "")))

            # Normalize diameter form
            if "Diameter" in data:
                data["Diameter"] = normalize_diameter(str(data.get("Diameter", "")))

            return {k: (data.get(k, "") if isinstance(data.get(k, ""), str) else "") for k in fields}
        except Exception:
            time.sleep(1.5 * (attempt + 1))

    return {k: "" for k in fields}

# --- [5] Process each asset ---
for qr, info in grouped.items():
    print(f"\nProcessing QR {qr} …")
    result = {
        "Manufacturer": "",
        "Model": "",
        "Serial Number": "",
        "Diameter": ""
    }

    # Try sequences 0 and 1; last non-empty wins (so 1 can fill gaps left by 0)
    for seq, path in info["images"].items():
        fields_for_seq = [f for f, srcs in FIELD_SOURCES.items() if seq in srcs]
        if not fields_for_seq:
            continue

        partial = ask_model_for_fields(path, fields_for_seq)

        # If Diameter is still empty from GPT, try OCR fallback on this image
        if "Diameter" in partial and not partial["Diameter"]:
            partial["Diameter"] = ocr_find_diameter(path)

        for k, v in partial.items():
            if isinstance(v, str) and v.strip():
                result[k] = v.strip()  # latest non-empty value overwrites

    output_data = {
        "qr_code":         qr,
        "building_number": info.get("building", ""),
        "asset_type":      f"- {info.get('asset_type', 'BF').upper()}",
        "structured_data": result
    }

    json_filename = f"{qr}_{info.get('asset_type', 'BF').upper()}_{info.get('building', '')}.json"
    out_path = os.path.join(output_folder, json_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved {out_path}")
