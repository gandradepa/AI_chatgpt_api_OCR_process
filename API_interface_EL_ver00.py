import os
import json
import base64
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# --- OCR / image libs ---
import cv2
import numpy as np
from PIL import Image  # noqa: F401
import pytesseract

# NEW: database
import sqlite3
from contextlib import closing

import platform, shutil, pytesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"


# --- [1] Load API key ---
load_dotenv(dotenv_path=r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\OpenAI_key_bryan.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- [2] Paths & constants ---
image_folder  = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\Capture_photos_upload"
output_folder = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\Output_jason_api"
debug_folder  = os.path.join(output_folder, "debug_ubc_tag")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# NEW: DB path (SQLite)
DB_PATH = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\asset_capture_app\data\QR_codes.db"
DB_TABLE = "QR_codes"

VALID_SUFFIXES = {"0", "1", "3"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FIELD_SOURCES: Dict[str, List[str]] = {
    "Manufacturer": ["0"],
    "Model": ["0"],
    "Serial Number": ["0"],
    "Year": ["0"],
    "UBC Tag": ["1"],
    "Technical Safety BC": ["3"],
}

UBC_TAG_PATTERN = re.compile(r"\b([A-Z]{1,3})[-\u2013]?\s?(\d{1,4})([A-Z]?)\b")

# --- [2.1] DB helpers (NEW) ---
def load_approved_qrs(db_path: str, table: str = "QR_codes") -> set:
    approved = set()
    if not os.path.exists(db_path):
        print(f"⚠ DB not found: {db_path}. Proceeding without approval filter.")
        return approved
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with closing(conn.cursor()) as cur:
                cur.execute(f"""
                    SELECT QR_code_ID
                    FROM {table}
                    WHERE (Approved = 1 OR Approved = '1')
                """)
                for row in cur.fetchall():
                    qrid = str(row["QR_code_ID"]).strip()
                    if qrid:
                        approved.add(qrid)
    except Exception as e:
        print(f"⚠ Error reading approvals from DB: {e}. Proceeding without approval filter.")
    return approved

APPROVED_QRS = load_approved_qrs(DB_PATH, DB_TABLE)
if APPROVED_QRS:
    print(f"Approval filter loaded: {len(APPROVED_QRS)} QR(s) will be skipped.")

# --- [3] Group files by QR ---
# Filename pattern: "<QR> <Building> EL - <Sequence>"
pattern = re.compile(
    r"^(\d+)\s+"
    r"(\d+(?:-\d+)?)\s+"
    r"(EL)\s*-\s*([013])$",
    re.IGNORECASE
)

grouped = defaultdict(lambda: {"images": {}, "building": "", "asset_type": "EL"})

for fn in os.listdir(image_folder):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in VALID_EXTS:
        continue
    m = pattern.match(base)
    if not m:
        print(f"⚠ Skipping unrecognized filename: {fn}")
        continue
    qr, building, asset_type, seq = m.groups()
    if seq not in VALID_SUFFIXES or asset_type.upper() != "EL":
        continue

    # NEW: skip if Approved=1
    if qr in APPROVED_QRS:
        print(f"⏭️  Skipping QR {qr} (Approved=1 in DB)")
        continue

    grouped[qr]["building"] = building
    grouped[qr]["images"][seq] = os.path.join(image_folder, fn)

print(f"\nTotal assets found (after approval filter): {len(grouped)}")

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

def normalize_year(value: str) -> str:
    if not value:
        return ""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    return m.group(0) if m else ""

def canonicalize_ubc_tag(text: str) -> str:
    if not text:
        return ""
    m = UBC_TAG_PATTERN.search(text.replace("—", "-").replace("–", "-"))
    if not m:
        return ""
    left, num, suffix = m.groups()
    return f"{left}-{num}{suffix}".strip("-")

# (OCR pipeline functions permanecem iguais...)

# --- [6] Process each asset ---
for qr, info in grouped.items():
    # Double-guard
    if qr in APPROVED_QRS:
        print(f"⏭️  Skipping QR {qr} (Approved=1 in DB)")
        continue

    print(f"\nProcessing QR {qr} …")

    result = {
        "Manufacturer": "",
        "Model": "",
        "Serial Number": "",
        "Year": "",
        "UBC Tag": "",
        "Technical Safety BC": ""
    }

    for seq, path in info["images"].items():
        fields_for_seq = [f for f, srcs in FIELD_SOURCES.items() if seq in srcs]
        if not fields_for_seq:
            continue

        if seq == "1" and "UBC Tag" in fields_for_seq:
            img_bgr = cv2.imread(path)
            tag_text, mean_conf = tesseract_read_tag(img_bgr)

            if not tag_text or mean_conf < 65.0:
                model_tag = ask_model_for_ubc_tag(path)
                tag_final = canonicalize_ubc_tag(tag_text) or canonicalize_ubc_tag(model_tag)
            else:
                tag_final = tag_text

            result["UBC Tag"] = tag_final or ""
            print(f"  → UBC Tag (seq 1): '{result['UBC Tag']}' (tesseract_conf≈{mean_conf:.1f})")
            continue

        partial = ask_model_for_fields(path, fields_for_seq)
        if "Year" in partial:
            partial["Year"] = normalize_year(partial.get("Year", ""))

        for k, v in partial.items():
            if isinstance(v, str):
                result[k] = v.strip()

    output_data = {
        "qr_code":         qr,
        "building_number": info.get("building", ""),
        "asset_type":      f"- {info.get('asset_type', 'EL').upper()}",
        "structured_data": result
    }

    json_filename = f"{qr}_{info.get('asset_type', 'EL').upper()}_{info.get('building', '')}.json"
    out_path = os.path.join(output_folder, json_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {out_path}")
