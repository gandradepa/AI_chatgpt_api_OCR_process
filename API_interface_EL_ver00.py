#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import shutil
import sqlite3
import platform
from typing import Dict, List, Tuple, Optional
from contextlib import closing
from collections import defaultdict

# OCR / image libs
import cv2
import numpy as np
from PIL import Image  # noqa: F401
import pytesseract

# OpenAI
from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# CONFIG
# =============================================================================

# Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"

# OpenAI key
load_dotenv(dotenv_path=r"/home/developer/API/OpenAI_key_bryan.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
IMAGE_DIR   = r"/home/developer/Capture_photos_upload"
OUTPUT_DIR  = r"/home/developer/Output_jason_api"
DEBUG_DIR   = os.path.join(OUTPUT_DIR, "debug_el")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# DB: use sdi_dataset_EL (Approved rule lives here)
DB_PATH  = r"/home/developer/asset_capture_app_dev/data/QR_codes.db"
DB_TABLE = "sdi_dataset_EL"

# Accept EL - 0 / 1 / 2
VALID_SUFFIXES = {"0", "1", "2"}
VALID_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Crop fraction for header on EL - 0 (top 25%)
HEADER_FRACTION = 0.25

# UBC Tag must contain at least one digit (avoid false positives like "LO OY")
UBC_TAG_CANDIDATE = re.compile(r"\b([A-Z]{2,5})[ -]?(?=[A-Z0-9]*\d)[A-Z0-9]{2,12}\b")

# =============================================================================
# HELPERS
# =============================================================================

def _normalize_qr(s: str) -> str:
    """Normalize QR for comparison by stripping leading zeros (keep only digits up front)."""
    s = str(s).strip()
    m = re.match(r"\d+", s)
    if not m:
        return s
    core = m.group(0).lstrip("0")
    return core or "0"

def _detect_columns(conn: sqlite3.Connection, table: str) -> Tuple[str, str]:
    """
    Find the QR code column and Approved column in `table`.
    Prefer: "QR Code", fallback: "QR_code_ID", etc. Approved column must be "Approved".
    Returns exact names from DB: (qr_col, approved_col).
    """
    wanted_qr_names = {"QR Code", "QR_code_ID", "QRCode", "QR", "QR_code"}
    wanted_approved_names = {"Approved"}

    cols = {}
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    for _, name, *_ in cur.fetchall():
        cols[name.lower()] = name

    qr_col = None
    for cand in wanted_qr_names:
        if cand.lower() in cols:
            qr_col = cols[cand.lower()]
            break
    if not qr_col:
        raise RuntimeError(f"Could not find QR column in {table}. Tried {sorted(wanted_qr_names)}")

    approved_col = None
    for cand in wanted_approved_names:
        if cand.lower() in cols:
            approved_col = cols[cand.lower()]
            break
    if not approved_col:
        raise RuntimeError(f"Could not find 'Approved' column in {table}")

    return qr_col, approved_col

def load_eligible_qrs(db_path: str, table: str) -> set:
    """
    Return set of QRs (normalized) that MUST be processed:
    rows where Approved == 1 or '1' in sdi_dataset_EL.
    """
    eligible = set()
    if not os.path.exists(db_path):
        print(f"❌ DB not found: {db_path}. No items will be processed.")
        return eligible

    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            qr_col, approved_col = _detect_columns(conn, table)
            sql = f'''
                SELECT "{qr_col}" AS qr
                FROM "{table}"
                WHERE {approved_col} = 1 OR {approved_col} = '1'
            '''
            for row in conn.execute(sql):
                qr_raw = str(row["qr"]).strip()
                if qr_raw:
                    eligible.add(_normalize_qr(qr_raw))
    except Exception as e:
        print(f"❌ Error reading approvals from DB ({table}): {e}")
        return set()

    return eligible

def encode_image_from_path(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"❌ encode_image_from_path({image_path}) failed: {e}")
        return None

def encode_image_from_ndarray(img: np.ndarray) -> Optional[str]:
    try:
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"❌ encode_image_from_ndarray failed: {e}")
        return None

def crop_header_top(image_path: str, fraction: float = HEADER_FRACTION) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        h = img.shape[0]
        crop_h = max(1, int(h * fraction))
        header = img[:crop_h, :, :]
        # save debug
        debug_name = os.path.join(DEBUG_DIR, f"header_{os.path.basename(image_path)}")
        cv2.imwrite(debug_name, header)
        return header
    except Exception as e:
        print(f"❌ crop_header_top({image_path}) failed: {e}")
        return None

def quick_ocr_text(img_path: str) -> str:
    try:
        img = cv2.imread(img_path)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        # optional: slight dilation can help
        text = pytesseract.image_to_string(gray, config="--psm 6")
        return text or ""
    except Exception as e:
        print(f"❌ quick_ocr_text failed on {img_path}: {e}")
        return ""

def find_ubc_tag_hint_from_el1(img_path: Optional[str]) -> str:
    if not img_path:
        return ""
    text = quick_ocr_text(img_path).upper()
    # Look for a candidate with at least one digit
    match = UBC_TAG_CANDIDATE.search(text)
    return match.group(0) if match else ""

# =============================================================================
# LOAD ELIGIBLE QRS
# =============================================================================

ELIGIBLE_QRS = load_eligible_qrs(DB_PATH, DB_TABLE)
print(f"Approval filter loaded from {DB_TABLE}: {len(ELIGIBLE_QRS)} QR(s) will be PROCESSED.")

# =============================================================================
# GROUP FILES BY QR
# Filename pattern: "<QR> <Building> EL - <0|1|2>.<ext>"
# =============================================================================

pattern = re.compile(
    r"^(\d+)\s+"
    r"(\d+(?:-\d+)?)\s+"
    r"(EL)\s*-\s*([012])$",
    re.IGNORECASE
)

groups: Dict[str, Dict] = defaultdict(lambda: {"building": "", "images": {}, "asset_type": "EL"})

for fn in os.listdir(IMAGE_DIR):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in VALID_EXTS:
        continue
    m = pattern.match(base)
    if not m:
        continue

    qr, building, asset_type, seq = m.groups()
    if seq not in VALID_SUFFIXES or asset_type.upper() != "EL":
        continue

    # Only process Approved=1
    if ELIGIBLE_QRS and _normalize_qr(qr) not in ELIGIBLE_QRS:
        continue

    groups[qr]["building"] = building
    groups[qr]["images"][seq] = os.path.join(IMAGE_DIR, fn)

print(f"Total assets to process (Approved=1): {len(groups)}")

# =============================================================================
# MAIN LOOP
# =============================================================================

for qr, info in groups.items():
    building = info.get("building", "")
    paths = info.get("images", {})
    print(f"\n📦 Processing QR {qr} (Building {building}) ...")

    # Prepare image payloads in recommended order:
    # 1) Header crop from EL - 0 (for Branch Panel, Supply From, Volts, Location, Ampere)
    # 2) EL - 1 (UBC Asset Tag)
    # 3) EL - 2 (spare / additional context)
    image_messages: List[Dict] = []

    # Header crop from EL - 0
    el0_path = paths.get("0")
    if el0_path:
        header_img = crop_header_top(el0_path, HEADER_FRACTION)
        if header_img is not None:
            enc = encode_image_from_ndarray(header_img)
            if enc:
                image_messages.append({"type": "image_url", "image_url": {"url": enc}})
        else:
            # fallback: send full image if crop failed
            enc0 = encode_image_from_path(el0_path)
            if enc0:
                image_messages.append({"type": "image_url", "image_url": {"url": enc0}})

    # EL - 1 (UBC Tag plate)
    el1_path = paths.get("1")
    if el1_path:
        enc1 = encode_image_from_path(el1_path)
        if enc1:
            image_messages.append({"type": "image_url", "image_url": {"url": enc1}})

    # EL - 2 (optional)
    el2_path = paths.get("2")
    if el2_path:
        enc2 = encode_image_from_path(el2_path)
        if enc2:
            image_messages.append({"type": "image_url", "image_url": {"url": enc2}})

    if not image_messages:
        print(f"⚠️ No usable images for QR {qr}. Skipping.")
        continue

    # UBC Tag hint from EL - 1 OCR (helps the model, still validated later)
    ubc_tag_hint = find_ubc_tag_hint_from_el1(el1_path)

    # Prompt (EL-specific)
    prompt = f"""
You will see up to three images for an ELECTRICAL PANEL:

• First image: a HEADER CROP from the panel schedule (EL - 0) — extract from the HEADER ONLY.
• Second image: UBC Asset Tag label (EL - 1) — primary source for the UBC Asset Tag.
• Third image: optional context (EL - 2).

Extract ONLY these fields (use exact field names):

- Description
- UBC Asset Tag
- Branch Panel
- Ampere
- Supply From
- Volts
- Location

Rules:
1) "UBC Asset Tag": take from EL - 1. If not found or it contains no digits, LEAVE IT EMPTY for now.
2) "Branch Panel", "Ampere", "Supply From", "Volts", "Location": read from the HEADER of EL - 0 (top of the document).
3) "Description" must be: "Panel - <UBC Asset Tag>" AFTER any fallback is applied.
4) If a value is missing or unreadable, return an empty string.
5) Return a STRICT JSON object – no markdown, no commentary.

Assistive hint (may be empty): UBC Asset Tag hint = "{ubc_tag_hint}"
    """.strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + image_messages}],
            max_tokens=800,
            temperature=0.1,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI error on QR {qr}: {e}")
        continue

    # Save raw
    raw_path = os.path.join(OUTPUT_DIR, f"{qr}_raw_ocr.txt")
    with open(raw_path, "w", encoding="utf-8") as rf:
        rf.write(reply)

    # Strip codefences if present
    if reply.startswith("```json"):
        reply = reply[7:].strip()
    elif reply.startswith("```"):
        reply = reply[3:].strip()
    if reply.endswith("```"):
        reply = reply[:-3].strip()

    # Expected fields
    fields = ["Description", "UBC Asset Tag", "Branch Panel", "Ampere", "Supply From", "Volts", "Location"]
    data: Dict[str, str] = {k: "" for k in fields}

    # Parse JSON
    parsed = None
    try:
        parsed = json.loads(reply)
    except Exception:
        # keep parsed=None; we'll output raw_response
        pass

    if isinstance(parsed, dict):
        for f in fields:
            for k in parsed.keys():
                if k.strip().lower() == f.lower():
                    data[f] = str(parsed[k]).strip()
    else:
        data["raw_response"] = reply

    # Post-processing:
    # Ensure UBC Asset Tag contains a digit; else empty (we fallback to Branch Panel next)
    ubc_val = data.get("UBC Asset Tag", "").upper().strip()
    if not UBC_TAG_CANDIDATE.search(ubc_val):
        ubc_val = ""

    # If missing, fallback to Branch Panel
    branch_panel = data.get("Branch Panel", "").upper().strip()
    if not ubc_val and branch_panel:
        ubc_val = branch_panel

    data["UBC Asset Tag"] = ubc_val

    # Ensure Description = "Panel - <UBC Asset Tag>" (even if empty -> "Panel - ")
    data["Description"] = f"Panel - {ubc_val}".strip()

    # Normalize trivial values
    for key in ["Ampere", "Supply From", "Volts", "Location", "Branch Panel"]:
        val = str(data.get(key, "")).strip()
        # Clean weird artifacts like single dots
        if val == ".":
            val = ""
        data[key] = val

    # Build final JSON
    result = {
        "qr_code": qr,                         # keep leading zeros from filename
        "building_number": building,
        "asset_type": "- EL",
        "structured_data": data,
    }

    out_path = os.path.join(OUTPUT_DIR, f"{qr}.json")
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {out_path}")

