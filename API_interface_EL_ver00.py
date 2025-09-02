import os
import json
import base64
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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

# -----------------------------
# CONFIG
# -----------------------------
import platform, shutil, pytesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"

# --- [1] Load API key ---
load_dotenv(dotenv_path=r"/home/developer/API/OpenAI_key_bryan.env") # adjust as needed
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- [2] Paths & constants ---
image_folder  = r"/home/developer/Capture_photos_upload"
output_folder = r"/home/developer/Output_jason_api"
debug_folder  = os.path.join(output_folder, "debug_el")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# NEW: DB path (SQLite)
DB_PATH = r"/home/developer/asset_capture_app_dev/data/QR_codes.db"
DB_TABLE = "QR_codes"

# Accept EL - 0, EL - 1, EL - 2
VALID_SUFFIXES = {"0", "1", "2"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Header crop fraction for EL-2 (top 25%)
HEADER_FRACTION = 0.25

# Candidate regex for EL-1 tag — require at least one digit to avoid "LO OY"
UBC_TAG_CANDIDATE = re.compile(r"\b([A-Z]{2,5})[ -]?(?=[A-Z0-9]*\d)[A-Z0-9]{2,8}\b")

# -----------------------------
# DB helpers
# -----------------------------
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

# -----------------------------
# Group files by QR
# Filename pattern: "<QR> <Building> EL - <Sequence>"
# -----------------------------
pattern = re.compile(
    r"^(\d+)\s+"
    r"(\d+(?:-\d+)?)\s+"
    r"(EL)\s*-\s*([012])$",
    re.IGNORECASE
)

grouped = defaultdict(lambda: {"images": {}, "building": "", "asset_type": "EL"})

for fn in os.listdir(image_folder):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in VALID_EXTS:
        continue
    m = pattern.match(base)
    if not m:
        continue
    qr, building, asset_type, seq = m.groups()
    if seq not in VALID_SUFFIXES or asset_type.upper() != "EL":
        continue

    # Skip if Approved=1
    if qr in APPROVED_QRS:
        print(f"⏭️  Skipping QR {qr} (Approved=1 in DB)")
        continue

    grouped[qr]["building"] = building
    grouped[qr]["images"][seq] = os.path.join(image_folder, fn)

print(f"\nTotal assets found (after approval filter): {len(grouped)}")

# -----------------------------
# General helpers
# -----------------------------
def normalize_code(s: str) -> str:
    """Uppercase, collapse spaces, normalize dashes."""
    if not s:
        return ""
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s

def is_reasonable_tag(s: str) -> bool:
    """
    Heuristics for a valid UBC Asset Tag:
    - Has at least one digit
    - Only A-Z, 0-9, spaces, or single dashes
    - Length not too short/long
    - Not just two short words with no digits (e.g., 'LO OY')
    """
    if not s:
        return False
    if len(s) < 3 or len(s) > 16:
        return False
    if re.search(r"[^A-Z0-9 \-]", s):
        return False
    if not re.search(r"\d", s):
        return False
    if re.fullmatch(r"[A-Z]{1,3}\s+[A-Z]{1,3}", s):
        return False
    return True

# -----------------------------
# Image / OCR helpers
# -----------------------------
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

def ocr_basic(image_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    txt = pytesseract.image_to_string(thr)
    return txt

def crop_header(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    h_head = max(40, int(h * HEADER_FRACTION))
    return image_bgr[0:h_head, 0:w].copy()

def draw_header_overlay(image_bgr: np.ndarray, header_bgr: np.ndarray, header_text: str) -> np.ndarray:
    out = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    h_head = header_bgr.shape[0]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h_head), (0, 255, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
    snippet = (header_text or "").strip().replace("\n", " ")
    snippet = snippet[:120] + ("..." if len(snippet) > 120 else "")
    cv2.putText(out, snippet, (10, min(h_head-10, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 120, 30), 2, cv2.LINE_AA)
    return out

def find_text_boxes(image_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 12:
            boxes.append((x, y, w, h))
    return boxes

def add_boxes_overlay(image_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = image_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return out

# -----------------------------
# OpenAI helpers (Vision)
# -----------------------------
def call_openai_structured_header(image_path: str) -> Dict[str, str]:
    """
    Ask the model to read ONLY the header from an EL schedule (EL - 2).
    Returns: Branch Panel, Location, Supply From, Volts, Ampere
    """
    prompt = (
        "You are reading the HEADER ONLY of an electrical panel schedule image. "
        "Ignore the table rows, breakers, and all details below the header line. "
        "Extract these fields from the header:\n"
        "- Branch Panel (e.g., 2NRM1, CDP 2N0D1)\n"
        "- Location (e.g., Mechanical 7000)\n"
        "- Supply From (e.g., CAD2)\n"
        "- Volts (e.g., 120/208, 347/600)\n"
        "- Ampere (e.g., 100 A, 225A)\n\n"
        "Rules:\n"
        "1) Return ONLY a compact JSON object with keys exactly: "
        "{\"Branch Panel\", \"Location\", \"Supply From\", \"Volts\", \"Ampere\"}\n"
        "2) Use strings for all values. If unknown, use empty string.\n"
        "3) Do not add extra keys or explanations.\n"
        "4) If the Branch Panel shows a prefix like 'Panel', exclude the word 'Panel'. "
        "Return just the code (e.g., 'CDP 2N0D1').\n"
    )

    image_b64 = encode_image(image_path)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured fields from images precisely and return strict JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=200,
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return {
                "Branch Panel": parsed.get("Branch Panel", "").strip(),
                "Location": parsed.get("Location", "").strip(),
                "Supply From": parsed.get("Supply From", "").strip(),
                "Volts": parsed.get("Volts", "").strip(),
                "Ampere": parsed.get("Ampere", "").strip(),
            }
    except Exception as e:
        print(f"⚠ OpenAI header extraction error: {e}")
    return {"Branch Panel":"", "Location":"", "Supply From":"", "Volts":"", "Ampere":""}

def call_openai_tag(image_path: str) -> str:
    """
    Read UBC Asset Tag on EL - 1 (e.g., 'CDP 2N0D1'). Return ONLY the tag text.
    """
    prompt = (
        "Read the asset tag shown in this image. It usually looks like a code such as 'CDP 2N0D1'. "
        "Return ONLY the tag text as a single line. Do not include the word 'Panel'. "
        "If uncertain, return an empty string."
    )
    image_b64 = encode_image(image_path)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You read single short codes from images and return exactly the code."},
                {
                    "role": "user",
                    "content": [
                        {"type":"text", "text": prompt},
                        {"type":"image_url", "image_url": {"url": image_b64}},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=30,
        )
        txt = (resp.choices[0].message.content or "").strip()
        txt = re.sub(r"[\n\r]", " ", txt)
        txt = re.sub(r"(?i)^panel\s*[-–—]\s*", "", txt).strip()
        return txt
    except Exception as e:
        print(f"⚠ OpenAI tag error: {e}")
        return ""

# -----------------------------
# OCR fallback parsing
# -----------------------------
def parse_header_text_freeform(text: str) -> Dict[str, str]:
    """
    Very lenient free-text parser for header fields (OCR fallback).
    """
    flat = " ".join(text.split())
    out = {"Branch Panel":"", "Location":"", "Supply From":"", "Volts":"", "Ampere":""}

    # Branch Panel: like 'CDP 2N0D1' or '2NRM1'
    m_bp = re.search(r"\b([A-Z]{2,4}\s*[A-Z0-9]{2,6}|[A-Z0-9]{4,6})\b", flat)
    if m_bp:
        out["Branch Panel"] = m_bp.group(0).strip()

    # Volts: like 120/208
    m_volt = re.search(r"\b(\d{3}\/\d{3}|\d{3}\/\d{2,3})\b", flat)
    if m_volt:
        out["Volts"] = m_volt.group(0)

    # Ampere: like 100 A, 225A
    m_amp = re.search(r"\b(\d{2,4})\s*A\b", flat, re.IGNORECASE) or re.search(r"\b(\d{2,4})A\b", flat, re.IGNORECASE)
    if m_amp:
        out["Ampere"] = f"{m_amp.group(1)} A"

    # Supply From: short code like CAD2
    m_sup = re.search(r"\b([A-Z]{2,4}\s?\d{0,2}[A-Z]?)\b", flat)
    if m_sup:
        out["Supply From"] = m_sup.group(0).strip()

    # Location: phrase after 'Location'
    m_loc = re.search(r"(?i)location[:\-\s]+([A-Za-z0-9 \-_/]+)", flat)
    if m_loc:
        out["Location"] = m_loc.group(1).strip()

    return out

def tesseract_read_text(image_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    txt = pytesseract.image_to_string(thr)
    return txt

def tesseract_read_tag(image_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    txt = pytesseract.image_to_string(thr)
    m = UBC_TAG_CANDIDATE.search(" ".join(txt.split()))
    return m.group(0) if m else ""

# -----------------------------
# Main per-asset processing
# -----------------------------
for qr, info in grouped.items():
    if qr in APPROVED_QRS:
        print(f"⏭️  Skipping QR {qr} (Approved=1 in DB)")
        continue

    print(f"\nProcessing QR {qr} …")

    # Final EL structure
    result = {
        "Description": "",       # will derive after we know the tag
        "UBC Asset Tag": "",     # from EL - 1 (fallback to Branch Panel)
        "Branch Panel": "",
        "Ampere": "",
        "Supply From": "",
        "Volts": "",
        "Location": "",
    }

    # --- 1) Extract header fields from EL - 2 (to have Branch Panel for fallback) ---
    if "2" in info["images"]:
        path_el2 = info["images"]["2"]
        img2 = cv2.imread(path_el2)
        if img2 is None:
            print(f"⚠ Could not read EL-2 image for QR {qr}")
        else:
            header = crop_header(img2)

            header_fields = call_openai_structured_header(path_el2)

            # OCR fallback for any missing key
            if any(not header_fields.get(k, "") for k in ["Branch Panel","Location","Supply From","Volts","Ampere"]):
                ocr_txt = ocr_basic(header)
                parsed = parse_header_text_freeform(ocr_txt)
                for k in header_fields:
                    if not header_fields.get(k):
                        header_fields[k] = parsed.get(k, "")

            # Assign to result
            result["Branch Panel"] = header_fields.get("Branch Panel", "")
            result["Location"]     = header_fields.get("Location", "")
            result["Supply From"]  = header_fields.get("Supply From", "")
            result["Volts"]        = header_fields.get("Volts", "")
            result["Ampere"]       = header_fields.get("Ampere", "")

            # Visual debug overlays
            header_txt = ocr_basic(header)
            overlay = draw_header_overlay(img2, header, header_txt)
            boxes2 = find_text_boxes(header)
            overlay2 = add_boxes_overlay(overlay, boxes2)
            cv2.imwrite(os.path.join(debug_folder, f"{qr}_EL2_header_overlay.jpg"), overlay2)
            cv2.imwrite(os.path.join(debug_folder, f"{qr}_EL2_header_crop.jpg"), header)

            print(f"  → Header fields: {header_fields}")

    # --- 2) Extract UBC Asset Tag from EL - 1 (preferred) ---
    ubc_tag_raw = ""
    if "1" in info["images"]:
        path_tag = info["images"]["1"]
        img_tag = cv2.imread(path_tag)

        # Try OCR first
        tag_ocr = tesseract_read_tag(img_tag)
        # If OCR weak/empty, ask LLM
        tag_model = ""
        if not tag_ocr:
            tag_model = call_openai_tag(path_tag)

        ubc_tag_raw = (tag_ocr or tag_model).strip()
        ubc_tag_raw = re.sub(r"(?i)^panel\s*[-–—]\s*", "", ubc_tag_raw).strip()

        # Debug overlays for EL-1
        boxes = find_text_boxes(img_tag)
        tag_overlay = add_boxes_overlay(img_tag, boxes)
        cv2.imwrite(os.path.join(debug_folder, f"{qr}_EL1_tag_overlay.jpg"), tag_overlay)
        cv2.imwrite(os.path.join(debug_folder, f"{qr}_EL1_tag_raw.jpg"), img_tag)

        print(f"  → EL-1 tag candidates: OCR='{tag_ocr}' | LLM='{tag_model}' | chosen(raw)='{ubc_tag_raw}'")

    # --- 3) Finalize UBC Asset Tag with validation & fallback to Branch Panel ---
    bp = normalize_code(result.get("Branch Panel", ""))
    tag_candidate = normalize_code(ubc_tag_raw)

    if is_reasonable_tag(tag_candidate):
        chosen_tag = tag_candidate
    else:
        chosen_tag = bp  # EL-1 was junk → fallback to Branch Panel

    result["UBC Asset Tag"] = chosen_tag  # no "Panel - " prefix here

    # --- 4) Derive Description = "Panel - <UBC Asset Tag>" ---
    result["Description"] = f"Panel - {chosen_tag}" if chosen_tag else "Panel"

    # Save JSON output
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
