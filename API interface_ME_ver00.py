import os
import json
import base64
import re
from dotenv import load_dotenv
from collections import defaultdict
from openai import OpenAI

# --- [1] Load API key ---
load_dotenv(dotenv_path=r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\OpenAI_key_bryan.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- [2] Define folders & valid suffixes ---
image_folder  = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\Capture_photos_upload"
output_folder = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\Output_jason_api"
os.makedirs(output_folder, exist_ok=True)

# Only these suffixes correspond to the pictures we want (sequence identifiers)
VALID_SUFFIXES = {"0", "1", "3"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Prepare grouping structure:
grouped = defaultdict(lambda: {"images": {}})

# --- [3] Scan & parse filenames ---
# Filename pattern: "<QR> <Building> ME - <Sequence>"
pattern = re.compile(
    r"^(\d+)\s+"            # 1) QR code
    r"(\d+(?:-\d+)?)\s+"    # 2) Building number (digits, optional '-digits')
    r"(ME)\s*-\s*([013])$", # 3) Asset type 'ME' + dash + sequence (0,1,3)
    re.IGNORECASE
)

for fn in os.listdir(image_folder):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in VALID_EXTS:
        continue

    m = pattern.match(base)
    if not m:
        print(f"⚠ Skipping unrecognized filename: {fn}")
        continue

    qr, building, asset_type, seq = m.groups()
    if asset_type.upper() != "ME":
        continue
    if seq not in VALID_SUFFIXES:
        continue

    grouped[qr]["building"]    = building
    grouped[qr]["asset_type"]  = "ME"
    grouped[qr]["images"][seq] = os.path.join(image_folder, fn)
    print(f"→ QR={qr}, Building={building}, Type=ME, Seq={seq}")

print(f"\nTotal assets found: {len(grouped)}")

# --- [4] Helper to base64-encode with correct MIME type ---
def encode_image(path):
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

# --- [5] Loop over each asset group and call GPT-4o ---
for qr, info in grouped.items():
    print(f"\nProcessing QR {qr} …")
    image_messages = []
    for seq in ("0", "1", "3"):
        if seq in info["images"]:
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": encode_image(info["images"][seq])}
            })

    if not image_messages:
        print("  No images for this QR; skipping.")
        continue

    prompt = (
        "You will receive up to three images of an asset plate and labels. "
        "Extract ONLY the following fields into a strict JSON object (use empty strings if missing):\n"
        "- Manufacturer\n"
        "- Model\n"
        "- Serial Number\n"
        "- Year (4-digit)\n"
        "- UBC Tag\n"
        "- Technical Safety BC\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + image_messages}],
        temperature=0.0,
        max_tokens=1000
    )

    reply = response.choices[0].message.content or ""
    reply = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply, flags=re.IGNORECASE)

    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        print("  ❌ JSON parse error; saving raw reply for inspection.")
        parsed = {}

    # Build final payload
    output_data = {
        "qr_code":         qr,
        "building_number": info.get("building", ""),
        "asset_type":      f"- {info.get('asset_type', '').upper()}",
        "structured_data": {
            "Manufacturer":         parsed.get("Manufacturer", ""),
            "Model":                parsed.get("Model", ""),
            "Serial Number":        parsed.get("Serial Number", ""),
            "Year":                 parsed.get("Year", ""),
            "UBC Tag":              parsed.get("UBC Tag", ""),
            "Technical Safety BC":  parsed.get("Technical Safety BC", "")
        }
    }

    # Build JSON filename as: QR_ME_Building.json
    json_filename = f"{qr}_{info.get('asset_type', '').upper()}_{info.get('building', '')}.json"
    out_path = os.path.join(output_folder, json_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved {out_path}")
