import fitz  # PyMuPDF
import json
import base64
import pytesseract
import cv2
import numpy as np
import os
from PIL import Image
from dotenv import load_dotenv
import io
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
print("Checking environment variables...")
print(f"API Key exists: {bool(os.getenv('OPENAI_API_KEY'))}")
if os.getenv('OPENAI_API_KEY'):
    print(f"API Key starts with: {os.getenv('OPENAI_API_KEY')[:10]}...")
print()

# Initialize Azure client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def process_cover_sheet_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # --- 1. RENDER PAGE TO IMAGE ---
    # Zoom=4 for higher resolution (improves OCR accuracy)
    mat = fitz.Matrix(4, 4) 
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Convert to PIL Image for Tesseract
    pil_image = Image.open(io.BytesIO(img_data))

    # --- PREPARE IMAGE FOR LLM (Vision) ---
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # --- AUTO-DETECT AND ROTATE ---
    try:
        # Get OSD data as dict
        osd_data = pytesseract.image_to_osd(pil_image, output_type=pytesseract.Output.DICT)
        rotation = osd_data['rotate']
        
        print(f"Detected rotation: {rotation} degrees")
        print(f"Confidence: {osd_data.get('orientation_conf', 'N/A')}")
        
        # Rotate image if needed
        if rotation != 0:
            pil_image = pil_image.rotate(-rotation, expand=True)
            
    except Exception as e:
        print(f"OSD failed: {e}. Proceeding without rotation correction.")
    
    # Get dimensions of the *image* (not the pdf point size)
    # Because OCR returns coords relative to this image
    img_width, img_height = pil_image.size

    # --- 2. RUN OCR --- TODO: Later replace with API calls for higher accuracy 
    # This returns a dict with: 'text', 'left', 'top', 'width', 'height', 'conf'
    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

    document_layer = {
        "page": 1,
        "width": img_width,
        "height": img_height,
        "blocks": []
    }
    
    indexed_text = ""
    block_counter = 0

    # Iterate through OCR results
    n_boxes = len(ocr_data['text'])
    for i in range(n_boxes):
        # Filter out empty text (OCR often returns empty blocks for layout)
        text = ocr_data['text'][i].strip()
        if not text:
            continue
            
        # Capture specific bbox
        x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                      ocr_data['width'][i], ocr_data['height'][i])
        
        # Normalize bbox [0..1]
        norm_bbox = [
            x / img_width,
            y / img_height,
            (x + w) / img_width,
            (y + h) / img_height
        ]
        
        block_id = f"b{block_counter}"
        
        # Add to Document Layer
        document_layer["blocks"].append({
            "id": block_id,
            "text_raw": text,
            "bbox": norm_bbox,
            "confidence": float(ocr_data['conf'][i]) / 100.0, # Tesseract is 0-100
            "source": "ocr"
        })
        
        # Add to Context for LLM
        indexed_text += f"ID: {block_id} | Text: {text}\n"
        block_counter += 1

    # --- 3. CALL LLM (Semantic Layer) ---
    # We ask the LLM to find values and cite the IDs
    system_prompt = """Extract solar permit cover sheet data into 6 required categories.

    CATEGORIES & FIELDS:
    1. project_location: street_address, apn
    2. scope_of_work: work_description, system_type
    3. governing_codes: codes_list
    4. owner_information: name, address, phone, email
    5. designer_contractor_info: company_name, address, phone, email, license_number
    6. sheet_index: sheets_listed, dated_and_signed

    RULES:
    - Match semantically (e.g., "Applicant"→owner, "Codes Used"→governing_codes)
    - Format: {value, confidence, evidence_ids, source_heading}
    - If missing: {value: null, confidence: null, evidence_ids: [], source_heading: null}
    - Include ALL 6 categories even if empty
    - Extra data → "other_information"

    JSON structure:
    {
    "requirements": {
        "project_location": {"extracted_data": {"street_address": {...}, "apn": {...}}},
        "scope_of_work": {"extracted_data": {"work_description": {...}, "system_type": {...}}},
        "governing_codes": {"extracted_data": {"codes_list": {...}}},
        "owner_information": {"extracted_data": {"name": {...}, "address": {...}, "phone": {...}, "email": {...}}},
        "designer_contractor_info": {"extracted_data": {"company_name": {...}, "address": {...}, "phone": {...}, "email": {...}, "license_number": {...}}},
        "sheet_index": {"extracted_data": {"sheets_listed": {...}, "dated_and_signed": {...}}}
    },
    "other_information": {}
    }
    """

    # Check image size before sending
    base64_size_mb = len(base64_image) / (1024 * 1024)
    print(f"Base64 image size: {base64_size_mb:.2f} MB")
    print(f"OCR blocks extracted: {block_counter}")
    print(f"Calling OpenAI API...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Simple! No deployment names needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the text map:\n" + indexed_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        
        print("✓ API call successful!")
        
    except Exception as e:
        print(f"✗ API call failed: {e}")
        raise
    
    llm_output = json.loads(response.choices[0].message.content)

    # --- 4. SYNTHESIS (Merge LLM choice with Exact BBox) ---
    semantic_layer = {}
    
    # Navigate through the nested structure from LLM
    if "requirements" in llm_output:
        for category, category_data in llm_output["requirements"].items():
            if "extracted_data" in category_data:
                for field, field_data in category_data["extracted_data"].items():
                    evidence_list = []
                    for eid in field_data.get("evidence_ids", []):
                        # Look up the exact bbox from our Document Layer
                        original_block = next((b for b in document_layer["blocks"] if b["id"] == eid), None)
                        if original_block:
                            evidence_list.append({
                                "page": 1,
                                "bbox": original_block["bbox"],
                                "text": original_block["text_raw"]
                            })
                    
                    field_key = f"{category}.{field}"
                    semantic_layer[field_key] = {
                        "value": field_data.get("value"),
                        "confidence": field_data.get("confidence"),
                        "evidence": evidence_list
                    }

    # Final Output Construction
    final_json = {
        "doc_id": "cover_sheet_001",
        "document_layer": document_layer,
        "semantic_layer": semantic_layer,
        "llm_extraction": llm_output
    }
    
    # Save to file
    output_path = "extraction_output.json"
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=2)
    
    print(f"✓ Output saved to {output_path}")
    
    return json.dumps(final_json, indent=2)


def test_openai_connection():
    """Test OpenAI API connection before processing"""
    try:
        print("Testing OpenAI API connection...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✓ Connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


def main():
    result = process_cover_sheet_ocr(r"S:\Work\Permi\Permi\data\Cover Sheet.pdf")
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    if test_openai_connection():
        main()
    else:
        print("Fix connection issues before proceeding")