import fitz  # PyMuPDF
import json
import base64
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF while preserving structure"""
    doc = fitz.open(pdf_path)
    
    full_text = ""
    page_texts = []
    
    for page_num, page in enumerate(doc, start=1):
        # Extract text with layout preservation
        text = page.get_text("text")
        page_texts.append({
            "page_number": page_num,
            "text": text
        })
        full_text += f"\n\n--- PAGE {page_num} ---\n\n{text}"
    
    doc.close()
    
    return full_text, page_texts


def extract_rules_with_llm(pdf_text, pdf_filename):
    """Use LLM to extract structured requirements from permit document"""
    
    system_prompt = """You are a permit requirements extraction agent. Extract structured requirements from permit guideline documents.

                       YOUR TASK:
                        Given a permit requirements document, extract all requirements into a structured JSON format.

                       EXTRACTION RULES:

                        1. IDENTIFY SECTIONS: Find major sections (e.g., "COVER SHEET", "SITE PLAN", "ROOF PLAN")

                        2. EXTRACT REQUIREMENTS: For each section, identify individual requirements

                        3. CLASSIFY VALIDATION TYPE:
                            - field_presence: Must include specific information (e.g., "provide owner name and address")
                            - enum_value: Must be one of specific values (e.g., "system type: grid-tied, off-grid, or hybrid")
                            - content_match: Must contain specific keywords/phrases (e.g., "include County ordinances Title 19, 22, 23")
                            - format_spec: Must follow specific format (e.g., "minimum scale 1/8" = 1'-0"")
                            - conditional: If X then Y requirements (e.g., "if weight > 3 lbs/sqft then require engineering")
                            - content_quality: Description must be adequate/clear (requires judgment)

                        4. MAP TO STANDARDIZED FIELDS:
                            Common mappings:
                                - "Owner" / "Applicant" / "Property Owner" → owner_information
                                - "Designer" / "Contractor" / "Engineer" → designer_contractor_info  
                                - "System Description" / "Scope of Work" / "Project Description" → scope_of_work
                                - "Codes" / "Standards" / "Regulations" → governing_codes
                                - "Contact info" typically means: name, address, phone, email
                                - "Site Plan" → site_plan
                                - "Roof Plan" → roof_plan

                        5. DETERMINE SEVERITY:
                            - critical: Must have for permit approval (explicit requirements)
                            - warning: Should have, but might be conditional
                            - info: Nice to have, or informational

                        6. EXTRACT METADATA:
                            - requirement_text: Exact quote from document
                            - required_fields: Array of standardized field paths (e.g., ["owner_information.name", "owner_information.address"])
                            - validation_type: One of the types above
                            - validation_criteria: Additional details (e.g., allowed_values, must_contain keywords, min_length)
                            - severity: critical/warning/info
                            - source_section: Which section this came from
                            - source_page: Page number where found

                        OUTPUT SCHEMA:
                        {
                        "jurisdiction": "string - extracted from document header/title",
                        "document_type": "string - infer type (e.g., solar_permit_requirements, building_permit_requirements)",
                        "document_id": "string - document number if present",
                        "version": "string - version/date if present",
                        "extraction_metadata": {
                            "extraction_date": "ISO timestamp",
                            "source_filename": "string"
                        },
                        "sections": [
                            {
                            "section_id": "string - lowercase_underscore (e.g., cover_sheet)",
                            "section_name": "string - as appears in doc (e.g., COVER SHEET)",
                            "page_number": number,
                            "requirements": [
                                {
                                "requirement_id": "string - auto-generate like SECTION_001",
                                "requirement_text": "string - exact quote from document",
                                "validation_type": "string - one of: field_presence, enum_value, content_match, format_spec, conditional, content_quality",
                                "required_fields": ["array of field paths"],
                                "validation_criteria": {
                                    // Optional, depends on validation_type:
                                    "allowed_values": ["for enum_value"],
                                    "must_contain": ["for content_match"],
                                    "min_length": number,
                                    "format": "for format_spec"
                                },
                                "severity": "critical/warning/info",
                                "source_section": "string - section name",
                                "source_page": number
                                }
                            ]
                            }
                        ]
                        }

                        IMPORTANT:
                            - Extract ALL requirements, even if some seem minor
                            - Use exact quotes for requirement_text
                            - Be generous with field mapping (if unsure, include multiple possible fields)
                            - Generate sequential requirement_ids within each section (e.g., COVER_001, COVER_002)
                            - If requirement lists multiple items (e.g., "name, address, phone, email"), create separate required_fields for each
                    """

    user_prompt = f"""Extract all requirements from this permit guideline document.

                      Document text:
                      {pdf_text}

                      Return ONLY valid JSON following the schema provided in the system prompt."""

    print("Calling OpenAI API to extract requirements...")
    print(f"Text length: {len(pdf_text)} characters")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        print("✓ API call successful!")
        
        rules_json = json.loads(response.choices[0].message.content)
        
        # Add extraction metadata
        if "extraction_metadata" not in rules_json:
            rules_json["extraction_metadata"] = {}
        
        rules_json["extraction_metadata"]["extraction_date"] = datetime.now().isoformat()
        rules_json["extraction_metadata"]["source_filename"] = pdf_filename
        
        return rules_json
        
    except Exception as e:
        print(f"✗ API call failed: {e}")
        raise


def process_requirements_pdf(pdf_path):
    """Main function to extract rules from a requirements PDF"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {pdf_path}")
    print(f"{'='*60}\n")
    
    # Extract text from PDF
    print("Step 1: Extracting text from PDF...")
    full_text, page_texts = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(page_texts)} pages")
    print(f"✓ Total characters: {len(full_text)}")
    
    # Extract rules using LLM
    print("\nStep 2: Extracting requirements with LLM...")
    pdf_filename = Path(pdf_path).name
    rules_json = extract_rules_with_llm(full_text, pdf_filename)
    
    # Count requirements
    total_requirements = sum(
        len(section.get("requirements", [])) 
        for section in rules_json.get("sections", [])
    )
    print(f"✓ Extracted {len(rules_json.get('sections', []))} sections")
    print(f"✓ Extracted {total_requirements} requirements")
    
    # Save output
    print("\nStep 3: Saving output...")
    
    # Setup output directory
    parent_dir = Path(__file__).parent.parent
    outputs_dir = parent_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"rules_{Path(pdf_path).stem}_{timestamp}.json"
    output_path = outputs_dir / output_filename
    
    # Save JSON
    with open(output_path, "w") as f:
        json.dump(rules_json, f, indent=2)
    
    print(f"✓ Rules saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Jurisdiction: {rules_json.get('jurisdiction', 'N/A')}")
    print(f"Document Type: {rules_json.get('document_type', 'N/A')}")
    print(f"Sections: {len(rules_json.get('sections', []))}")
    print(f"Total Requirements: {total_requirements}")
    print(f"\nSections extracted:")
    for section in rules_json.get("sections", []):
        req_count = len(section.get("requirements", []))
        print(f"  - {section['section_name']}: {req_count} requirements")
    print(f"{'='*60}\n")
    
    return rules_json, output_path


def main():
    """Main entry point"""
    
    # Check if API key exists
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
        return
    
    # Example usage - update this path to your requirements PDF
    pdf_path = r"S:\Work\Permi\Permi\data\County of SLO Solar Requirements.pdf"
    
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        print("Please update the pdf_path in main() to point to your requirements document")
        return
    
    # Process the PDF
    rules_json, output_path = process_requirements_pdf(pdf_path)
    
    print(f"✓ Complete! Rules extracted and saved.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()