import pytesseract
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
import pdfplumber
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please add it in your .env file.")
genai.configure(api_key=gemini_api_key)


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication



# Configure path to Tesseract (adjust path as needed)
# For Windows - point to tesseract.exe, not the installer
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def summarize_with_gemini(text: str) -> dict:
    """
    Summarize document text using Gemini.
    Returns a dict: {
      "human_report": str,   # full human-readable report (both languages when Malayalam present)
      "json": dict or None,  # parsed machine-readable JSON (if model produced valid JSON)
      "raw": str             # raw LLM output
    }
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are a *multilingual expert document analyst* (fluent in English and Malayalam).\n\n"
            "Task: Analyze the full document text provided and produce TWO things, in this order:\n"
            "  1) A human-readable report (headings + bullets). If the document contains Malayalam, produce BOTH a Malayalam summary and an English summary. If the document is only English, produce only the English summary and set the Malayalam summary area to null or a short note.\n"
            "  2) A single machine-readable JSON object (valid JSON) inside a fenced block labeled ```json ... ``` that contains extracted fields (language, title, summaries.en, summaries.ml, metadata, dates, amounts, action_items, tables, uncertainties, confidence_overall).\n\n"
            "Important rules for the human report:\n"
            "- Detect primary language(s) and add a confidence score for each (0.0–1.0).\n"
            "- If Malayalam is present, provide a short Malayalam summary (3–6 bullets) AND a short English summary (3–6 bullets). Preserve original Malayalam text where quoting.\n"
            "- Provide a Detailed Analysis with labeled subsections: Key dates (ISO YYYY-MM-DD where possible), Numeric/funding details (normalize numbers), Actionable items (action, owner, due date, source snippet), Tables & figures (brief CSV or JSON), Uncertainties (OCR noise etc.), and Recommended next steps (2–5 items).\n"
            "- Quote up to 25 words from the original for any critical extracted item and preserve original characters (Malayalam included). Mark OCR/uncertain snippets with \"uncertain\".\n\n"
            "JSON rules:\n"
            "- In addition to existing fields, also add a key \"department\" with the most likely department this document belongs to (e.g., Finance, HR, Legal, Operations, Engineering, Procurement, Transport, Safety, etc.).\n"
            "- If uncertain, set department to \"Unknown\".\n"
            "- Output valid JSON only (no surrounding commentary inside the JSON block). Use keys exactly as below example and fill nulls where missing:\n"
            "{\n"
            '  "language": {"detected": ["en","ml"], "confidence": {"en":0.0,"ml":0.0}},\n'
            '  "title": null,\n'
            '  "summaries": {"en": "short english summary", "ml": "short malayalam summary or null"},\n'
            '  "metadata": {"authors": [], "references": [], "document_date": null},\n'
            '  "dates": [], "amounts": [], "action_items": [], "tables": [], "uncertainties": [], "confidence_overall": 0.0,\n'
            '  "department": "Unknown"\n'
            "}\n\n"
            "Presentation: First the human-readable report (headings + bullets). Then, on a new line, place the valid JSON inside a fenced block labeled ```json ... ```.\n\n"
            "If the document contains Malayalam text, ensure the Malayalam summary is natural Malayalam (not transliteration) and the English summary accurately reflects the Malayalam content. If any translation is low-confidence, mark the translated passage with confidence < 0.7 and include the original Malayalam snippet.\n\n"
            "Document Content:\n"
            f"{text}\n"
        )

        # ask model to generate (adjust tokens if needed)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2500  # Corrected parameter name
            )
        )
        raw = response.text

        # try to extract the JSON block (prefer ```json ... ``` then any fenced block)
        json_text = None
        m = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if not m:
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if m:
            json_text = m.group(1).strip()
        else:
            # fallback: try to find the last { ... } in the output
            brace_match = re.search(r'(\{[\s\S]*\})\s*$', raw)
            if brace_match:
                json_text = brace_match.group(1)

        parsed = None
        if json_text:
            try:
                parsed = json.loads(json_text)
            except Exception:
                # best-effort cleanup: remove trailing commas, fix common issues, then try again
                cleaned = re.sub(r',\s*([}\]])', r'\1', json_text)  # remove trailing commas
                try:
                    parsed = json.loads(cleaned)
                except Exception:
                    parsed = None

        # human_report: everything before the JSON fenced block if present, else full raw
        human_report = raw
        fenced_index = raw.find('```json')
        if fenced_index != -1:
            human_report = raw[:fenced_index].strip()
        else:
            # if any fenced block exists, strip it out for the human report
            fenced_generic = re.search(r'```(?:json)?', raw)
            if fenced_generic:
                human_report = raw[:fenced_generic.start()].strip()

        return {"human_report": human_report, "json": parsed, "raw": raw}

    except Exception as e:
        return {"error": str(e)}

@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_ext = file.filename.rsplit('.', 1)[1].lower()
        extracted_text = ""

        if file_ext == 'pdf':
            # Use pdfplumber to extract text and tables
            with pdfplumber.open(file) as pdf:
                all_text = []
                all_tables = []

                for page in pdf.pages:
                    # Extract page text
                    text = page.extract_text()
                    if text:
                        all_text.append(text)

                    # Extract tables from page
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table)
                        table_text = df.to_markdown(index=False)
                        all_tables.append(table_text)

            extracted_text = "\n\n".join(all_text)
            extracted_tables = "\n\n".join(all_tables)

            combined_result = f"{extracted_text}\n\nExtracted Tables:\n{extracted_tables}"

            if not combined_result.strip():
                return jsonify({"text": "No text or tables could be extracted from the document.", "success": True})

            # Summarize with Gemini
            summary = summarize_with_gemini(combined_result)

            return jsonify({
                "text": combined_result,
                "summary": summary,
                "success": True,
                "message": "Text and tables extracted + summarized successfully"
            })

        else:
            # For image files, use OCR
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Invalid image file"}), 400

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            extracted_text = pytesseract.image_to_string(gray, lang='mal+eng').strip()

            if not extracted_text:
                return jsonify({"text": "No text could be extracted from the image.", "success": True})

            # Summarize with Gemini
            summary = summarize_with_gemini(extracted_text)

            return jsonify({
                "text": extracted_text,
                "summary": summary,
                "success": True,
                "message": "Image text extracted + summarized successfully"
            })

    except Exception as e:
        return jsonify({"error": f"OCR processing failed: {str(e)}", "success": False}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OCR + Gemini summarizer is running"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)







    # @app.route('/extract-text', methods=['POST'])
# def extract_text():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No file selected"}), 400
        
#         file_ext = file.filename.rsplit('.', 1)[1].lower()
#         extracted_text = ""

#         if file_ext == 'pdf':
#             # Use pdfplumber to extract text and tables
#             with pdfplumber.open(file) as pdf:
#                 all_text = []
#                 all_tables = []
                
#                 for page in pdf.pages:
#                     # Extract page text
#                     text = page.extract_text()
#                     if text:
#                         all_text.append(text)
                    
#                     # Extract tables from page
#                     tables = page.extract_tables()  # Returns list of tables (list of lists of cells)
#                     for table in tables:
#                         df = pd.DataFrame(table)
#                         # Convert table to markdown or text format, you can customize this
#                         table_text = df.to_markdown(index=False)
#                         all_tables.append(table_text)

#             extracted_text = "\n\n".join(all_text)
#             extracted_tables = "\n\n".join(all_tables)

#             combined_result = f"{extracted_text}\n\nExtracted Tables:\n{extracted_tables}"

#             if not combined_result.strip():
#                 return jsonify({"text": "No text or tables could be extracted from the document.", "success": True})

#             return jsonify({"text": combined_result, "success": True, "message": "Text and tables extracted successfully"})

#         else:
#             # For image files, use your existing OCR logic
#             npimg = np.frombuffer(file.read(), np.uint8)
#             img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#             if img is None:
#                 return jsonify({"error": "Invalid image file"}), 400
            
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             gray = cv2.GaussianBlur(gray, (5, 5), 0)
#             gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
#             extracted_text = pytesseract.image_to_string(gray, lang='mal+eng')
#             extracted_text = extracted_text.strip()
            
#             if not extracted_text:
#                 return jsonify({"text": "No text could be extracted from the image.", "success": True})

#             return jsonify({"text": extracted_text, "success": True, "message": "Text extracted successfully"})
    
#     except Exception as e:
#         return jsonify({"error": f"OCR processing failed: {str(e)}", "success": False}), 500



# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "OCR service is running"})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
