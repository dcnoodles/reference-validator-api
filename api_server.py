"""
api_server.py – Flask REST API wrapper for reference_validator-GUIDE-1.py
─────────────────────────────────────────────────────────────────────────────
Setup:
  pip install flask flask-cors

Run (from the same folder as your validator script):
  python api_server.py

The server starts at http://localhost:5000
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import importlib.util
import re
from pathlib import Path

# ── Load the validator module (handles the hyphenated filename) ──────────────
_VALIDATOR_FILE = Path(__file__).parent / "reference_validator-GUIDE-1.py"

if not _VALIDATOR_FILE.exists():
    print(f"\n  ✘ Validator script not found: {_VALIDATOR_FILE}")
    print("  Make sure api_server.py is in the same folder as reference_validator-GUIDE-1.py\n")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("reference_validator", _VALIDATOR_FILE)
_mod = importlib.util.module_from_spec(spec)

print("Loading reference validator…")
spec.loader.exec_module(_mod)
print("Validator ready.\n")

parse_reference          = _mod.parse_reference
validate_reference       = _mod.validate_reference
split_text_into_references = _mod.split_text_into_references
detect_citation_format   = _mod.detect_citation_format      # New Rec E: auto-detect on PDF upload
analyze_thesis_integrity = _mod.analyze_thesis_integrity    # Ch2 integrity checker
SUPPORTED_FORMATS        = _mod.SUPPORTED_FORMATS
FORMAT_APA7              = _mod.FORMAT_APA7
# New Rec E: Cross-Encoder / Intelligent Source Recommendation status
_NEURAL_AVAILABLE       = getattr(_mod, "_NEURAL_AVAILABLE", False)
CROSS_ENCODER_SIM_MODEL = getattr(_mod, "CROSS_ENCODER_SIM_MODEL", "")
CROSS_ENCODER_NLI_MODEL = getattr(_mod, "CROSS_ENCODER_NLI_MODEL", "")

# ── Flask setup ──────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "--break-system-packages", "flask", "flask-cors"])
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS

import io

app = Flask(__name__)
CORS(app)


# ── Serialisers ──────────────────────────────────────────────────────────────
def ref_to_dict(ref) -> dict:
    return {
        "raw":                ref.raw,
        "authors":            ref.authors,
        "year":               ref.year,
        "title":              ref.title,
        "doi":                ref.doi,
        "url":                ref.url,
        "valid_apa":          ref.valid_apa,
        "parse_error":        ref.parse_error,
        "fmt":                ref.fmt,
        "doi_only":           ref.doi_only,
        "formatted_citation": ref.formatted_citation,
    }


def result_to_dict(vr) -> dict:
    return {
        "reference":         ref_to_dict(vr.reference),
        "link_reachable":    vr.link_reachable,
        "link_status":       vr.link_status,
        "title_match":       vr.title_match,
        "title_score":       vr.title_score,
        "title_found":       vr.title_found,
        "authors_match":     vr.authors_match,
        "authors_found":     vr.authors_found,
        "doi_match":         vr.doi_match,
        "doi_found":         vr.doi_found,
        "date_match":        vr.date_match,
        "date_found":        vr.date_found,
        "recommendations":   [],
        "errors":            vr.errors,
        "warnings":          vr.warnings,
        "metadata_source":       vr.metadata_source,
        "verified_metadata":     vr.verified_metadata,   # Rec 3: provenance data
        "similarity_breakdown":  vr.similarity_breakdown, # Rec 4: full calculation details
        "recommended_sources":   vr.recommended_sources,  # New Rec E: Intelligent Source Recommendation
    }


# ── New Rec E: Concurrent batch validation ────────────────────────────────────
# Each validate_reference() call is dominated by network waits (CrossRef,
# arXiv, Semantic Scholar) with some CPU-bound cross-encoder scoring mixed
# in — running a batch sequentially means a 20-reference list pays for
# 20 round trips back to back. A thread pool lets those network waits
# overlap. reference_validator-GUIDE-1.py's diagnostic globals were
# converted to thread-local storage (_tls) and the cross-encoder lazy-load
# guarded with a lock specifically to make this safe.
from concurrent.futures import ThreadPoolExecutor

_BATCH_WORKERS = 6


def _validate_batch(raw_refs: list, fmt: str) -> list:
    def _one(raw):
        ref = parse_reference(str(raw).strip(), fmt)
        return result_to_dict(validate_reference(ref))

    with ThreadPoolExecutor(max_workers=min(_BATCH_WORKERS, len(raw_refs)) or 1) as pool:
        return list(pool.map(_one, raw_refs))


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/api/validate", methods=["POST"])
def validate_endpoint():
    """
    POST /api/validate
    Body: {
      "references": ["ref 1", "ref 2", ...],
      "format":     "apa7" | "mla" | "chicago" | (optional, default: apa7)
    }
    Returns: { "results": [ ValidationResult, ... ] }
    """
    body = request.get_json(force=True, silent=True)
    if not body or "references" not in body:
        return jsonify({"error": "Expected JSON body: { \"references\": [...] }"}), 400

    raw_refs = body["references"]
    if not isinstance(raw_refs, list) or not raw_refs:
        return jsonify({"error": "\"references\" must be a non-empty list of strings"}), 400

    # ── Rec 5: No artificial cap.  Optional pagination for large batches. ──
    #   page:     1-based page number (default: 1)
    #   per_page: results per page    (default: 0 = all results, no pagination)
    page     = max(1, int(body.get("page", 1)))
    per_page = max(0, int(body.get("per_page", 0)))

    # ── Resolve format parameter ──────────────────────────────────────────────
    fmt = body.get("format", FORMAT_APA7).strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        return jsonify({
            "error": f"Unknown format '{fmt}'. "
                     f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        }), 400

    results = _validate_batch(raw_refs, fmt)

    # ── Rec 5: Pagination ─────────────────────────────────────────────────────
    total = len(results)
    if per_page > 0:
        start = (page - 1) * per_page
        end   = start + per_page
        return jsonify({
            "results":     results[start:end],
            "format":      fmt,
            "total":       total,
            "page":        page,
            "per_page":    per_page,
            "total_pages": (total + per_page - 1) // per_page,
        })

    return jsonify({"results": results, "format": fmt, "total": total})


# ── Rec 6: PDF Upload endpoint ───────────────────────────────────────────────
# Temporary storage for uploaded PDFs (keyed by a simple upload ID)
_uploads: dict[str, dict] = {}


def _extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from a PDF byte stream.
    Returns (full_text, page_count).
    Uses pypdf (PyPDF2 successor) — the same library listed in the thesis.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader      # fallback to older name
        except ImportError:
            raise ImportError(
                "PDF parsing requires the 'pypdf' library. "
                "Install it with: pip install pypdf"
            )

    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(reader.pages)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text, page_count


def _extract_pdf_metadata(pdf_bytes: bytes, full_text: str) -> dict:
    """
    New Rec A: Extract structured metadata from a PDF using two strategies:

    1. PDF info dict — embedded metadata (title, author, subject, keywords, creator)
    2. Text-based regex extraction — scans the first ~3000 chars of extracted text
       for patterns that indicate title, authors, abstract, keywords, DOI, year,
       journal, conference, and publisher.

    Returns a dict with all discovered fields (empty string/list for missing ones).
    """
    meta = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": [],
        "doi": "",
        "year": "",
        "journal": "",
        "conference": "",
        "publisher": "",
    }

    # ── Strategy 1: PDF embedded info dict ────────────────────────────────
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        info = reader.metadata
        if info:
            if info.title and len(info.title.strip()) > 3:
                meta["title"] = info.title.strip()
            if info.author:
                # PDF author field can be "Smith, J.; Jones, B." or "Smith, J. and Jones, B."
                raw_authors = re.split(r'[;,]\s*(?=[A-Z])|(?:\band\b)', info.author)
                meta["authors"] = [a.strip() for a in raw_authors if a and len(a.strip()) > 1]
            if info.subject:
                meta["abstract"] = info.subject.strip()
            if hasattr(info, 'keywords') and info.keywords:
                meta["keywords"] = [k.strip() for k in info.keywords.split(",") if k.strip()]
            if info.creator:
                meta["publisher"] = info.creator.strip()
    except Exception as _pdf_info_err:
        # PDF info dict may be corrupt or missing — continue with text-based extraction
        meta["_info_error"] = str(_pdf_info_err)[:100]

    # ── Strategy 2: Text-based extraction from first pages ────────────────
    # Focus on the first ~3000 chars where metadata typically appears.
    head = full_text[:3000] if full_text else ""

    # DOI — scan full text since DOI can appear anywhere
    doi_m = re.search(
        r'(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,}/\S+)',
        full_text, re.IGNORECASE
    )
    if doi_m:
        meta["doi"] = doi_m.group(1).rstrip(".,;)")

    # Year — look for 4-digit year in the head
    year_m = re.search(r'\b((?:19|20)\d{2})\b', head)
    if year_m and not meta["year"]:
        meta["year"] = year_m.group(1)

    # Abstract — look for "Abstract" heading followed by text
    abs_m = re.search(
        r'(?:^|\n)\s*[Aa][Bb][Ss][Tt][Rr][Aa][Cc][Tt][:\s—–-]*\n?\s*(.+?)(?:\n\s*(?:Keywords?|Introduction|1[\s.]|I[\s.])|$)',
        head, re.DOTALL
    )
    if abs_m and not meta["abstract"]:
        abstract_text = " ".join(abs_m.group(1).split()).strip()
        if len(abstract_text) > 30:
            meta["abstract"] = abstract_text[:2000]

    # Keywords — look for "Keywords:" line
    kw_m = re.search(
        r'(?:^|\n)\s*[Kk]ey\s*[Ww]ords?[:\s—–-]+(.+?)(?:\n\s*\n|\n\s*[A-Z1-9])',
        head, re.DOTALL
    )
    if kw_m and not meta["keywords"]:
        kw_text = kw_m.group(1).strip()
        # Keywords can be separated by commas, semicolons, or bullet points
        keywords = re.split(r'[;,•·]\s*', kw_text)
        meta["keywords"] = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 1]

    # Journal / Conference — look for common patterns
    journal_m = re.search(
        r'(?:journal|published\s+in|appears\s+in)[:\s]+([^\n]+)',
        head, re.IGNORECASE
    )
    if journal_m and not meta["journal"]:
        meta["journal"] = journal_m.group(1).strip().rstrip(".,;")

    conf_m = re.search(
        r'(?:proceedings?\s+of|conference\s+on|presented\s+at|symposium\s+on)[:\s]+([^\n]+)',
        head, re.IGNORECASE
    )
    if conf_m and not meta["conference"]:
        meta["conference"] = conf_m.group(1).strip().rstrip(".,;")

    # Publisher — look for common publisher patterns
    pub_m = re.search(
        r'(?:published\s+by|publisher|©\s*\d{4}\s+)([^\n,]+)',
        head, re.IGNORECASE
    )
    if pub_m and not meta["publisher"]:
        meta["publisher"] = pub_m.group(1).strip().rstrip(".,;")

    # Title — if not from PDF info, use the first substantial line of text
    if not meta["title"] and head:
        lines = [l.strip() for l in head.split('\n') if l.strip()]
        for line in lines[:5]:
            # Skip short lines (page numbers, headers) and all-caps labels
            if len(line) > 15 and not re.match(r'^(abstract|keywords?|introduction|references|doi)', line, re.I):
                meta["title"] = line[:300]
                break

    # Authors — if not from PDF info, look for author-line patterns
    if not meta["authors"] and head:
        # Pattern: lines with multiple comma-separated names, often after the title
        author_m = re.search(
            r'(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+(?:\s*[,;&]\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+)+)',
            head
        )
        if author_m:
            raw = author_m.group(1)
            parts = re.split(r'[,;&]\s*(?=[A-Z])', raw)
            meta["authors"] = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]

    # Clean up: remove empty values
    meta = {k: v for k, v in meta.items() if v}

    return meta


# ── Rec 6B: Common file validation helper ────────────────────────────────────
def _validate_uploaded_file(request_obj):
    """
    Validates the uploaded file from a Flask request.
    Returns (pdf_bytes, filename) on success or (None, error_response) on failure.
    """
    if "file" not in request_obj.files:
        return None, (jsonify({"error": "No file uploaded. Send a PDF as the 'file' field."}), 400)

    uploaded = request_obj.files["file"]
    if not uploaded.filename:
        return None, (jsonify({"error": "Empty filename."}), 400)

    filename = uploaded.filename
    if not filename.lower().endswith(".pdf"):
        return None, (jsonify({
            "error": f"Only PDF files are supported. Received: '{filename}'"
        }), 400)

    pdf_bytes = uploaded.read()
    if len(pdf_bytes) == 0:
        return None, (jsonify({"error": "Uploaded file is empty."}), 400)

    return (pdf_bytes, filename), None


def _store_upload(pdf_bytes, filename, file_size, page_count, pdf_metadata=None):
    """Stores the upload and returns the upload_id and preview URL."""
    import hashlib
    upload_id = hashlib.sha256(pdf_bytes[:4096] + filename.encode()).hexdigest()[:16]
    _uploads[upload_id] = {
        "bytes":        pdf_bytes,
        "filename":     filename,
        "file_size":    file_size,
        "page_count":   page_count,
        "pdf_metadata": pdf_metadata or {},
    }
    return upload_id


# ── Rec 6B: Inspect endpoint (verify before processing) ─────────────────────
@app.route("/api/upload/inspect", methods=["POST"])
def upload_inspect_endpoint():
    """
    POST /api/upload/inspect  (multipart/form-data)
    ────────────────────────────────────────────────
    Field: file  – a PDF file

    Lightweight verification step.  Accepts the PDF, extracts ONLY the
    document-level metadata (title, authors, abstract, keywords, DOI, year,
    journal, conference, publisher) and basic file info — WITHOUT splitting
    into individual references or running validation.

    This lets the user:
      1. See the filename, size, and page count.
      2. Preview the PDF via the returned preview_url.
      3. Review the extracted metadata to confirm the correct file.
      4. Then decide whether to proceed with full extraction (/api/upload)
         or batch validation (/api/upload-and-validate).

    Returns:
    {
      "upload_id":    "abc123...",
      "filename":     "references.pdf",
      "file_size":    102400,
      "file_size_display": "100.0 KB",
      "page_count":   3,
      "preview_url":  "/api/upload/preview/abc123...",
      "pdf_metadata": { "title": "...", "authors": [...], ... },
      "status":       "ready",
      "next_steps": {
        "extract_refs":  "POST /api/upload with same file to extract references",
        "batch_validate": "POST /api/upload-and-validate with same file to extract + validate"
      }
    }
    """
    validated, err = _validate_uploaded_file(request)
    if err:
        return err
    pdf_bytes, filename = validated
    file_size = len(pdf_bytes)

    try:
        full_text, page_count = _extract_text_from_pdf(pdf_bytes)

        if not full_text.strip():
            return jsonify({
                "error": "Could not extract text from the PDF. "
                         "The file may be scanned/image-based (OCR is not supported).",
                "filename":  filename,
                "file_size": file_size,
            }), 400

        pdf_metadata = _extract_pdf_metadata(pdf_bytes, full_text)
        detected_format = detect_citation_format(full_text)   # New Rec E

        upload_id = _store_upload(pdf_bytes, filename, file_size, page_count, pdf_metadata)

        return jsonify({
            "upload_id":        upload_id,
            "filename":         filename,
            "file_size":        file_size,
            "file_size_display": _format_bytes(file_size),
            "page_count":       page_count,
            "preview_url":      f"/api/upload/preview/{upload_id}",
            "pdf_metadata":     pdf_metadata,
            "detected_format":  detected_format,
            "status":           "ready",
            "next_steps": {
                "extract_refs":   "POST /api/upload with same file to extract references",
                "batch_validate": "POST /api/upload-and-validate with same file to extract + validate",
            },
        })

    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to inspect PDF: {str(e)}"}), 500


def _format_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1048576:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1048576:.1f} MB"


@app.route("/api/upload", methods=["POST"])
def upload_endpoint():
    """
    POST /api/upload  (multipart/form-data)
    Field: file     – a PDF file
    Field: format   – citation format (optional, default: apa7)

    Stages: upload → extract text → detect references → split → return.
    Each stage reports diagnostics on failure.
    """
    validated, err = _validate_uploaded_file(request)
    if err:
        return err
    pdf_bytes, filename = validated
    file_size = len(pdf_bytes)

    try:
        # Stage 1: Text extraction
        full_text, page_count = _extract_text_from_pdf(pdf_bytes)

        if not full_text.strip():
            return jsonify({
                "error": "Could not extract text from the PDF. "
                         "The file may be scanned/image-based (OCR is not supported).",
                "stage": "text_extraction",
                "filename": filename,
            }), 400

        # New Rec E: auto-detect citation format from the PDF's own content
        # rather than requiring the user to correctly pre-select it.
        detected_format = detect_citation_format(full_text)
        fmt = detected_format

        # Stage 2: Reference detection + splitting
        refs = split_text_into_references(full_text, fmt)

        # Stage 3: Metadata extraction
        pdf_metadata = _extract_pdf_metadata(pdf_bytes, full_text)

        upload_id = _store_upload(pdf_bytes, filename, file_size, page_count, pdf_metadata)

        response = {
            "upload_id":        upload_id,
            "filename":         filename,
            "file_size":        file_size,
            "file_size_display": _format_bytes(file_size),
            "page_count":       page_count,
            "preview_url":      f"/api/upload/preview/{upload_id}",
            "references":       refs,
            "total":            len(refs),
            "raw_preview":      full_text[:5000],
            "pdf_metadata":     pdf_metadata,
            "format":           fmt,
            "detected_format":  detected_format,
            "extraction_info": {
                "text_length":    len(full_text),
                "pages_with_text": sum(1 for p in __import__('pypdf').PdfReader(io.BytesIO(pdf_bytes)).pages if p.extract_text()),
                "refs_extracted": len(refs),
            },
        }

        if not refs:
            response["warning"] = (
                "No individual references could be extracted. "
                "The PDF may not contain a recognizable References/Bibliography section, "
                "or the references may use an unsupported layout. "
                "You can paste references manually in the text input instead."
            )

        return jsonify(response)

    except ImportError as e:
        return jsonify({"error": str(e), "stage": "library"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}", "stage": "processing"}), 500


@app.route("/api/upload/preview/<upload_id>", methods=["GET"])
def upload_preview(upload_id):
    """
    GET /api/upload/preview/<upload_id>
    ────────────────────────────────────
    Returns the uploaded PDF as a downloadable/viewable file.
    The frontend can embed this in an <iframe> or <object> tag for preview.
    """
    if upload_id not in _uploads:
        return jsonify({"error": "Upload not found. It may have expired."}), 404

    entry = _uploads[upload_id]
    return send_file(
        io.BytesIO(entry["bytes"]),
        mimetype="application/pdf",
        as_attachment=False,
        download_name=entry["filename"],
    )


@app.route("/api/upload/info/<upload_id>", methods=["GET"])
def upload_info(upload_id):
    """
    GET /api/upload/info/<upload_id>
    ─────────────────────────────────
    Rec 6B: Now returns extracted metadata alongside file info,
    plus a preview URL for embedding the PDF in an iframe.
    """
    if upload_id not in _uploads:
        return jsonify({"error": "Upload not found. It may have expired."}), 404

    entry = _uploads[upload_id]
    return jsonify({
        "upload_id":        upload_id,
        "filename":         entry["filename"],
        "file_size":        entry["file_size"],
        "file_size_display": _format_bytes(entry["file_size"]),
        "page_count":       entry["page_count"],
        "preview_url":      f"/api/upload/preview/{upload_id}",
        "pdf_metadata":     entry.get("pdf_metadata", {}),
    })


# ── New Rec C: Batch upload-and-validate endpoint ────────────────────────────
@app.route("/api/upload-and-validate", methods=["POST"])
def upload_and_validate_endpoint():
    """
    POST /api/upload-and-validate  (multipart/form-data)
    ─────────────────────────────────────────────────────
    Field: file     – a PDF file containing references
    Field: format   – citation format (optional, default: apa7)

    Combines upload + extraction + validation in a single request.
    Rec 6B: now includes preview_url and file_size_display.
    """
    validated, err = _validate_uploaded_file(request)
    if err:
        return err
    pdf_bytes, filename = validated
    file_size = len(pdf_bytes)

    try:
        full_text, page_count = _extract_text_from_pdf(pdf_bytes)

        if not full_text.strip():
            return jsonify({
                "error": "Could not extract text from the PDF. "
                         "The file may be scanned/image-based (OCR is not supported)."
            }), 400

        # New Rec E: auto-detect citation format from the PDF's own content
        # rather than requiring the user to correctly pre-select it.
        detected_format = detect_citation_format(full_text)
        fmt = detected_format

        refs = split_text_into_references(full_text, fmt)

        if not refs:
            return jsonify({
                "error": "No references could be extracted from the PDF. "
                         "The document may not contain a recognisable reference list."
            }), 400

        pdf_metadata = _extract_pdf_metadata(pdf_bytes, full_text)

        results = _validate_batch(refs, fmt)

        upload_id = _store_upload(pdf_bytes, filename, file_size, page_count, pdf_metadata)

        return jsonify({
            "upload_id":        upload_id,
            "filename":         filename,
            "file_size":        file_size,
            "file_size_display": _format_bytes(file_size),
            "page_count":       page_count,
            "preview_url":      f"/api/upload/preview/{upload_id}",
            "format":           fmt,
            "detected_format":  detected_format,
            "pdf_metadata":     pdf_metadata,
            "results":          results,
            "total":            len(results),
        })

    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500


@app.route("/api/analyze-thesis", methods=["POST"])
def analyze_thesis_endpoint():
    """
    POST /api/analyze-thesis  (multipart/form-data)
    Field: file   – a thesis PDF
    Field: format – citation format (optional, default: apa7)

    Analyzes Chapter 2 citations against the References section.
    Returns a complete integrity report.
    """
    validated, err = _validate_uploaded_file(request)
    if err:
        return err
    pdf_bytes, filename = validated
    file_size = len(pdf_bytes)

    try:
        full_text, page_count = _extract_text_from_pdf(pdf_bytes)
        if not full_text.strip():
            return jsonify({
                "error": "Could not extract text from the PDF.",
                "stage": "text_extraction",
            }), 400

        # New Rec E: auto-detect citation format from the PDF's own content
        # rather than requiring the user to correctly pre-select it.
        detected_format = detect_citation_format(full_text)
        fmt = detected_format

        report = analyze_thesis_integrity(full_text, fmt)

        upload_id = _store_upload(pdf_bytes, filename, file_size, page_count)

        return jsonify({
            "filename":         filename,
            "file_size":        file_size,
            "file_size_display": _format_bytes(file_size),
            "page_count":       page_count,
            "preview_url":      f"/api/upload/preview/{upload_id}",
            "upload_id":        upload_id,
            "format":           fmt,
            "detected_format":  detected_format,
            "report":           report,
        })

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}", "stage": "analysis"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Quick health check."""
    return jsonify({
        "status":            "ok",
        "validator":         "reference_validator-GUIDE-1",
        "supported_formats": SUPPORTED_FORMATS,
        "max_references":    "unlimited",
        "pagination":        "supported (pass page & per_page in request body)",
        "pdf_inspect":       "POST /api/upload/inspect (verify file before processing)",
        "pdf_upload":        "POST /api/upload (extract references from PDF)",
        "pdf_preview":       "GET /api/upload/preview/<upload_id>",
        "pdf_info":          "GET /api/upload/info/<upload_id>",
        "batch_validate":    "POST /api/upload-and-validate",
        "thesis_analysis":   "POST /api/analyze-thesis (Ch2 citation-reference integrity)",
        "cross_encoder":     (f"enabled ({CROSS_ENCODER_SIM_MODEL} + {CROSS_ENCODER_NLI_MODEL})"
                               if _NEURAL_AVAILABLE else
                               "unavailable (sentence-transformers not installed) — falling back to string similarity"),
        "source_recommendation": "CrossRef + arXiv, neural re-ranked (see 'recommended_sources' in /api/validate results)",
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # PORT defaults to 5000 for local dev (matches the frontend's local BASE
    # constant); the deployment Dockerfile sets PORT=7860 for Hugging Face
    # Spaces, which expects a Docker Space to listen on that port.
    _port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  Multi-Format Reference Validator — API Server")
    print(f"  Listening on port {_port}")
    print(f"  Health check: http://localhost:{_port}/api/health")
    print(f"  Formats: {', '.join(SUPPORTED_FORMATS)}")
    print("  Max references per request: unlimited")
    print("  Pagination: supported (page & per_page)")
    print("  PDF inspect: POST /api/upload/inspect")
    print("  PDF upload:  POST /api/upload")
    print("  Batch:       POST /api/upload-and-validate")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=_port, debug=False)
