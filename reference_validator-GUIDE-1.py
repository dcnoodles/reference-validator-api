import os
import re
import sys
import time
import json
import threading
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Dependency bootstrap ──────────────────────────────────────────────────────
def _install(pkg: str, import_as: str | None = None) -> None:
    import importlib.util, subprocess
    name = import_as or pkg.split("[")[0].replace("-", "_")
    if importlib.util.find_spec(name) is None:
        print(f"  Installing {pkg}…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--break-system-packages", pkg]
        )

print("Checking dependencies…")
for _pkg, _imp in [
    ("requests",       "requests"),
    ("beautifulsoup4", "bs4"),
    ("colorama",       "colorama"),
    ("lxml",           "lxml"),
]:
    _install(_pkg, _imp)
print("All dependencies ready.\n")

import requests
from bs4 import BeautifulSoup
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

# ── Neural cross-encoder bootstrap (best-effort) ──────────────────────────────
# The Cross-Encoder / RTE features are the core of this study's framework, but
# they depend on a large ML stack (torch + sentence-transformers) that may not
# install on every machine (no internet, disk space, unsupported platform).
# Rather than crash the whole validator, installation failure here degrades
# gracefully: _NEURAL_AVAILABLE is set False and every neural function returns
# "unavailable" instead of a score, so string-similarity checks still run.
_NEURAL_AVAILABLE = True
try:
    _install("sentence-transformers", "sentence_transformers")
    from sentence_transformers import CrossEncoder
except Exception as _neural_install_err:
    print(f"  ⚠ Neural cross-encoder stack unavailable ({_neural_install_err}). "
          f"Continuing with string-similarity checks only.")
    _NEURAL_AVAILABLE = False

# ── Format type constants ─────────────────────────────────────────────────────
FORMAT_APA7    = "apa7"
FORMAT_MLA     = "mla"
FORMAT_CHICAGO = "chicago"
FORMAT_IEEE    = "ieee"    # New Rec B
SUPPORTED_FORMATS = [FORMAT_APA7, FORMAT_MLA, FORMAT_CHICAGO, FORMAT_IEEE]

# ── Constants ─────────────────────────────────────────────────────────────────
CROSSREF_API        = "https://api.crossref.org/works/{doi}"
CROSSREF_SEARCH_API = "https://api.crossref.org/works"
DOI_RESOLVER        = "https://doi.org/{doi}"
HEADERS       = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ReferenceValidator/2.0; "
        "mailto:validator@research.tool)"
    )
}
TIMEOUT              = 15
TITLE_CHAR_THRESHOLD = 0.82   # character-level similarity
TITLE_WORD_THRESHOLD = 0.60   # word-level Jaccard (catches completely different topics)
TITLE_LEN_RATIO      = 0.85   # cited title must be ≥85% as long as found title (catches truncation)
TITLE_WORD_RECALL    = 0.85   # ≥85% of found's keywords must appear (fuzzily) in cited
# ── Rec 5: Removed MAX_REFERENCES = 10. No artificial cap on reference count.
# ── Bug fix: author matching used to compare the whole "Surname, I." string
# against a single AUTHOR_THRESHOLD (0.75), which was far too lenient for
# short surnames — a single character substitution/insertion/deletion in a
# typical 5-9 letter surname (e.g. "Xaswani" vs "Vaswani") still scores
# 0.86-0.93 under SequenceMatcher. Worse, comparing the combined string also
# conflated two very different situations that score in the SAME range: a
# legitimate initials-completeness difference (cited "Smith, J. A." vs found
# "Smith, J.") scores ~0.857, while an actual surname typo scores ~0.857-
# 0.933 too — no single threshold on the combined string can separate them.
# Replaced with author_name_similarity() below, which compares surname and
# initials separately: SURNAME_THRESHOLD applies only to the surname (where
# a true match is always exactly 1.0 once initials aren't mixed in, giving a
# clean gap above the 0.857-0.933 typo band), and initials are compared as a
# prefix relationship instead of raw similarity.

# ── New Rec E: Neural cross-encoder configuration ─────────────────────────────
# CROSS_ENCODER_SIM_MODEL scores semantic similarity between a cited title and
# a candidate title — used to rescue paraphrased/reworded titles that fail the
# strict string-based thresholds above (see title_similarity()).
# CROSS_ENCODER_NLI_MODEL performs the Recognizing Textual Entailment (RTE)
# classification described in the study: does the cited source's abstract
# (premise) support the sentence in which it is cited (hypothesis)?
#
# Bug fix: CROSS_ENCODER_SIM_MODEL was originally ms-marco-MiniLM-L-6-v2, a
# *query-passage relevance* model (trained so a short query scores highly
# against any longer passage it's relevant to). That is exactly the wrong
# inductive bias here — verified directly that it rated the single word
# "Attention" as 99.76% "similar" to the full title "Attention is all you
# need", which meant a title truncated down to one word still passed as a
# match. Swapped to an actual semantic-*equivalence* (STS) model, which
# correctly separates "is a fragment of" from "means the same as": the same
# truncation scores only ~73% with this model, and two genuinely different
# papers on a similar topic score under 30%, while a true same-content
# paraphrase scores ~75%. STS cross-encoders output an already-normalized
# [0,1] score directly (no extra sigmoid needed — see neural_similarity()).
CROSS_ENCODER_SIM_MODEL   = "cross-encoder/stsb-distilroberta-base"
CROSS_ENCODER_NLI_MODEL   = "cross-encoder/nli-MiniLM2-L6-H768"
NEURAL_OVERRIDE_THRESHOLD = 0.75   # neural score needed to rescue a failed title match
SEMANTIC_SCHOLAR_API        = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
SEMANTIC_SCHOLAR_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_TIMEOUT = 5   # short on purpose — see arxiv_search() for why
SEMANTIC_SCHOLAR_TIMEOUT = 6   # same reasoning — observed rate-limiting during testing

YEAR_PATTERN   = re.compile(r"\b(19|20)\d{2}\b")

# ── DOI-only detection pattern ────────────────────────────────────────────────
DOI_ONLY_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}(?:\.\d+)*/\S+)$",
    re.IGNORECASE
)

# ── APA7 patterns ─────────────────────────────────────────────────────────────
_INIT       = r"[A-Z]\.(?:-[A-Z]\.)?"
_ONE_AUTHOR = (
    r"[A-ZÁÉÍÓÚÄÖÜÀÂÆÇÈÊËÎÏÔŒÙÛÜŸ][A-Za-záéíóúäöüàâæçèêëîïôœùûüÿ'\-]+"
    r"(?:,\s*" + _INIT + r"(?:\s+" + _INIT + r"){0,4})?"
)
APA_AUTHOR_BLOCK = (
    r"(?P<authors>"
    + _ONE_AUTHOR
    + r"(?:\s*,\s*(?:&\s*)?" + _ONE_AUTHOR + r"){0,19}"
    + r"(?:\s*,\s*\.\.\.\s*" + _ONE_AUTHOR + r")?"
    + r")"
)
APA7_PATTERN = re.compile(
    APA_AUTHOR_BLOCK +
    r"\s*\("
    r"(?P<year>\d{4}(?:,\s*(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December|Spring|Summer|Fall|Winter|n\.d\.)"
    r"(?:\s+\d{1,2})?)?)"
    r"\)\.\s*"
    r"(?P<title>[^.]+?)\s*\.\s*"
    r"(?P<rest>.+)",
    re.DOTALL | re.UNICODE,
)
DOI_PATTERN = re.compile(
    r"https?://(?:dx\.)?doi\.org/(?P<doi>10\.\S+)|"
    r"\bdoi:\s*(?P<doi2>10\.\S+)",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s,]+", re.IGNORECASE)


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class ParsedReference:
    raw:         str
    authors:     list[str] = field(default_factory=list)
    year:        str       = ""
    title:       str       = ""
    doi:         str       = ""
    url:         str       = ""
    rest:        str       = ""
    valid_apa:   bool      = False
    parse_error: str       = ""
    fmt:         str       = FORMAT_APA7
    # For DOI-only resolved references
    doi_only:    bool      = False
    formatted_citation: str = ""   # The resolved citation in requested format


@dataclass
class ValidationResult:
    reference:      ParsedReference
    link_reachable: Optional[bool] = None
    link_status:    str            = ""
    title_match:    Optional[bool] = None
    title_score:    float          = 0.0
    title_found:    str            = ""
    authors_match:  Optional[bool] = None
    authors_found:  list[str]      = field(default_factory=list)
    doi_match:      Optional[bool] = None
    doi_found:      str            = ""
    date_match:     Optional[bool] = None
    date_found:     str            = ""
    recommendations: list          = field(default_factory=list)
    errors:          list[str]     = field(default_factory=list)
    warnings:        list[str]     = field(default_factory=list)
    metadata_source: str           = ""   # Rec 1: which lookup path found metadata
    # ── Rec 3: Extended provenance metadata retrieved from the source ─────
    verified_metadata: dict        = field(default_factory=dict)
    # ── Rec 4: Transparent similarity calculation breakdown ───────────────
    similarity_breakdown: dict     = field(default_factory=dict)
    # ── New Rec E: Intelligent Source Recommendation (CrossRef + arXiv,
    #    neural re-ranked) — populated only when verification fails ───────
    recommended_sources: list      = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    New Rec A: Levenshtein (edit) distance — counts minimum single-character
    insertions, deletions, and substitutions to transform s1 into s2.
    Provides a complementary metric to SequenceMatcher's ratio.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,         # insert
                prev_row[j + 1] + 1,     # delete
                prev_row[j] + cost,      # substitute
            ))
        prev_row = curr_row
    return prev_row[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """
    New Rec A: Normalized Levenshtein similarity (0.0 = completely different, 1.0 = identical).
    Complement to char_similarity (SequenceMatcher).
    Formula: 1 - (edit_distance / max_length)
    """
    na, nb = normalize(a), normalize(b)
    max_len = max(len(na), len(nb))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(na, nb) / max_len)


_STOPWORDS = {
    "a","an","the","of","in","on","at","to","and","or","for",
    "with","by","from","is","are","was","were","be","been",
    "its","their","this","that","these","those","it"
}

def word_jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity (ignores stop words & short tokens)."""
    wa = {w for w in normalize(a).split() if w not in _STOPWORDS and len(w) > 2}
    wb = {w for w in normalize(b).split() if w not in _STOPWORDS and len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def word_fuzzy_recall(cited: str, found: str) -> float:
    """
    Fraction of found's content words that have a *fuzzy* match in cited.
    Words within ~18% edit distance count as matching (handles single-char typos,
    pluralisation, etc.) but dissimilar words like 'arctic'/'tropical' do not.
    This catches swapped content words that pure Jaccard misses.
    """
    wc = [w for w in normalize(cited).split() if w not in _STOPWORDS and len(w) > 2]
    wf = [w for w in normalize(found).split() if w not in _STOPWORDS and len(w) > 2]
    if not wf:
        return 1.0
    matched = sum(
        1 for fw in wf
        if any(SequenceMatcher(None, fw, cw).ratio() >= 0.82 for cw in wc)
    )
    return matched / len(wf)


def word_fuzzy_precision(cited: str, found: str) -> float:
    """
    Fraction of cited's content words that have a *fuzzy* match in found.
    Complements word_fuzzy_recall: catches cases where the cited title contains
    random words that do not appear in the actual (found) title.
    """
    wc = [w for w in normalize(cited).split() if w not in _STOPWORDS and len(w) > 2]
    wf = [w for w in normalize(found).split() if w not in _STOPWORDS and len(w) > 2]
    if not wc:
        return 1.0
    matched = sum(
        1 for cw in wc
        if any(SequenceMatcher(None, cw, fw).ratio() >= 0.82 for fw in wf)
    )
    return matched / len(wc)


def numeric_tokens_match(cited: str, found: str) -> bool:
    """
    Returns True only if the sorted list of numeric tokens in cited and found are
    identical.  A single changed/added/removed digit (e.g., '5' → '9999') returns
    False, exposing title edits that character-similarity alone would miss.
    Empty token lists on both sides are considered a match.
    """
    nums_c = sorted(re.findall(r'\b\d+\b', normalize(cited)))
    nums_f = sorted(re.findall(r'\b\d+\b', normalize(found)))
    # If neither title has numbers, no mismatch possible
    if not nums_c and not nums_f:
        return True
    return nums_c == nums_f


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL CROSS-ENCODER & RTE MODULE (New Rec E)
# ══════════════════════════════════════════════════════════════════════════════
# Implements the Cross-Encoder-based semantic verification and Recognizing
# Textual Entailment (RTE) classification named in the study's title and
# Conceptual Framework (Figure 3.1: Syntax Validator -> Metadata Verifier ->
# Intelligent Recommendations -> Neural Re-Ranking -> Output). Models load
# lazily on first use (not at import time) and cache in module-level globals
# so repeated calls don't reload them. Every function degrades to a clearly
# labeled "unavailable" result instead of raising if the neural stack could
# not be installed or a model fails to download.

_sim_cross_encoder = None
_nli_cross_encoder = None
_NLI_LABELS = ("contradiction", "entailment", "neutral")   # sentence-transformers NLI convention


def _sigmoid(x: float) -> float:
    import math
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


_model_load_lock = threading.Lock()   # New Rec E: batch validation runs references
                                        # concurrently — without this, two threads
                                        # racing on first use would both try to load
                                        # the same multi-hundred-MB model at once.


def _resource_base_dir() -> str:
    """
    Resolves the directory this script's bundled resources live in, whether
    running as a plain .py file or frozen into a standalone executable by
    PyInstaller (which extracts bundled data files to a temp dir exposed as
    sys._MEIPASS, not the .py file's own on-disk location).
    """
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


_BUNDLED_MODEL_PATHS = {
    "cross-encoder/stsb-distilroberta-base": "bundled_models/similarity",
    "cross-encoder/nli-MiniLM2-L6-H768":     "bundled_models/nli",
}


def _load_cross_encoder(model_name: str, label: str):
    """
    Loads a CrossEncoder using the quantized ONNX backend when available —
    measured directly: both models loaded via raw PyTorch cost ~1041MB RSS,
    versus ~606MB for the same two models via quantized ONNX (int8,
    avx512). Falls back to the default torch backend if the ONNX runtime
    stack (the 'optimum' package) isn't installed or that specific
    quantized file isn't available for a model, so this degrades
    gracefully rather than failing outright.

    Prefers a locally bundled copy of the model over the Hugging Face Hub
    when one is shipped alongside this script — the standalone desktop app
    build bundles these directly (see bundled_models/) so end users don't
    need internet access or a first-run download just to launch the app.
    The web-hosted deployment has no such bundle, so it transparently falls
    back to downloading from the Hub, same as before.
    """
    local_dir = os.path.join(_resource_base_dir(), _BUNDLED_MODEL_PATHS.get(model_name, ""))
    source = local_dir if os.path.isdir(local_dir) else model_name
    try:
        print(f"  Loading {label} ({source}, ONNX backend)… (first use only)")
        return CrossEncoder(source, backend="onnx",
                            model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"})
    except Exception as e:
        print(f"  ⚠ ONNX backend unavailable for {label} ({e}) — falling back to PyTorch.")
        return CrossEncoder(source)


def _get_similarity_cross_encoder():
    global _sim_cross_encoder
    if not _NEURAL_AVAILABLE:
        return None
    if _sim_cross_encoder is None:
        with _model_load_lock:
            if _sim_cross_encoder is None:   # re-check: another thread may have just finished
                try:
                    _sim_cross_encoder = _load_cross_encoder(CROSS_ENCODER_SIM_MODEL, "neural similarity cross-encoder")
                except Exception as e:
                    print(f"  ⚠ Could not load similarity cross-encoder: {e}")
                    _sim_cross_encoder = False   # sentinel — don't retry on every call
    return _sim_cross_encoder or None


def _get_nli_cross_encoder():
    global _nli_cross_encoder
    if not _NEURAL_AVAILABLE:
        return None
    if _nli_cross_encoder is None:
        with _model_load_lock:
            if _nli_cross_encoder is None:
                try:
                    _nli_cross_encoder = _load_cross_encoder(CROSS_ENCODER_NLI_MODEL, "neural entailment cross-encoder")
                except Exception as e:
                    print(f"  ⚠ Could not load NLI cross-encoder: {e}")
                    _nli_cross_encoder = False
    return _nli_cross_encoder or None


def neural_similarity(text_a: str, text_b: str) -> Optional[float]:
    """
    Cross-Encoder semantic similarity between two texts, in [0, 1].
    Unlike SequenceMatcher/Jaccard, the cross-encoder reads both texts
    together and can recognize paraphrases with little surface overlap.
    Returns None if the neural stack is unavailable.

    CROSS_ENCODER_SIM_MODEL is an STS (semantic textual similarity) model,
    trained with a sigmoid output head so predict() already returns a score
    in [0, 1] directly — applying _sigmoid() again here would distort it
    (e.g. squashing an already-high 0.98 down to ~0.73). That double-sigmoid
    bug existed under the previous ms-marco relevance model, which returned
    unbounded raw logits and needed the extra squashing step.
    """
    if not text_a or not text_b:
        return None
    model = _get_similarity_cross_encoder()
    if model is None:
        return None
    try:
        raw = float(model.predict([(text_a, text_b)])[0])
        return round(max(0.0, min(1.0, raw)), 4)
    except Exception:
        return None


def neural_entailment(premise: str, hypothesis: str) -> dict:
    """
    Recognizing Textual Entailment (RTE) classification: does the premise
    (the cited source's abstract) support the hypothesis (the sentence in
    which the author cites it)?  Returns:
        {"label": "supported"|"refuted"|"no_evidence"|"unavailable",
         "confidence": float, "scores": {...}, "reason": str (on failure)}
    """
    result = {"label": "unavailable", "confidence": 0.0, "scores": {}}
    if not premise or not hypothesis:
        result["reason"] = "Missing premise or hypothesis text."
        return result
    model = _get_nli_cross_encoder()
    if model is None:
        result["reason"] = "Neural entailment model unavailable."
        return result
    try:
        import math
        logits = list(model.predict([(premise, hypothesis)])[0])
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        total = sum(exps) or 1.0
        probs = [v / total for v in exps]
        scores = {label: round(p, 4) for label, p in zip(_NLI_LABELS, probs)}
        top_label = max(scores, key=scores.get)
        _LABEL_MAP = {"entailment": "supported", "contradiction": "refuted", "neutral": "no_evidence"}
        result["label"]      = _LABEL_MAP.get(top_label, "no_evidence")
        result["confidence"] = scores[top_label]
        result["scores"]     = scores
    except Exception as e:
        result["reason"] = f"Entailment classification failed: {str(e)[:100]}"
    return result


def title_similarity(cited: str, found: str) -> tuple[float, bool, dict]:
    """
    Returns (display_score, is_match, breakdown).

    Rec 4: The breakdown dict exposes every intermediate computation so the
    user can see exactly how the score was calculated.

    Two separate regimes based on the length relationship:

    ① cited ≤ found (normal or truncated):
       Requires sym_ratio  ≥ TITLE_CHAR_THRESHOLD   (chars overall)
             AND word_sim  ≥ TITLE_WORD_THRESHOLD    (topic similarity)
             AND len_ratio ≥ TITLE_LEN_RATIO         (no missing chunk)
             AND recall    ≥ TITLE_WORD_RECALL       (all keywords present)
       All four must pass.

    ② cited > found (user added a subtitle / qualifier):
       The symmetric SequenceMatcher ratio shrinks as cited grows even when
       found is perfectly contained in cited — an unfair penalty.
       Instead use *recall* = M / |found|, which ≈ 1.0 whenever found is fully
       covered inside cited regardless of how much extra text cited carries.
       Also require word_recall to catch swapped content words.
    """
    norm_cited = normalize(cited)
    norm_found = normalize(found)

    matcher   = SequenceMatcher(None, norm_cited, norm_found)
    sym_ratio = matcher.ratio()          # 2M / (|cited| + |found|)
    word_sim  = word_jaccard(cited, found)
    combined  = sym_ratio * 0.5 + word_sim * 0.5

    len_cited = len(norm_cited)
    len_found = len(norm_found)
    len_ratio = len_cited / len_found if len_found else 1.0

    recall_kw = word_fuzzy_recall(cited, found)   # keyword coverage (found → cited)
    nums_ok   = numeric_tokens_match(cited, found)
    lev_sim   = levenshtein_similarity(cited, found)   # New Rec A: alternative metric
    neural_score = neural_similarity(cited, found)     # New Rec E: cross-encoder signal

    # ── Build the breakdown dict ──────────────────────────────────────────
    breakdown = {
        "char_similarity":           round(sym_ratio, 4),
        "char_similarity_threshold": TITLE_CHAR_THRESHOLD,
        "levenshtein_similarity":    round(lev_sim, 4),       # New Rec A
        "word_overlap":              round(word_sim, 4),
        "word_overlap_threshold":    TITLE_WORD_THRESHOLD,
        "length_ratio":              round(len_ratio, 4),
        "length_ratio_threshold":    TITLE_LEN_RATIO,
        "keyword_recall":            round(recall_kw, 4),
        "keyword_recall_threshold":  TITLE_WORD_RECALL,
        "numeric_tokens_match":      nums_ok,
        "display_score":             round(combined, 4),
        "display_score_formula":     "char_similarity × 0.5 + word_overlap × 0.5",
        # ── New Rec E: Cross-Encoder semantic signal ───────────────────────
        "neural_similarity":         neural_score,
        "neural_similarity_model":   CROSS_ENCODER_SIM_MODEL if neural_score is not None else None,
    }

    if len_ratio > 1.0:
        # ② cited is longer — compute char recall = M / |found|
        # sym_ratio = 2M/(|cited|+|found|)  →  M = sym_ratio*(|cited|+|found|)/2
        char_recall = sym_ratio * (len_cited + len_found) / (2 * len_found)
        char_recall = min(char_recall, 1.0)
        is_match = (char_recall >= TITLE_CHAR_THRESHOLD and
                    recall_kw  >= TITLE_WORD_RECALL)

        breakdown["regime"]          = "cited_longer"
        breakdown["char_recall"]     = round(char_recall, 4)
        breakdown["match_criteria"]  = (
            f"char_recall ≥ {TITLE_CHAR_THRESHOLD} AND "
            f"keyword_recall ≥ {TITLE_WORD_RECALL}"
        )
        breakdown["criteria_results"] = {
            "char_recall":   {"value": round(char_recall, 4), "threshold": TITLE_CHAR_THRESHOLD, "passed": char_recall >= TITLE_CHAR_THRESHOLD},
            "keyword_recall": {"value": round(recall_kw, 4),  "threshold": TITLE_WORD_RECALL,    "passed": recall_kw >= TITLE_WORD_RECALL},
        }
    else:
        # ① cited is shorter or equal — strict five-threshold check
        precision_kw = word_fuzzy_precision(cited, found)
        is_match = (sym_ratio    >= TITLE_CHAR_THRESHOLD and
                    word_sim     >= TITLE_WORD_THRESHOLD  and
                    len_ratio    >= TITLE_LEN_RATIO        and
                    recall_kw    >= TITLE_WORD_RECALL      and
                    precision_kw >= TITLE_WORD_RECALL)

        breakdown["regime"]             = "standard"
        breakdown["keyword_precision"]           = round(precision_kw, 4)
        breakdown["keyword_precision_threshold"] = TITLE_WORD_RECALL
        breakdown["match_criteria"]     = (
            f"ALL of: char_similarity ≥ {TITLE_CHAR_THRESHOLD}, "
            f"word_overlap ≥ {TITLE_WORD_THRESHOLD}, "
            f"length_ratio ≥ {TITLE_LEN_RATIO}, "
            f"keyword_recall ≥ {TITLE_WORD_RECALL}, "
            f"keyword_precision ≥ {TITLE_WORD_RECALL}"
        )
        breakdown["criteria_results"] = {
            "char_similarity":   {"value": round(sym_ratio, 4),     "threshold": TITLE_CHAR_THRESHOLD, "passed": sym_ratio >= TITLE_CHAR_THRESHOLD},
            "word_overlap":      {"value": round(word_sim, 4),      "threshold": TITLE_WORD_THRESHOLD, "passed": word_sim >= TITLE_WORD_THRESHOLD},
            "length_ratio":      {"value": round(len_ratio, 4),     "threshold": TITLE_LEN_RATIO,      "passed": len_ratio >= TITLE_LEN_RATIO},
            "keyword_recall":    {"value": round(recall_kw, 4),     "threshold": TITLE_WORD_RECALL,    "passed": recall_kw >= TITLE_WORD_RECALL},
            "keyword_precision": {"value": round(precision_kw, 4),  "threshold": TITLE_WORD_RECALL,    "passed": precision_kw >= TITLE_WORD_RECALL},
        }

    # Numeric-token guard (both regimes)
    if is_match and not nums_ok:
        is_match = False

    # ── New Rec E: Neural cross-encoder override ───────────────────────────
    # A title that fails the strict string-based thresholds may simply be a
    # paraphrase (e.g. "Machine Learning for Tumor Detection" vs. "Deep
    # Learning Approaches to Identify Cancerous Growths") — semantically the
    # same paper, near-zero word overlap. A high-confidence cross-encoder
    # score rescues that case. The numeric-token guard still applies: a
    # changed digit is a stronger tampering signal than semantic similarity
    # can override.
    #
    # Bug fix: a truncated title (e.g. cited "Attention is all you" against
    # the real title "Attention is all you need") is a literal prefix/
    # substring of the full title, not a reworded paraphrase — but even the
    # STS model above still rates a close truncation as highly "similar"
    # (fragments of a sentence remain topically near-identical to the whole
    # sentence). Verified directly: dropping the title to a single word
    # still scored high enough to incorrectly pass. A neural score alone
    # can't tell "truncated" from "reworded" apart, but a substring check
    # can: a genuine paraphrase uses different wording and is essentially
    # never a literal contiguous substring of the original, while a
    # truncation always is. So the override is blocked whenever the shorter
    # title is fully contained in the longer one — that pattern is
    # truncation, never legitimate paraphrase, regardless of neural score.
    _is_substring_truncation = (
        (len_cited < len_found and norm_cited in norm_found) or
        (len_found < len_cited and norm_found in norm_cited)
    )
    neural_override = False
    if (not is_match and neural_score is not None and neural_score >= NEURAL_OVERRIDE_THRESHOLD
            and nums_ok and not _is_substring_truncation):
        is_match = True
        neural_override = True
    breakdown["neural_override"]           = neural_override
    breakdown["neural_override_threshold"] = NEURAL_OVERRIDE_THRESHOLD
    breakdown["neural_override_blocked_as_truncation"] = (
        _is_substring_truncation if neural_score is not None else False
    )

    breakdown["is_match"] = is_match

    return combined, is_match, breakdown


def first_url(text: str) -> str:
    m = URL_PATTERN.search(text)
    return m.group(0).rstrip(".,;)") if m else ""


# ── Shared year-token validator ───────────────────────────────────────────────
def validate_year_token(token: str, fmt_name: str) -> tuple[bool, str]:
    """
    Validate that *token* is a clean 4-digit year (1900-2099).
    Returns (is_valid, error_message).

    MLA 9 and Chicago accept only a bare 4-digit year in the year slot.
    Tokens like '2024, 23', '2024-23', '2024/23', '2024 23' are all invalid.
    """
    token = token.strip()

    # Must contain at least one 4-digit year group
    if not re.search(r"(19|20)\d{2}", token):
        return False, (
            f"Invalid year '{token}'. "
            f"{fmt_name} requires a 4-digit year (e.g. 2024)."
        )

    # Whitelist: only a standalone 4-digit year is allowed
    _VALID_BARE_YEAR = re.compile(r"^(19|20)\d{2}$")
    if _VALID_BARE_YEAR.match(token):
        return True, ""

    # Everything else is malformed — give a targeted message
    if re.search(r"(19|20)\d{2}.{0,5}(19|20)\d{2}", token):
        return False, (
            f"Multiple years detected: '{token}'. "
            f"{fmt_name} requires exactly one 4-digit year (e.g. 2024)."
        )
    if re.search(r"\d{4}[\-/]\d+", token):
        return False, (
            f"Invalid year format: '{token}'. "
            f"Year ranges like '2024-23' or '2024/23' are not valid in {fmt_name}. "
            "Use a single 4-digit year (e.g. 2024)."
        )
    if re.search(r"\d{4},\s*\d+", token):
        return False, (
            f"Invalid year format: '{token}'. "
            f"A bare number after the year like '2024, 23' is not valid in {fmt_name}. "
            "Use a single 4-digit year (e.g. 2024)."
        )
    if re.search(r"\d{4}\s+\d+", token):
        return False, (
            f"Invalid year format: '{token}'. "
            f"Extra digits after the year are not allowed in {fmt_name}. "
            "Use a single 4-digit year (e.g. 2024)."
        )
    return False, (
        f"Invalid year format: '{token}'. "
        f"{fmt_name} requires a single 4-digit year (e.g. 2024)."
    )


def extract_doi(text: str) -> str:
    m = DOI_PATTERN.search(text)
    if m:
        return (m.group("doi") or m.group("doi2") or "").rstrip(".,;)")
    return ""


# New Rec E: thread-local (not a plain module global) so concurrent
# validate_reference() calls — added to parallelize batch validation —
# cannot clobber each other's diagnostic error strings mid-request.
_tls = threading.local()

def safe_get(url: str, *, allow_redirects: bool = True) -> Optional[requests.Response]:
    """
    Performs an HTTP GET with error capture.
    New Rec A: sets _tls.request_error with the specific failure reason so
    downstream code can report what went wrong instead of a generic message.
    """
    _tls.request_error = ""
    if not url:
        _tls.request_error = "No URL provided"
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                         allow_redirects=allow_redirects)
        return r
    except requests.exceptions.SSLError as e:
        try:
            return requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                                allow_redirects=allow_redirects, verify=False)
        except Exception as e2:
            _tls.request_error = f"SSL error: {str(e2)[:100]}"
            return None
    except requests.exceptions.ConnectionError:
        _tls.request_error = "Connection refused or DNS resolution failed"
        return None
    except requests.exceptions.Timeout:
        _tls.request_error = f"Request timed out after {TIMEOUT}s"
        return None
    except requests.exceptions.TooManyRedirects:
        _tls.request_error = "Too many redirects (possible redirect loop)"
        return None
    except Exception as e:
        _tls.request_error = f"Request failed: {type(e).__name__}: {str(e)[:100]}"
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT VALIDATORS & PARSERS
# ══════════════════════════════════════════════════════════════════════════════

# ── APA 7 ─────────────────────────────────────────────────────────────────────
def is_valid_apa7(ref: str) -> tuple[bool, str]:
    ref = ref.strip()

    # ── Reject MLA citations submitted as APA 7 ──────────────────────────────
    # MLA signature: Lastname, Firstname (full name, not initials), then ". \"Title.\""
    # No year in parentheses after the author block.
    if (re.match(r"^[A-Z][a-z'\-]+,\s+[A-Z][a-z]+", ref) and
            not re.search(r"\((19|20)\d{2}", ref)):
        # Has quoted article title AND vol./no. → strong MLA signal
        if (re.search(r'"[^"]+"', ref) and
                re.search(r"\bvol\.\s*\d|\bno\.\s*\d|\bpp\.\s*\d", ref, re.IGNORECASE)):
            return False, (
                "This looks like an MLA 9 citation — it uses a full first name, "
                "a quoted title, and vol./no./pp. fields, with no year in parentheses after "
                "the author. Switch the format selector to 'MLA 9' to validate it correctly."
            )

    # ── Reject Chicago Author-Date citations submitted as APA 7 ──────────────
    # Chicago Author-Date: Lastname, Firstname [, and Co-author]*. YYYY. "Title." ...
    # The year appears as a bare 4-digit number after a period following the author block.
    if (re.match(r"^[A-Z][a-z'\-]+,\s+[A-Z][a-z]", ref) and
            not re.search(r"\((19|20)\d{2}", ref) and
            re.search(r"\.\s+(19|20)\d{2}\.\s+", ref)):
        return False, (
            "This looks like a Chicago 17 Author-Date citation — the year appears "
            "as a bare number after the author block (e.g. 'Smith, John. 2021.'), "
            "not in parentheses. Switch the format selector to 'Chicago 17' to validate it correctly."
        )

    if not re.match(r"^[A-ZÁÉÍÓÚ]", ref):
        return False, "Reference must begin with an author surname (capitalize first letter)."

    # ── Repeated-punctuation guard ────────────────────────────────────────────
    # Strip known multi-dot abbreviations (e.g. "n.d.", "et al.") and APA 7's
    # own ellipsis convention for truncating a 20+ author list (e.g. "Li, M.,
    # ... & Bendersky, M.") before testing so neither trips the double-period
    # check. Bug fix: verified on a real reference list that a legitimate
    # ellipsis-truncated author list was rejected as "repeated periods"
    # before this was added.
    _punc_test = re.sub(r'\b(?:et\s+al|n\.d)\b\.', 'X', ref, flags=re.IGNORECASE)
    _punc_test = re.sub(r'\s*\.\.\.\s*', ' X ', _punc_test)
    if re.search(r',{2,}', _punc_test):
        return False, (
            "Repeated commas detected in the author block (e.g. 'Smith, J. A.,,,, Jones'). "
            "APA 7 separates authors with a single comma followed by a space: "
            "'Smith, J. A., Jones, B. C., & Williams, D. E.'"
        )
    if re.search(r'\.{2,}', _punc_test):
        return False, (
            "Repeated periods detected. "
            "Check the reference for accidental double periods (e.g. '..', 'J..A.')."
        )

    # ── Rec 2: Also accept (n.d.) as a valid APA 7 date ─────────────────────
    if not re.search(r"\((19|20)\d{2}", ref) and not re.search(r"\(n\.d\.\)", ref, re.IGNORECASE):
        return False, "Missing publication year in parentheses, e.g. (2023) or (n.d.)."

    # ── Strict whitelist: year parentheses must contain ONLY a valid APA 7 date ─
    # Find the FIRST parenthesised group that starts with a year (the date group).
    year_group_m = re.search(r"\(((19|20)\d{2}[^)]*)\)", ref)
    if year_group_m:
        year_content = year_group_m.group(1).strip()

        # Allowed APA 7 date formats (whitelist):
        #   (YYYY)
        #   (YYYY, Month)           e.g. (2024, March)
        #   (YYYY, Month D)         e.g. (2024, March 5)
        #   (YYYY, Month DD)        e.g. (2024, March 15)
        #   (YYYY, Season)          e.g. (2024, Spring)
        #   (n.d.)                  no date
        _MONTH_OR_SEASON = (
            r"(?:January|February|March|April|May|June|July|August"
            r"|September|October|November|December"
            r"|Spring|Summer|Fall|Winter|n\.d\.)"
        )
        _VALID_YEAR_CONTENT = re.compile(
            r"^(19|20)\d{2}"                          # 4-digit year
            r"(?:,\s*" + _MONTH_OR_SEASON +           # optional: , Month/Season
            r"(?:\s+\d{1,2})?)?"                      # optional: day number
            r"$",
            re.IGNORECASE,
        )

        if not _VALID_YEAR_CONTENT.match(year_content):
            # Provide a targeted error message depending on what we detect
            if re.search(r"(19|20)\d{2}.*(19|20)\d{2}", year_content):
                msg = (
                    f"Multiple years detected in parentheses: ({year_content}). "
                    "APA 7 requires exactly one year, e.g. (2024) or (2024, March 5)."
                )
            elif re.search(r"\d{4}[\-/]\d+", year_content):
                msg = (
                    f"Invalid year format: ({year_content}). "
                    "Year ranges like '2024-23' or '2024/23' are not valid in APA 7. "
                    "Use a single 4-digit year, e.g. (2024)."
                )
            elif re.search(r"\d{4},\s*\d+", year_content):
                msg = (
                    f"Invalid year format: ({year_content}). "
                    "A bare number after the year like '2024, 23' is not valid. "
                    "APA 7 allows only a month name or season after the year, "
                    "e.g. (2024, March 5) or (2024, Spring)."
                )
            elif re.search(r"\d{4}\s+\d+", year_content):
                msg = (
                    f"Invalid year format: ({year_content}). "
                    "Extra digits after the year are not allowed. "
                    "Use a single 4-digit year, e.g. (2024)."
                )
            else:
                msg = (
                    f"Invalid year format: ({year_content}). "
                    "APA 7 requires a single 4-digit year, optionally followed by "
                    "a month name and day, e.g. (2024) or (2024, March 5)."
                )
            return False, msg

    # ── Fix: Require a period immediately after the closing year parenthesis ────
    # Bug: '(2023) Title...' (no period after paren) was accepted.
    # Rec 2: also accept (n.d.). as valid.
    _has_year_period = re.search(r"\((19|20)\d{2}[^)]*\)\.", ref)
    _has_nd_period   = re.search(r"\(n\.d\.\)\.", ref, re.IGNORECASE)
    if not _has_year_period and not _has_nd_period:
        return False, (
            "Missing period after the publication year. "
            "APA 7 requires a period immediately after the closing parenthesis: "
            "e.g., '(2023). Title of article.' not '(2023) Title'."
        )

    if not re.search(r"[.)\w/]$", ref.rstrip()):
        return False, "Reference should end with a period or URL."

    first_segment = ref.split("(")[0]
    if not re.search(r"[A-Z\u00C0-\u00DC][A-Za-z\u00e0-\u00ff]+,\s+[A-Z\u00C0-\u00DC]\.", first_segment):
        return False, (
            "Author format appears incorrect. "
            "APA 7 uses 'Surname, I.' (e.g., Smith, J. A.)."
        )

    # ── Fix: Validate vol / issue / page-range format in journal-style refs ────
    # Bug: random / impossibly large numbers in vol/page were silently accepted.
    # Bug: previous check looked for *markdown italics* around the journal name,
    #      but raw APA 7 references use plain text — journal is never asterisked.
    #      Detect the APA 7 journal pattern directly: ..., VOL(ISSUE), PAGES.
    #      The signature is a bare integer immediately followed by '(' after the title.
    _vol_issue_m = re.search(r',\s*(\d+)\((\d+)\)\s*,\s*([\d\u2013\-]+)', ref)
    if _vol_issue_m:
        _vol_str  = _vol_issue_m.group(1)
        _iss_str  = _vol_issue_m.group(2)
        _pgs_str  = _vol_issue_m.group(3).strip().rstrip('.,')

        if not re.match(r'^\d{1,4}$', _vol_str):
            return False, (
                f"Invalid volume number '{_vol_str}' in APA 7 journal reference. "
                "Volume must be a 1–4 digit integer (e.g., Journal, 12(3), 45–67.)."
            )
        if not re.match(r'^\d{1,4}$', _iss_str):
            return False, (
                f"Invalid issue number '{_iss_str}' in APA 7 journal reference. "
                "Issue number must be a 1–4 digit integer (e.g., Journal, 12(3), 45–67.)."
            )
        if not re.match(r'^\d+(?:[\u2013\-]\d+)?$', _pgs_str):
            return False, (
                f"Invalid page range '{_pgs_str}' in APA 7 journal reference. "
                "Pages must be digits with an optional range separator "
                "(e.g., 45–67 or 45-67)."
            )

    return True, ""


def parse_authors_block(raw: str) -> list[str]:
    # Strip et al. (with or without period, with or without leading comma)
    # Must happen before splitting so it does not attach to the last author name
    raw = re.sub(r",?\s*et\s+al\.?", "", raw, flags=re.IGNORECASE).strip().rstrip(",")
    raw = raw.replace("& ", "")
    parts = re.split(r",\s+(?=[A-Z])", raw)
    authors: list[str] = []
    i = 0
    while i < len(parts):
        surname = parts[i].strip()
        if i + 1 < len(parts) and re.match(r"^[A-Z]\.", parts[i + 1].strip()):
            initials = parts[i + 1].strip()
            if i + 2 < len(parts) and re.match(r"^[A-Z]\.$", parts[i + 2].strip()):
                initials += " " + parts[i + 2].strip()
                i += 3
            else:
                i += 2
            authors.append(f"{surname}, {initials}")
        else:
            if surname:
                authors.append(surname)
            i += 1
    return [a for a in authors if a and a != "et al"]


def parse_apa7_reference(raw: str) -> ParsedReference:
    ref = ParsedReference(raw=raw.strip(), fmt=FORMAT_APA7)
    valid, reason = is_valid_apa7(raw)
    if not valid:
        ref.parse_error = reason
        return ref
    ref.valid_apa = True
    # Temporarily strip "et al." from the string before running the APA7 regex,
    # because APA_AUTHOR_BLOCK does not include an et-al token in its grammar.
    # We still parse against the original for DOI/URL, but the cleaned version
    # lets APA7_PATTERN capture the leading authors correctly.
    raw_for_match = re.sub(r",?\s*et\s+al\.?", "", raw.strip(), flags=re.IGNORECASE)
    m = APA7_PATTERN.match(raw_for_match)
    if m:
        ref.authors = parse_authors_block(m.group("authors"))
        # Extract ONLY a valid standalone 4-digit year; reject garbage like 203333
        year_raw = m.group("year")
        year_digits = re.match(r"((19|20)\d{2})(?!\d)", year_raw)
        ref.year = year_digits.group(1) if year_digits else year_raw
        ref.title = m.group("title").strip().strip('"').strip("*").strip()
        ref.rest  = m.group("rest").strip()
    else:
        # Fallback: APA7_PATTERN didn't match (e.g. n.d., unusual layout)
        # ── Rec 2: Extract year, accepting both numeric years and n.d. ────
        year_m = re.search(r"\(((19|20)\d{2})(?!\d)", raw)
        if year_m:
            ref.year = year_m.group(1)
        elif re.search(r"\(n\.d\.\)", raw, re.IGNORECASE):
            ref.year = "n.d."
        title_m = re.search(r"\)\.\s*(.+?)\s*\.", raw)
        ref.title = title_m.group(1).strip() if title_m else ""
        # ── Rec 2: Parse authors from fallback — everything before '(' ────
        _author_block = raw_for_match.split("(")[0].strip().rstrip("., ")
        if _author_block:
            ref.authors = parse_authors_block(_author_block)
    ref.doi = extract_doi(raw)
    ref.url = (f"https://doi.org/{ref.doi}" if ref.doi else first_url(raw))

    # ── Rec 2: Post-parse verification ────────────────────────────────────
    # The structural validator (is_valid_apa7) may pass a reference whose
    # internal layout prevents the regex from extracting title or authors.
    # Catch this so downstream metadata checks do not run on empty data.
    if not ref.title:
        ref.valid_apa = False
        ref.parse_error = (
            "Could not extract the article title. "
            "APA 7 expects the title immediately after '(Year). ' — "
            "e.g., 'Smith, J. (2024). Article title here. Journal, 1(2), 3–4.'"
        )
    if not ref.authors:
        ref.valid_apa = False
        ref.parse_error = (
            "Could not extract author name(s). "
            "APA 7 expects 'Surname, I.' at the start — "
            "e.g., 'Smith, J. A., & Jones, B. C. (2024).'"
        )

    return ref


# ── MLA 9 ─────────────────────────────────────────────────────────────────────
def parse_mla_reference(raw: str) -> ParsedReference:
    """
    Parse an MLA 9th Edition reference.
    Expected forms:
      Lastname, Firstname. "Article Title." *Journal*, vol. #, no. #, Year, pp. #-#. URL.
      Lastname, Firstname. *Book Title*. Publisher, Year.
    """
    ref = ParsedReference(raw=raw.strip(), fmt=FORMAT_MLA)
    ref.doi = extract_doi(raw)
    ref.url = f"https://doi.org/{ref.doi}" if ref.doi else first_url(raw)

    # Author/Title Boundary: treat everything up to the first quote as the boundary
    author_boundary = raw.split('"')[0] if '"' in raw else raw

    # Basic check: Flexible regex looking for an initial Last, First group, optional 'and' / 'et al.',
    # and a period terminating the author block.
    # Fix: removed '.' from the character class and replaced the '.*?\\.' catch-all so that
    # random text injected after the first author no longer passes validation.
    if not re.match(
        r"^[A-Z][A-Za-z'\-]+,\s+"                                    # Lastname,
        r"[A-Za-z][A-Za-z'\-\s]*"                                    # Firstname [Middle]
        r"(?:,\s+and\s+[A-Z][A-Za-z'\-]+(?:,?\s+[A-Za-z][A-Za-z'\-\s]*)?)?"  # [, and Co-author]
        r"(?:,?\s*et\s+al\.?)?"                                       # [et al.]
        r"\s*\.",                                                      # terminal period
        author_boundary.strip(),
        re.IGNORECASE
    ):
        ref.parse_error = (
            "MLA reference must begin with the author's last name followed by "
            "first name (and optional co-authors), ending with a period before the title."
        )
        return ref

    # ── Reject APA 7 citations submitted as MLA ───────────────────────────────
    # APA signature: year in parentheses immediately after the author block
    if re.search(r"\((19|20)\d{2}(?:,\s*[\w\.]+(?:\s+\d{1,2})?)?\)\.", raw):
        ref.parse_error = (
            "This looks like an APA 7 citation — the year appears in parentheses "
            "after the author (e.g. '(2023).'). "
            "MLA 9 places the year near the end of the entry, not after the author. "
            "Switch the format selector to 'APA 7' to validate it correctly."
        )
        return ref

    # APA signature: author uses initials only (e.g. Smith, J. A.) — MLA uses full first names
    if re.match(r"^[A-Z][A-Za-z\s\-]+,\s+([A-Z]\.\s*)+", raw):
        ref.parse_error = (
            "MLA 9 requires the author's full first name (e.g. 'Smith, John'), "
            "not initials like 'Smith, J. A.' — this looks like an APA 7 citation. "
            "Switch the format selector to 'APA 7' to validate it correctly."
        )
        return ref

    # ── Reject Chicago Author-Date citations submitted as MLA ─────────────────
    # Chicago Author-Date: Lastname, Firstname [, and Co-author]*. YYYY. "Title." ...
    if (re.match(r"^[A-Z][A-Za-z\s\-]+,\s+[A-Z][a-z]", raw) and
            not re.search(r"\((19|20)\d{2}", raw) and
            re.search(r"\.\s+(19|20)\d{2}\.\s+", raw)):
        ref.parse_error = (
            "This looks like a Chicago 17 Author-Date citation — the year appears "
            "as a bare number after the author block "
            "(e.g. 'Smith, John. 2021. \"Title.\"'). "
            "Switch the format selector to 'Chicago 17' to validate it correctly."
        )
        return ref

    # Must have a period somewhere
    if raw.count(".") < 2:
        ref.parse_error = "MLA reference appears incomplete (too few periods)."
        return ref

    # ── Fix: Repeated-punctuation guard ──────────────────────────────────────
    # Bug: ",," / ".." / ";;" were silently accepted.
    # Bug fix: also tolerate the standard ellipsis convention for a
    # truncated 20+ author list (e.g. "... & Bendersky, M."), which
    # otherwise reads as three repeated periods.
    _punc_check = re.sub(
        r'\b(?:et\s+al|e\.g|i\.e|n\.d)\b\.', 'X', raw, flags=re.IGNORECASE
    )
    _punc_check = re.sub(r'\s*\.\.\.\s*', ' X ', _punc_check)
    if re.search(r'\.{2,}|,{2,}|;{2,}', _punc_check):
        ref.parse_error = (
            "Repeated punctuation detected (e.g. '..' or ','','). "
            "Check your reference for double periods, commas, or semicolons."
        )
        return ref

    # ── Fix: No random text between the author block and the title ────────────
    # Bug: 'Smith, John. GARBAGE "Title."' passed because the author regex
    # used '.*?\\.' as a permissive catch-all.
    if '"' in raw:
        _pre_title = raw.split('"')[0].rstrip()
        if not _pre_title.endswith('.'):
            ref.parse_error = (
                "The author block must end with a period immediately before the "
                "quoted title. MLA 9 format: 'Author(s). \"Title.\"'"
            )
            return ref
        _author_content = _pre_title[:-1].strip()   # drop the terminal period
        # A period-space-letter sequence inside the author block means an extra
        # sentence (random text) was injected between the authors and the title.
        if re.search(r'\.\s+[A-Za-z]', _author_content):
            ref.parse_error = (
                "Unexpected text found between the author block and the title. "
                "MLA 9 format: 'Author(s). \"Article Title.\"' — no extra text "
                "should appear between the final author and the opening quotation mark."
            )
            return ref

    # ── Fix: Require a quoted title for journal / article references ──────────
    # Bug: a reference with *Journal* + vol/pp but no quotation marks passed.
    _has_italic_source = bool(re.search(r'\*[^*]+\*', raw))
    _has_vol = bool(re.search(r'\bvol\.\s*\d', raw, re.IGNORECASE))
    _has_pp  = bool(re.search(r'\bpp\.\s*\d',  raw, re.IGNORECASE))
    _has_no  = bool(re.search(r'\bno\.\s*\d',  raw, re.IGNORECASE))
    _is_journal_article = _has_italic_source and (_has_vol or _has_pp or _has_no)

    title_m = re.search(r'"([^"]+)"', raw)
    if not title_m:
        if _is_journal_article:
            ref.parse_error = (
                "Journal article titles in MLA 9 must be enclosed in double "
                "quotation marks: '\"Article Title.\"' — no quoted title found."
            )
            return ref
        # For books the title may be in asterisks; fall through to the book path below.

    # ── Fix: For journal articles, both vol. and pp. must be present ──────────
    # Bug: removing vol/pp from a journal ref still validated.
    if _is_journal_article:
        if not _has_vol:
            ref.parse_error = (
                "MLA 9 journal articles require a volume number: 'vol. N'. "
                "Volume number is missing."
            )
            return ref
        if not _has_pp:
            ref.parse_error = (
                "MLA 9 journal articles require page numbers: 'pp. N–N'. "
                "Page numbers are missing."
            )
            return ref

    # ── Fix: Validate vol / no / pp numeric format ────────────────────────────
    # Bug: random / impossibly large numbers in these fields were accepted.
    _vol_m = re.search(r'\bvol\.\s*(\S+)', raw, re.IGNORECASE)
    if _vol_m:
        _vol_val = _vol_m.group(1).rstrip('.,;')
        if not re.match(r'^\d{1,4}$', _vol_val):
            ref.parse_error = (
                f"Invalid volume number 'vol. {_vol_val}'. "
                "MLA 9 requires a numeric volume number, e.g. vol. 5."
            )
            return ref

    _no_m = re.search(r'\bno\.\s*(\S+)', raw, re.IGNORECASE)
    if _no_m:
        _no_val = _no_m.group(1).rstrip('.,;')
        if not re.match(r'^\d{1,4}$', _no_val):
            ref.parse_error = (
                f"Invalid issue number 'no. {_no_val}'. "
                "MLA 9 requires a numeric issue number, e.g. no. 3."
            )
            return ref

    _pp_m = re.search(r'\bpp\.\s*(\S+)', raw, re.IGNORECASE)
    if _pp_m:
        _pp_val = _pp_m.group(1).rstrip('.,;')
        if not re.match(r'^\d+(?:[-–]\d+)?$', _pp_val):
            ref.parse_error = (
                f"Invalid page range 'pp. {_pp_val}'. "
                "MLA 9 requires a page range like 'pp. 10–25' or single page 'pp. 10'."
            )
            return ref

    # ── Parse authors ──────────────────────────────────────────────────────────
    # Authors come before the first quoted title or italic marker
    author_end_m = re.search(r'(?:\.\s+")|\.\s+\*', raw)
    authors_raw = raw[:author_end_m.start()] if author_end_m else raw.split(".")[0]
    
    # Handle et al. immediately following author list
    if re.search(r",?\s*et\s+al\.?", authors_raw, flags=re.IGNORECASE):
        authors_raw = re.sub(r",?\s*et\s+al\.?", "", authors_raw, flags=re.IGNORECASE)

    # MLA uses "Lastname, Firstname, and Firstname2 Lastname2."
    for part in re.split(r",\s*and\s+", authors_raw, flags=re.IGNORECASE):
        a = part.strip().rstrip(".,")
        if a and re.search(r"[A-Za-z]", a):
            ref.authors.append(a)

    # ── Parse title ───────────────────────────────────────────────────────────
    # title_m was already computed earlier in the validation guards (re-use it).
    # For books without quotation marks, fall back to the second dot-segment.
    if title_m:
        ref.title = title_m.group(1).strip().rstrip(".")
    else:
        # Book title (between first and second period after authors)
        parts = raw.split(".")
        if len(parts) > 1:
            ref.title = parts[1].strip().strip("*").strip()

    # ── Parse year ────────────────────────────────────────────────────────────
    # MLA places the year near the end: ..., vol. 5, no. 3, 2024, pp. 10-20.
    # We must find the year AND verify it is a clean 4-digit token, not a
    # malformed value like '2024, 23', '2024-23', '2024/23', '2024 23'.
    year_m = YEAR_PATTERN.search(raw)
    ref.year = year_m.group(0) if year_m else ""

    if ref.year:
        # Extract the raw year-bearing token — everything from the year start
        # up to the next field boundary (", pp." / ". " / URL / end).
        # This correctly captures "2024, 23", "2024-23", "2024 23" as full tokens.
        year_pos  = year_m.start()
        year_end  = year_m.end()
        rest_after = raw[year_end:]
        stop_m = re.search(
            r"(?=,\s+(?:pp\.|no\.|vol\.|[A-Z]|https?:))|(?=\.\s)|(?=\s*$)|(?=\s+https?:)",
            rest_after, re.IGNORECASE
        )
        token_end  = year_end + (stop_m.start() if stop_m else len(rest_after))
        raw_year_token = raw[year_pos:token_end].strip().rstrip(".,;")

        year_ok, year_err = validate_year_token(raw_year_token, "MLA 9")
        if not year_ok:
            ref.parse_error = year_err
            return ref

    if not ref.title:
        ref.parse_error = (
            "Could not extract title. MLA uses quoted titles for articles: "
            "\"Title of Article.\""
        )
        return ref

    if not ref.year:
        ref.parse_error = "Could not extract publication year from MLA reference."
        return ref

    # ── Rec 2 Fix 1: Ending character check ───────────────────────────────
    # MLA 9 references must end with a period or a URL.
    # A stray comma, semicolon, or other character at the end is an error.
    if not re.search(r'[.)\w/]$', raw.rstrip()):
        ref.parse_error = (
            "MLA 9 reference must end with a period or URL — "
            "found unexpected trailing character."
        )
        return ref

    # ── Rec 2 Fix 2: Author extraction verification ───────────────────────
    # The author regex may pass validation but the split may produce nothing.
    if not ref.authors:
        ref.parse_error = (
            "Could not extract author name(s). "
            "MLA 9 requires: 'Lastname, Firstname.' at the beginning — "
            "e.g., 'Smith, John, and Jane Doe.'"
        )
        return ref

    # ── Rec 2 Fix 3: Quoted title must end with punctuation inside quotes ─
    # MLA 9 rule: article titles end with a period (or ? or !) inside the
    # closing quotation mark: "Title of Article." not "Title of Article"
    if title_m:
        _inner_title = title_m.group(1)
        if _inner_title and not re.search(r'[.?!]$', _inner_title.rstrip()):
            ref.parse_error = (
                "MLA 9 requires the article title to end with a period, "
                "question mark, or exclamation mark inside the closing "
                "quotation mark — e.g., '\"Title of Article.\"' not "
                "'\"Title of Article\"'."
            )
            return ref

    # ── Rec 2 Fix 4: Detect journal articles even without italic markers ──
    # The original _is_journal_article required *Journal* (asterisks),
    # but users submitting plain text don't use markdown italics.
    # Detect the vol./no./pp. pattern alone as a journal article signal
    # and enforce the quoted-title + pp. requirements accordingly.
    if not _is_journal_article and (_has_vol or _has_no) and _has_pp:
        if not title_m:
            ref.parse_error = (
                "This appears to be a journal article (has vol./no./pp.) "
                "but the article title is not in quotation marks. "
                "MLA 9 format: 'Author. \"Article Title.\" Journal, "
                "vol. N, no. N, Year, pp. N–N.'"
            )
            return ref

    ref.valid_apa = True   # reusing flag to mean "valid format"
    return ref


# ── Chicago 17 ────────────────────────────────────────────────────────────────
def parse_chicago_reference(raw: str) -> ParsedReference:
    """
    Parse a Chicago 17th Edition reference.
    Supports both Author-Date and Notes-Bibliography styles.
    Author-Date:     Lastname, Firstname. Year. "Title." Journal vol. (no.): pages. DOI.
    Notes-Biblio:    Lastname, Firstname. "Title." Journal vol., no. # (Year): pages. DOI.
    """
    ref = ParsedReference(raw=raw.strip(), fmt=FORMAT_CHICAGO)
    ref.doi = extract_doi(raw)
    ref.url = f"https://doi.org/{ref.doi}" if ref.doi else first_url(raw)

    if not re.match(r"^[A-Z][a-z'\-]+,\s+[A-Z]", raw):
        ref.parse_error = (
            "Chicago reference must begin with the author's last name, "
            "e.g.: Smith, John."
        )
        return ref

    # ── Reject APA 7 citations submitted as Chicago ───────────────────────────
    # APA signatures:
    #   (a) Year in parentheses after the author block: (YYYY).
    #   (b) Author block uses initials: Surname, I. or Surname, I. A.
    # Both must be true to avoid false positives.
    _has_apa_year    = bool(re.search(r"\((19|20)\d{2}(?:,\s*[\w\.]+(?:\s+\d{1,2})?)?\)\.", raw))
    _has_apa_initial = bool(re.match(r"^[A-Z][a-z'\-]+,\s+[A-Z]\.", raw))
    if _has_apa_year and _has_apa_initial:
        ref.parse_error = (
            "This looks like an APA 7 citation — the year appears in parentheses "
            "after the author block (e.g. 'Smith, J. A. (2021).'), and the author "
            "uses initials rather than a full first name. "
            "Chicago uses full first names and places the year differently. "
            "Switch the format selector to 'APA 7' to validate it correctly."
        )
        return ref

    # ── Reject MLA citations submitted as Chicago ─────────────────────────────
    # MLA signature: full name, quoted title, vol./no./pp., no year right after author.
    # Exception: Chicago NB uses "no. N (YEAR): pages" — the year is in parentheses
    # after the issue number, which is structurally different from MLA's comma-separated
    # "vol. N, no. N, YEAR, pp. N" layout.  We must NOT reject Chicago NB here.
    _looks_mla = (
        re.match(r"^[A-Z][a-z'\-]+,\s+[A-Z][a-z]+", raw) and
        re.search(r'"[^"]+"', raw) and
        re.search(r"\bvol\.\s*\d|\bno\.\s*\d|\bpp\.\s*\d", raw, re.IGNORECASE) and
        not re.match(r"^[A-Z][a-z'\-]+,\s+[A-Z][a-z]+\.\s+(19|20)\d{2}\.", raw)
    )
    # Chicago NB uses "no. N (YEAR):" — year in parens right after issue number
    _looks_chicago_nb = bool(re.search(r"\bno\.\s*\d+\s*\((19|20)\d{2}", raw, re.IGNORECASE))
    if _looks_mla and not _looks_chicago_nb:
        ref.parse_error = (
            "This looks like an MLA 9 citation — it uses a quoted title and "
            "vol./no./pp. fields in MLA style. "
            "Switch the format selector to 'MLA 9' to validate it correctly."
        )
        return ref

    # ── Rec 2 Fix 1: Repeated-punctuation guard ──────────────────────────
    # APA 7 and MLA 9 both have this guard; Chicago was missing it.
    # Bug fix: also tolerate the ellipsis convention for a truncated
    # 20+ author list, which otherwise reads as three repeated periods.
    _punc_check_ch = re.sub(
        r'\b(?:et\s+al|e\.g|i\.e|n\.d)\b\.', 'X', raw, flags=re.IGNORECASE
    )
    _punc_check_ch = re.sub(r'\s*\.\.\.\s*', ' X ', _punc_check_ch)
    if re.search(r'\.{2,}|,{2,}|;{2,}', _punc_check_ch):
        ref.parse_error = (
            "Repeated punctuation detected (e.g. '..' or ','','). "
            "Check your reference for double periods, commas, or semicolons."
        )
        return ref

    # ── Rec 2 Fix 2: Minimum period count ─────────────────────────────────
    # A valid Chicago reference has at minimum: author block period,
    # title-ending period, and source period = at least 3 periods.
    # Two periods are needed as a bare minimum (author. and ending.).
    if raw.count(".") < 2:
        ref.parse_error = (
            "Chicago 17 reference appears incomplete — "
            "too few periods found (expected at least author block + ending)."
        )
        return ref

    # ── Try Author-Date style: Author. YYYY. "Title." ... ───────────────────
    # The ad_m regex requires ". YYYY." — year flanked by periods — so a token like
    # "2024, 23" or "2024-23" will NOT be captured here (the period after the year
    # won't be found), forcing fallback to nb_m / YEAR_PATTERN which silently
    # accept the garbage year.  We therefore check the raw text for a candidate
    # Author-Date year token and validate it BEFORE trying the regex.

    # Pattern: <author block period>  <space>  <candidate year token>  <period or quote>
    # The token can include: "2024", "2024-23", "2024, 23", "2024 23", etc.
    _ad_candidate = re.match(
        r"^.+?\.\s+((19|20)\d{2}[^\.\"]*)[\.\"]",
        raw, re.DOTALL
    )
    if _ad_candidate:
        _raw_year_tok = _ad_candidate.group(1).strip().rstrip(".,; ")
        # Only validate if this token is in the Author-Date year position
        # (before the title, not deep inside the reference body)
        _yr_ok, _yr_err = validate_year_token(_raw_year_tok, "Chicago 17")
        if not _yr_ok:
            ref.parse_error = _yr_err
            return ref

    # ── Fix: No random text allowed between the year and the title ────────────
    # Bug: 'Author. 2023. RANDOM WORDS "Title."' passed because the ad_m regex
    # consumed everything after ". YYYY. " into group(3) uncritically.
    # For Author-Date: after ". YYYY." the next non-space must open a title
    # (either a " for article or * for book / online source).
    # For references that do have a quoted title elsewhere in the string, check
    # that nothing sits between the year and the opening quote.
    _ad_year_m = re.search(r'\.\s+((?:19|20)\d{2})\.\s+', raw)
    if _ad_year_m:
        _after_year_start = _ad_year_m.end()
        _first_quote_pos  = raw.find('"', _after_year_start)
        _first_star_pos   = raw.find('*', _after_year_start)
        # Pick whichever title marker comes first
        _title_start = min(
            p for p in (_first_quote_pos, _first_star_pos) if p >= 0
        ) if (_first_quote_pos >= 0 or _first_star_pos >= 0) else -1

        if _title_start >= 0:
            _gap = raw[_after_year_start:_title_start].strip()
            if _gap:
                ref.parse_error = (
                    "Unexpected text found between the year and the title in "
                    "Chicago 17 Author-Date format. "
                    "Use: 'Author. Year. \"Title.\" ...' with no text between the "
                    "year period and the opening quotation mark."
                )
                return ref

    ad_m = re.match(
        r"^(.+?)\.\s+((?:19|20)\d{2})\.\s+[\"*]?(.+?)[\"*]?\s*\.",
        raw, re.DOTALL
    )
    # ── Try Notes-Bibliography style (year in parentheses later) ─────────────
    nb_m = re.match(r'^(.+?)\.\s+"([^"]+)"', raw, re.DOTALL)

    if ad_m:
        authors_raw = ad_m.group(1)
        ref.year    = ad_m.group(2)
        ref.title   = ad_m.group(3).strip().strip('"').strip("*").strip()
    elif nb_m:
        authors_raw = nb_m.group(1)
        ref.title   = nb_m.group(2).strip().rstrip(".")

        # NB style: year lives in parentheses, e.g. (2024), or as a bare year.
        # Extract the raw parenthesised content and validate it.
        _paren_m = re.search(r"\(([^)]+)\)", raw)
        if _paren_m:
            _paren_content = _paren_m.group(1).strip()
            # Only treat as year if it looks year-like (starts with 19xx/20xx)
            if re.match(r"(19|20)\d{2}", _paren_content):
                _yr_ok, _yr_err = validate_year_token(_paren_content, "Chicago 17")
                if not _yr_ok:
                    ref.parse_error = _yr_err
                    return ref
                ref.year = re.match(r"(19|20)\d{2}", _paren_content).group(0)
            else:
                # Parentheses held something else (e.g. issue number); fall back
                _fallback_ym = YEAR_PATTERN.search(raw)
                ref.year = _fallback_ym.group(0) if _fallback_ym else ""
        else:
            _fallback_ym = YEAR_PATTERN.search(raw)
            ref.year = _fallback_ym.group(0) if _fallback_ym else ""

        # Validate whatever year we ended up with from plain text as well
        if ref.year:
            # Find the actual token in the raw string surrounding that year position
            _ym2 = re.search(r"(19|20)\d{2}", raw)
            if _ym2:
                _tok_start = _ym2.start()
                _tok_end_m = re.search(r"[,.\s)]", raw[_ym2.end():])
                _tok_end   = _ym2.end() + (_tok_end_m.start() if _tok_end_m else 0)
                _plain_tok = raw[_tok_start:_tok_end].strip()
                _yr_ok, _yr_err = validate_year_token(_plain_tok, "Chicago 17")
                if not _yr_ok:
                    ref.parse_error = _yr_err
                    return ref
    else:
        authors_raw = raw.split(".")[0]
        year_m      = YEAR_PATTERN.search(raw)
        ref.year    = year_m.group(0) if year_m else ""
        title_m     = re.search(r'"([^"]+)"', raw)
        ref.title   = title_m.group(1).strip() if title_m else ""

    for part in re.split(r",\s+and\s+", authors_raw, flags=re.IGNORECASE):
        a = part.strip().rstrip(".,")
        if a and re.search(r"[A-Za-z]", a):
            ref.authors.append(a)

    if not ref.title:
        ref.parse_error = (
            "Could not extract title from Chicago reference. "
            "Articles should use quoted titles: \"Title of Article.\""
        )
        return ref

    if not ref.year:
        ref.parse_error = "Could not extract publication year from Chicago reference."
        return ref

    # ── Fix: Validate vol / issue / page-range format ─────────────────────────
    # Bug: random / impossibly large numbers in these fields were accepted.
    _ch_vol_m = re.search(r'\bvol\.\s*(\S+)', raw, re.IGNORECASE)
    if _ch_vol_m:
        _ch_vol = _ch_vol_m.group(1).rstrip('.,;:()')
        if not re.match(r'^\d{1,4}$', _ch_vol):
            ref.parse_error = (
                f"Invalid volume number 'vol. {_ch_vol}' in Chicago 17 reference. "
                "Volume must be a plain integer (e.g., vol. 5)."
            )
            return ref

    _ch_no_m = re.search(r'\bno\.\s*(\S+)', raw, re.IGNORECASE)
    if _ch_no_m:
        _ch_no = _ch_no_m.group(1).rstrip('.,;:()')
        if not re.match(r'^\d{1,4}$', _ch_no):
            ref.parse_error = (
                f"Invalid issue number 'no. {_ch_no}' in Chicago 17 reference. "
                "Issue number must be a plain integer (e.g., no. 3)."
            )
            return ref

    # Chicago page range: "Journal vol. (no.): pages." — colon-separated pages
    _ch_pg_m = re.search(r':\s*([\w–\-]+?)(?=\s*[\.,]|\s*https?:|\s*$)', raw)
    if _ch_pg_m:
        _ch_pg = _ch_pg_m.group(1).strip().rstrip('.,')
        # Make sure this looks like a page token (has digits), not something else
        if re.search(r'\d', _ch_pg) and not re.match(r'^https?:|^10\.', _ch_pg):
            if not re.match(r'^\d+(?:[-–]\d+)?$', _ch_pg):
                ref.parse_error = (
                    f"Invalid page range '{_ch_pg}' in Chicago 17 reference. "
                    "Pages must be digits with an optional range separator "
                    "(e.g., 45–67 or 45-67)."
                )
                return ref

    ref.valid_apa = True

    # ── Rec 2 Fix 3: Ending character check ───────────────────────────────
    if not re.search(r'[.)\w/]$', raw.rstrip()):
        ref.valid_apa = False
        ref.parse_error = (
            "Chicago 17 reference must end with a period or URL — "
            "found unexpected trailing character."
        )
        return ref

    # ── Rec 2 Fix 4: Author extraction verification ───────────────────────
    if not ref.authors:
        ref.valid_apa = False
        ref.parse_error = (
            "Could not extract author name(s). "
            "Chicago 17 requires: 'Lastname, Firstname.' at the beginning — "
            "e.g., 'Smith, John, and Jane Doe.'"
        )
        return ref

    return ref


# ── IEEE ──────────────────────────────────────────────────────────────────────
def parse_ieee_reference(raw: str) -> ParsedReference:
    """
    New Rec B: Parse and validate an IEEE-style citation.

    IEEE format examples:
      Journal article:
        [1] A. B. Smith and C. D. Jones, "Title of article," *Journal Name*,
            vol. 5, no. 3, pp. 45-50, 2021.
      Conference paper:
        [2] A. B. Smith, "Title of paper," in *Proc. Conf. Name*, City, 2021,
            pp. 1-5.
      Book:
        [3] A. B. Smith, *Title of Book*. City, ST: Publisher, 2021.

    Required fields:
      - Authors (initials first: A. B. Smith)
      - Title (in quotes for articles/papers, italicized for books)
      - Year (4-digit, typically near the end)

    Optional fields (validated if present):
      - Bracket number [N]
      - vol., no., pp.
      - Journal or conference name
      - DOI / URL
    """
    ref = ParsedReference(raw=raw.strip(), fmt=FORMAT_IEEE)

    # ── Cross-format rejection ────────────────────────────────────────────
    # Reject APA 7 (has year in parentheses after author)
    if re.search(r"\)\.\s+[A-Z]", raw) and re.search(r"\(\d{4}", raw):
        ref.parse_error = (
            "This looks like an APA 7 reference, not IEEE. "
            "Switch to APA 7 format."
        )
        return ref

    # Reject MLA (has vol./no./pp. with 'Lastname, Firstname.' author style
    # and quoted title but NO bracket number)
    if re.search(r"\bvol\.", raw) and re.search(r"\bpp\.", raw) and not re.match(r"\s*\[", raw):
        # Check for MLA author style: Lastname, Firstname (full name, not initials)
        if re.match(r"^[A-Z][a-z]+,\s+[A-Z][a-z]{2,}", raw):
            ref.parse_error = (
                "This looks like an MLA reference, not IEEE. "
                "Switch to MLA format."
            )
            return ref

    # ── Repeated-punctuation guard ────────────────────────────────────────
    # Bug fix: also tolerate the ellipsis convention for a truncated
    # long author list, which otherwise reads as three repeated periods.
    _punc_check = re.sub(
        r'\b(?:et\s+al|e\.g|i\.e|n\.d)\b\.', 'X', raw, flags=re.IGNORECASE
    )
    _punc_check = re.sub(r'\s*\.\.\.\s*', ' X ', _punc_check)
    if re.search(r'\.{2,}|,{2,}|;{2,}', _punc_check):
        ref.parse_error = (
            "Repeated punctuation detected (e.g. '..' or ','','). "
            "Check your reference for double periods, commas, or semicolons."
        )
        return ref

    # ── Strip optional bracket number [N] ─────────────────────────────────
    working = raw.strip()
    bracket_m = re.match(r'\s*\[(\d+)\]\s*', working)
    if bracket_m:
        working = working[bracket_m.end():]

    # ── Extract DOI and URL ───────────────────────────────────────────────
    ref.doi = extract_doi(raw)
    ref.url = (f"https://doi.org/{ref.doi}" if ref.doi else first_url(raw))

    # ── Parse authors ─────────────────────────────────────────────────────
    # IEEE uses initials-first: A. B. Smith, C. D. Jones, and E. F. Brown
    # The author block ends at the first quoted title or comma-delimited title.
    # Split at the first occurrence of a quoted string or at the title marker.
    title_start_m = re.search(r'[,]\s*[""\u201c]', working)
    if title_start_m:
        authors_block = working[:title_start_m.start()].strip()
        after_authors = working[title_start_m.end() - 1:]  # keep the quote
    else:
        # No quoted title — try to find the title via comma-then-italic or fallback
        first_comma = working.find(",")
        if first_comma > 5:
            authors_block = working[:first_comma].strip()
            after_authors = working[first_comma + 1:].strip()
        else:
            authors_block = ""
            after_authors = working

    # Parse individual author names from the block
    if authors_block:
        # Split on " and " or ", " but keep the structure
        # IEEE: "A. B. Smith, C. D. Jones, and E. F. Brown"
        auth_block_clean = re.sub(r'\band\b', ',', authors_block, flags=re.IGNORECASE)
        parts = [p.strip().rstrip(".,") for p in auth_block_clean.split(",") if p.strip()]

        # Merge initials with surnames: ["A.", "B.", "Smith"] → "A. B. Smith"
        merged = []
        current = []
        for part in parts:
            if re.match(r'^[A-Z]\.$', part) or re.match(r'^[A-Z]\.\s*[A-Z]\.$', part):
                # This is an initial or pair of initials
                current.append(part)
            elif re.match(r'^[A-Z][a-z]', part):
                # This is a surname — attach to preceding initials
                if current:
                    current.append(part)
                    merged.append(" ".join(current))
                    current = []
                else:
                    merged.append(part)
            else:
                if current:
                    current.append(part)
                    merged.append(" ".join(current))
                    current = []
                else:
                    merged.append(part)
        if current:
            merged.append(" ".join(current))

        ref.authors = [a for a in merged if a and len(a) > 1]

    # ── Extract title ─────────────────────────────────────────────────────
    # Quoted title: "Title of article"
    title_m = re.search(r'[""\u201c]([^""\u201d]+)[""\u201d]', raw)
    if title_m:
        ref.title = title_m.group(1).strip().rstrip(",.")
    else:
        # No quoted title — only accept unquoted title for book-style references
        # (i.e., no vol./no./pp. markers which indicate a journal article)
        _has_journal_markers = bool(re.search(r'\bvol\.|\bno\.|\bpp\.', raw, re.IGNORECASE))
        if _has_journal_markers:
            # This is a journal/conference article missing its quoted title
            ref.parse_error = (
                "Could not extract article title. "
                "IEEE requires quoted titles for journal and conference articles: "
                '"A. B. Smith, "Title of Article," Journal, vol. 1, 2021."'
            )
            return ref
        elif after_authors:
            book_title_m = re.match(r'\s*\*?([^.]+?)\*?\s*\.', after_authors)
            if book_title_m:
                ref.title = book_title_m.group(1).strip().strip("*").strip()

    # ── Extract year ──────────────────────────────────────────────────────
    # IEEE typically places the year near the end: ", 2021." or "(2021)"
    year_candidates = re.findall(r'\b((?:19|20)\d{2})\b', raw)
    if year_candidates:
        # Use the last year found (IEEE convention: year at end)
        ref.year = year_candidates[-1]

    # ── Extract rest (source/journal info) ────────────────────────────────
    if title_m:
        rest_start = title_m.end()
        ref.rest = raw[rest_start:].strip().lstrip(",").strip()

    # ══════════════════════════════════════════════════════════════════════
    #  VALIDATION
    # ══════════════════════════════════════════════════════════════════════

    # ── Title required ────────────────────────────────────────────────────
    if not ref.title:
        ref.parse_error = (
            "Could not extract the article title. "
            "IEEE uses quoted titles for articles: "
            '"A. B. Smith, "Title of Article," Journal, vol. 1, 2021."'
        )
        return ref

    # ── Year required ─────────────────────────────────────────────────────
    if not ref.year:
        ref.parse_error = (
            "Could not extract publication year. "
            "IEEE requires a 4-digit year, typically near the end of the reference."
        )
        return ref

    # ── Authors required ──────────────────────────────────────────────────
    if not ref.authors:
        ref.parse_error = (
            "Could not extract author name(s). "
            "IEEE uses initials-first format: "
            '"A. B. Smith, C. D. Jones, and E. F. Brown"'
        )
        return ref

    # ── Ending character ──────────────────────────────────────────────────
    if not re.search(r'[.)\w/]$', raw.rstrip()):
        ref.parse_error = (
            "IEEE reference must end with a period or URL."
        )
        return ref

    # ── vol./no./pp. validation (if present) ──────────────────────────────
    _ieee_vol_m = re.search(r'\bvol\.\s*(\S+)', raw, re.IGNORECASE)
    if _ieee_vol_m:
        _ieee_vol = _ieee_vol_m.group(1).rstrip('.,;:()')
        if not re.match(r'^\d{1,4}$', _ieee_vol):
            ref.parse_error = (
                f"Invalid volume number 'vol. {_ieee_vol}'. "
                "IEEE requires a numeric volume (e.g., vol. 5)."
            )
            return ref

    _ieee_no_m = re.search(r'\bno\.\s*(\S+)', raw, re.IGNORECASE)
    if _ieee_no_m:
        _ieee_no = _ieee_no_m.group(1).rstrip('.,;:()')
        if not re.match(r'^\d{1,4}$', _ieee_no):
            ref.parse_error = (
                f"Invalid issue number 'no. {_ieee_no}'. "
                "IEEE requires a numeric issue (e.g., no. 3)."
            )
            return ref

    _ieee_pp_m = re.search(r'\bpp\.\s*(\S+)', raw, re.IGNORECASE)
    if _ieee_pp_m:
        _ieee_pp = _ieee_pp_m.group(1).rstrip('.,;')
        if not re.match(r'^\d+(?:[\u2013\-]\d+)?$', _ieee_pp):
            ref.parse_error = (
                f"Invalid page range 'pp. {_ieee_pp}'. "
                "Pages must be digits with an optional range (e.g., pp. 45-50)."
            )
            return ref

    # ── Minimum period count ──────────────────────────────────────────────
    if raw.count(".") < 2:
        ref.parse_error = (
            "IEEE reference appears incomplete — too few periods found."
        )
        return ref

    ref.valid_apa = True   # reusing flag to mean "valid format"
    return ref


# ══════════════════════════════════════════════════════════════════════════════
#  DOI-ONLY RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _crossref_to_apa7(cr: dict, doi: str) -> str:
    """Generate an APA 7 citation string from CrossRef data."""
    from textwrap import wrap

    authors = []
    for a in cr.get("author", []):
        family = a.get("family", "")
        given  = a.get("given", "")
        if family:
            initials = " ".join(f"{p[0]}." for p in given.split() if p) if given else ""
            authors.append(f"{family}, {initials}".strip(", "))

    if len(authors) > 20:
        author_str = ", ".join(authors[:19]) + ", … " + authors[-1]
    elif len(authors) > 1:
        author_str = ", ".join(authors[:-1]) + ", & " + authors[-1]
    elif authors:
        author_str = authors[0]
    else:
        author_str = "Unknown Author"

    year  = ""
    for key in ("published", "published-print", "published-online", "issued"):
        dp = cr.get(key, {}).get("date-parts", [[]])
        if dp and dp[0]:
            year = str(dp[0][0])
            break
    year = year or "n.d."

    titles = cr.get("title", [])
    title  = titles[0] if titles else "Untitled"

    containers = cr.get("container-title", [])
    journal    = containers[0] if containers else ""

    volume  = cr.get("volume", "")
    issue   = cr.get("issue", "")
    pages   = cr.get("page", "")

    parts = [f"{author_str} ({year}). {title}."]
    if journal:
        journal_part = f" *{journal}*"
        if volume:
            journal_part += f", *{volume}*"
            if issue:
                journal_part += f"({issue})"
        if pages:
            journal_part += f", {pages}"
        parts.append(journal_part + ".")
    parts.append(f" https://doi.org/{doi}")
    return "".join(parts)


def _crossref_to_mla(cr: dict, doi: str) -> str:
    """Generate an MLA 9 citation string from CrossRef data."""
    authors = []
    for a in cr.get("author", []):
        family = a.get("family", "")
        given  = a.get("given", "")
        if family:
            authors.append(f"{family}, {given}".strip(", "))

    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        author_str = authors[0]
    else:
        author_str = authors[0] + ", and " + ", and ".join(authors[1:])

    year = ""
    for key in ("published", "published-print", "published-online", "issued"):
        dp = cr.get(key, {}).get("date-parts", [[]])
        if dp and dp[0]:
            year = str(dp[0][0])
            break

    titles   = cr.get("title", [])
    title    = titles[0] if titles else "Untitled"
    journal  = (cr.get("container-title") or [""])[0]
    volume   = cr.get("volume", "")
    issue    = cr.get("issue", "")
    pages    = cr.get("page", "")

    s = f'{author_str}. "{title}."'
    if journal:
        s += f" *{journal}*"
        if volume:
            s += f", vol. {volume}"
        if issue:
            s += f", no. {issue}"
    if year:
        s += f", {year}"
    if pages:
        s += f", pp. {pages}"
    s += f". https://doi.org/{doi}."
    return s


def _crossref_to_chicago(cr: dict, doi: str) -> str:
    """Generate a Chicago 17 Author-Date citation string from CrossRef data."""
    authors = []
    for a in cr.get("author", []):
        family = a.get("family", "")
        given  = a.get("given", "")
        if family:
            authors.append(f"{family}, {given}".strip(", "))

    author_str = authors[0] if authors else "Unknown Author"
    if len(authors) > 1:
        author_str = ", ".join(authors[:-1]) + ", and " + authors[-1]

    year = ""
    for key in ("published", "published-print", "published-online", "issued"):
        dp = cr.get(key, {}).get("date-parts", [[]])
        if dp and dp[0]:
            year = str(dp[0][0])
            break

    titles  = cr.get("title", [])
    title   = titles[0] if titles else "Untitled"
    journal = (cr.get("container-title") or [""])[0]
    volume  = cr.get("volume", "")
    issue   = cr.get("issue", "")
    pages   = cr.get("page", "")

    s = f'{author_str}. {year}. "{title}."'
    if journal:
        s += f" *{journal}*"
        if volume:
            s += f" {volume}"
        if issue:
            s += f" ({issue})"
    if pages:
        s += f": {pages}"
    s += f". https://doi.org/{doi}."
    return s


def _crossref_to_ieee(cr: dict, doi: str) -> str:
    """New Rec B: Generate an IEEE citation string from CrossRef data."""
    authors = []
    for a in cr.get("author", []):
        family = a.get("family", "")
        given  = a.get("given", "")
        if family:
            initials = " ".join(f"{p[0]}." for p in given.split() if p) if given else ""
            authors.append(f"{initials} {family}".strip())

    if len(authors) > 6:
        author_str = ", ".join(authors[:6]) + ", et al."
    elif len(authors) > 1:
        author_str = ", ".join(authors[:-1]) + ", and " + authors[-1]
    elif authors:
        author_str = authors[0]
    else:
        author_str = "Unknown Author"

    year = ""
    for key in ("published", "published-print", "published-online", "issued"):
        dp = cr.get(key, {}).get("date-parts", [[]])
        if dp and dp[0]:
            year = str(dp[0][0])
            break

    titles  = cr.get("title", [])
    title   = titles[0] if titles else "Untitled"
    journal = (cr.get("container-title") or [""])[0]
    volume  = cr.get("volume", "")
    issue   = cr.get("issue", "")
    pages   = cr.get("page", "")

    s = f'{author_str}, "{title},"'
    if journal:
        s += f" *{journal}*"
    if volume:
        s += f", vol. {volume}"
    if issue:
        s += f", no. {issue}"
    if pages:
        s += f", pp. {pages}"
    if year:
        s += f", {year}"
    s += f". https://doi.org/{doi}."
    return s


FORMAT_BUILDERS = {
    FORMAT_APA7:    _crossref_to_apa7,
    FORMAT_MLA:     _crossref_to_mla,
    FORMAT_CHICAGO: _crossref_to_chicago,
    FORMAT_IEEE:    _crossref_to_ieee,    # New Rec B
}


def resolve_doi_only(raw: str, fmt: str = FORMAT_APA7) -> ParsedReference:
    """
    Given only a DOI (or DOI URL), fetch full metadata from CrossRef and
    return a ParsedReference with all fields filled in.
    """
    m = DOI_ONLY_RE.match(raw.strip())
    if not m:
        ref = ParsedReference(raw=raw.strip(), fmt=fmt)
        ref.parse_error = "Input does not look like a valid DOI."
        return ref

    doi = m.group(1).rstrip(".,;)")
    ref = ParsedReference(raw=raw.strip(), fmt=fmt, doi_only=True)
    ref.doi = doi
    ref.url = f"https://doi.org/{doi}"

    cr = crossref_lookup(doi)
    if not cr:
        ref.parse_error = (
            f"Could not resolve DOI '{doi}' via CrossRef. "
            "Please check the DOI is correct."
        )
        return ref

    # Fill in metadata from CrossRef
    ref.authors = crossref_authors(cr)
    ref.year    = crossref_year(cr)
    ref.title   = crossref_title(cr)
    ref.valid_apa = bool(ref.title and ref.year)

    if not ref.valid_apa:
        ref.parse_error = "CrossRef returned incomplete metadata for this DOI."
        return ref

    # Build formatted citation in the requested format
    builder = FORMAT_BUILDERS.get(fmt, _crossref_to_apa7)
    ref.formatted_citation = builder(cr, doi)

    return ref


# ── Unified entry point for parsing ──────────────────────────────────────────
def parse_reference(raw: str, fmt: str = FORMAT_APA7) -> ParsedReference:
    """Detect input type and dispatch to the correct parser."""
    stripped = raw.strip()

    # ── DOI-only input ────────────────────────────────────────────────────────
    if DOI_ONLY_RE.match(stripped):
        return resolve_doi_only(stripped, fmt)

    # ── Format-specific parsers ───────────────────────────────────────────────
    if fmt == FORMAT_MLA:
        return parse_mla_reference(stripped)
    elif fmt == FORMAT_CHICAGO:
        return parse_chicago_reference(stripped)
    elif fmt == FORMAT_IEEE:
        return parse_ieee_reference(stripped)
    else:
        return parse_apa7_reference(stripped)


# ── CrossRef Lookup ───────────────────────────────────────────────────────────
def crossref_lookup(doi: str) -> dict:
    _tls.crossref_error = ""
    url = CROSSREF_API.format(doi=urllib.parse.quote(doi, safe="/"))
    r = safe_get(url)
    if r is None:
        _tls.crossref_error = f"CrossRef unreachable: {getattr(_tls, 'request_error', '')}"
        return {}
    if r.status_code == 200:
        try:
            return r.json().get("message", {})
        except Exception as e:
            _tls.crossref_error = f"CrossRef returned invalid JSON: {str(e)[:80]}"
            return {}
    elif r.status_code == 404:
        _tls.crossref_error = f"DOI not found in CrossRef (HTTP 404)"
    elif r.status_code == 429:
        _tls.crossref_error = f"CrossRef rate limit exceeded (HTTP 429) — try again later"
    else:
        _tls.crossref_error = f"CrossRef returned HTTP {r.status_code}"
    return {}


def crossref_authors(data: dict) -> list[str]:
    authors = []
    for a in data.get("author", []):
        family = a.get("family", "")
        given  = a.get("given", "")
        # ── Bug fix: CrossRef sometimes stores a stray middle initial glued
        # onto the front of the family name with no space, e.g.
        # {"given": "Aidan", "family": "N.Gomez"} for "Aidan N. Gomez".
        # Verified directly against CrossRef's live API (search "Attention
        # is all you need") — the "Gomez" author entry is stored exactly
        # this way, which previously made a correctly cited "Gomez, A. N."
        # get reported as "not found" (compared against "N.Gomez, A."
        # instead of "Gomez, A. N."). A single capital letter + period
        # immediately followed by another capitalized word is not a
        # plausible real family name, so it's treated as a misplaced
        # initial and moved back onto the given-name side before initials
        # are built.
        _stray_initial_m = re.match(r'^([A-Z])\.([A-Z][a-z].*)$', family)
        if _stray_initial_m:
            given  = f"{given} {_stray_initial_m.group(1)}.".strip()
            family = _stray_initial_m.group(2)
        if family:
            initials = " ".join(f"{p[0]}." for p in given.split() if p) if given else ""
            authors.append(f"{family}, {initials}".strip(", "))
    return authors


def crossref_year(data: dict) -> str:
    for key in ("published", "published-print", "published-online", "issued"):
        dp = data.get(key, {}).get("date-parts", [[]])
        if dp and dp[0]:
            return str(dp[0][0])
    return ""


def crossref_title(data: dict) -> str:
    titles = data.get("title", [])
    return titles[0] if titles else ""


def crossref_doi(data: dict) -> str:
    return data.get("DOI", "")


# ── Rec 3: Extended CrossRef metadata extraction ─────────────────────────────
def crossref_publisher(data: dict) -> str:
    return data.get("publisher", "")


def crossref_journal(data: dict) -> str:
    titles = data.get("container-title", [])
    return titles[0] if titles else ""


def crossref_type(data: dict) -> str:
    """Document type: journal-article, book-chapter, proceedings-article, etc."""
    return data.get("type", "")


def crossref_issn(data: dict) -> list[str]:
    return data.get("ISSN", [])


def crossref_isbn(data: dict) -> list[str]:
    return data.get("ISBN", [])


def crossref_volume(data: dict) -> str:
    return data.get("volume", "")


def crossref_issue(data: dict) -> str:
    return data.get("issue", "")


def crossref_page(data: dict) -> str:
    return data.get("page", "")


def crossref_conference(data: dict) -> str:
    """Conference name from proceedings. CrossRef stores it in 'event'."""
    event = data.get("event", {})
    if isinstance(event, dict):
        return event.get("name", "")
    return ""


def crossref_subject(data: dict) -> list[str]:
    return data.get("subject", [])


# ── CrossRef multi-metadata search (Recommendation 1) ────────────────────────
def crossref_search_by_metadata(title: str,
                                authors: list[str] | None = None,
                                year: str = "",
                                max_results: int = 15,
                                min_score: float = 0.45) -> dict:
    """
    Search CrossRef by bibliographic metadata (title, author, year) when
    no DOI is available.  Uses the /works?query.bibliographic= endpoint.

    Returns the best-matching work's full metadata dict, or {} if nothing
    meets the acceptance threshold.

    Rec 5B: max_results parameter controls how many candidates are fetched
    from CrossRef (default: 15, increased from the previous hardcoded 5).
    All scored candidates are stored in the returned dict under the key
    '_search_candidates' so downstream code can expose alternative matches
    to the user.
    """
    if not title or len(title.strip()) < 5:
        return {}

    # ── Build query parameters ────────────────────────────────────────────
    params: dict = {
        "query.bibliographic": title,
        "rows": max_results,     # Rec 5B: was hardcoded 5, now configurable
        "select": (
            "DOI,title,author,"
            "published,published-print,published-online,issued,"
            "container-title,volume,issue,page,publisher,type,ISSN,ISBN,subject,event"
        ),
    }

    # Narrow the search with the first author's surname when available.
    if authors:
        first_surname = authors[0].split(",")[0].strip()
        if first_surname and len(first_surname) >= 2:
            params["query.author"] = first_surname

    # ── Execute the query ─────────────────────────────────────────────────
    try:
        r = requests.get(
            CROSSREF_SEARCH_API, params=params,
            headers=HEADERS, timeout=TIMEOUT,
        )
        if r.status_code != 200:
            _tls.crossref_error = f"CrossRef title search returned HTTP {r.status_code}"
            return {}
        items = r.json().get("message", {}).get("items", [])
    except requests.exceptions.Timeout:
        _tls.crossref_error = f"CrossRef title search timed out after {TIMEOUT}s"
        return {}
    except requests.exceptions.ConnectionError:
        _tls.crossref_error = "CrossRef title search: connection refused or DNS failure"
        return {}
    except Exception as e:
        _tls.crossref_error = f"CrossRef title search failed: {type(e).__name__}"
        return {}

    if not items:
        return {}

    # ── New Rec D: Multi-field candidate scoring ─────────────────────────
    # Extract reference body text for journal comparison
    _ref_body_lower = normalize(title)
    # Collect cited author surnames for comparison
    _cited_surnames = set()
    if authors:
        for a in authors:
            surname = a.split(",")[0].strip().lower()
            if surname and len(surname) >= 2:
                _cited_surnames.add(surname)

    best_item  = None
    best_score = 0.0
    all_candidates = []   # Rec 5B: collect ALL scored candidates

    for item in items:
        item_title = (item.get("title") or [""])[0]
        if not item_title:
            continue

        # ── Title score (primary signal, 60% of base) ─────────────────
        c_sim = char_similarity(title, item_title)
        w_sim = word_jaccard(title, item_title)
        score = c_sim * 0.35 + w_sim * 0.25

        # ── Year bonus (+0.10) ────────────────────────────────────────
        item_year = ""
        if year:
            for key in ("published", "published-print",
                        "published-online", "issued"):
                dp = item.get(key, {}).get("date-parts", [[]])
                if dp and dp[0]:
                    item_year = str(dp[0][0])
                    break
            if item_year == year:
                score += 0.10

        # ── Author surname overlap (+0.15) ────────────────────────────
        if _cited_surnames:
            item_surnames = set()
            for a in item.get("author", []):
                fam = a.get("family", "").strip().lower()
                if fam and len(fam) >= 2:
                    item_surnames.add(fam)
            if item_surnames:
                overlap = len(_cited_surnames & item_surnames)
                max_possible = max(len(_cited_surnames), 1)
                score += 0.15 * (overlap / max_possible)

        # ── Journal/source word overlap (+0.15) ───────────────────────
        item_journal = (item.get("container-title") or [""])[0]
        if item_journal:
            j_sim = word_jaccard(title + " " + (authors[0] if authors else ""),
                                 item_journal)
            score += 0.15 * min(j_sim * 2, 1.0)

        # ── Rec 5B: Record this candidate ─────────────────────────────
        all_candidates.append({
            "title":   item_title,
            "authors": [f"{a.get('given','')} {a.get('family','')}"
                        for a in item.get("author", [])[:5]],
            "year":    item_year,
            "doi":     item.get("DOI", ""),
            "journal": item_journal,
            "score":   round(score, 4),
        })

        if score > best_score:
            best_score = score
            best_item  = item

    # Sort candidates by score descending
    all_candidates.sort(key=lambda c: c["score"], reverse=True)

    # ── Acceptance threshold ──────────────────────────────────────────────
    # New Rec E: min_score is configurable so recommend_alternative_sources()
    # can request the full ranked candidate pool (min_score=0.0) even when no
    # single candidate is confident enough to be treated as "the same paper"
    # for verification purposes (the default 0.45 gate).
    if best_item and best_score >= min_score:
        # Rec 5B: Attach the candidate list so validate_reference can
        # expose it in the similarity breakdown
        best_item["_search_candidates"] = all_candidates
        best_item["_best_score"]        = best_score
        return best_item

    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  INTELLIGENT SOURCE RECOMMENDATION (New Rec E)
# ══════════════════════════════════════════════════════════════════════════════
# Implements the "Intelligent Recommendations" + "Neural Re-Ranking" stages of
# the Conceptual Framework (Figure 3.1). Retrieves open-access candidates from
# CrossRef and arXiv, then re-ranks them with the Cross-Encoder against the
# cited title. Only invoked when a reference fails verification — this is a
# suggestion list surfaced to the user, never an automatic replacement.

def arxiv_search(title: str, max_results: int = 10) -> list[dict]:
    """
    Search arXiv's free public Atom API for candidate papers matching a title.
    Returns candidates in the same shape as the CrossRef candidate list so
    both sources can be pooled and re-ranked together.

    Bug fix: uses ARXIV_TIMEOUT (short) instead of the general TIMEOUT —
    arXiv's public API was directly observed to rate-limit and hang far
    more than CrossRef during testing, and this is only a "nice to have"
    recommendation lookup, not a core verification step. Letting one slow
    arXiv call eat the full 15s TIMEOUT per reference was a major
    contributor to multi-minute PDF batch hangs.
    """
    if not title or len(title.strip()) < 5:
        return []
    query = urllib.parse.quote(f'ti:"{title}"')
    url = f"{ARXIV_API}?search_query={query}&start=0&max_results={max_results}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=ARXIV_TIMEOUT)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    candidates = []
    for entry in root.findall("atom:entry", ns):
        entry_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        if not entry_title:
            continue
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        published   = entry.findtext("atom:published", default="", namespaces=ns) or ""
        arxiv_id_url = entry.findtext("atom:id", default="", namespaces=ns) or ""
        candidates.append({
            "title":   re.sub(r"\s+", " ", entry_title),
            "authors": [a for a in authors if a],
            "year":    published[:4] if published else "",
            "doi":     "",
            "url":     arxiv_id_url,
            "summary": re.sub(r"\s+", " ", summary)[:1000],
            "source":  "arXiv",
        })
    return candidates


def fetch_semantic_scholar_abstract(doi: str = "", title: str = "") -> str:
    """
    Best-effort abstract retrieval used for RTE classification. CrossRef
    rarely exposes abstracts, so Semantic Scholar's free Graph API is used
    as a secondary source. Returns "" on any failure (no result, rate limit,
    network error) — callers must treat that as "abstract unavailable",
    not as an error.
    """
    try:
        if doi:
            r = requests.get(
                SEMANTIC_SCHOLAR_API.format(doi=doi),
                params={"fields": "abstract,title"},
                headers=HEADERS, timeout=SEMANTIC_SCHOLAR_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("abstract"):
                    return data["abstract"]
        if title:
            r = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_API,
                params={"query": title, "fields": "abstract,title", "limit": 1},
                headers=HEADERS, timeout=SEMANTIC_SCHOLAR_TIMEOUT,
            )
            if r.status_code == 200:
                papers = r.json().get("data", [])
                if papers and papers[0].get("abstract"):
                    return papers[0]["abstract"]
    except Exception:
        pass
    return ""


def classify_citation_entailment(evidence_sentence: str, ref: "ParsedReference",
                                 citation_text: str = "") -> dict:
    """
    RTE classification for one Chapter 2 in-text citation: does the cited
    source's abstract (premise) support the sentence in which the student
    cites it (hypothesis)?  Used by match_citation_to_reference().

    The citation marker itself (e.g. "Vaswani et al. (2017)") is stripped
    from the hypothesis before scoring. Generic NLI models are trained to
    judge whether a premise's own content entails a hypothesis's content —
    they have no basis to confirm *authorship attribution* named in the
    hypothesis (the premise text alone never says who wrote it), so a raw
    "Author (Year) claimed X" hypothesis is judged "neutral" almost every
    time regardless of whether X is actually supported. Testing confirmed
    this: the same claim scores ~99% entailment with the marker removed and
    ~1% with it left in. Since extract_in_text_citations() always captures
    the marker inside the evidence sentence, skipping this step would make
    the RTE feature report "no_evidence" for essentially every citation.
    """
    if not evidence_sentence or ref is None:
        return {"label": "unavailable", "confidence": 0.0,
                "reason": "Missing evidence sentence or reference."}
    hypothesis = evidence_sentence
    if citation_text:
        hypothesis = hypothesis.replace(citation_text, "").strip()
        hypothesis = re.sub(r"^[,;:.\s]+", "", hypothesis)
        hypothesis = re.sub(r"\s{2,}", " ", hypothesis)
    if not hypothesis or len(hypothesis) < 5:
        return {"label": "unavailable", "confidence": 0.0,
                "reason": "No claim text remained after removing the citation marker."}
    abstract = fetch_semantic_scholar_abstract(doi=ref.doi, title=ref.title)
    if not abstract:
        return {"label": "no_evidence", "confidence": 0.0,
                "reason": "Source abstract unavailable for entailment check."}
    return neural_entailment(premise=abstract, hypothesis=hypothesis)


def recommend_alternative_sources(ref: "ParsedReference", max_candidates: int = 10) -> list[dict]:
    """
    Retrieve candidate open-access papers from CrossRef and arXiv, then
    re-rank them by Cross-Encoder semantic similarity against the cited
    title (the "Neural Re-Ranking" step). Falls back to the existing
    char_similarity() string measure if the neural stack is unavailable,
    so the feature still returns a ranked list either way.
    """
    if not ref.title or len(ref.title.strip()) < 5:
        return []

    # Bug fix: these two independent external lookups used to run one after
    # the other (up to TIMEOUT seconds each) — worst case ~2x TIMEOUT for a
    # single reference's recommendation step alone. Running them
    # concurrently caps this step at whichever one is slower, not both
    # combined. Contributed directly to multi-minute hangs on real PDFs
    # with many unverifiable references, each paying this cost.
    pool: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool_executor:
        cr_future  = pool_executor.submit(crossref_search_by_metadata, ref.title, ref.authors,
                                          ref.year, max_results=max_candidates, min_score=0.0)
        arx_future = pool_executor.submit(arxiv_search, ref.title, max_results=max_candidates)

        try:
            cr = cr_future.result()
        except Exception:
            cr = {}
        if cr:
            for c in cr.get("_search_candidates", [])[:max_candidates]:
                pool.append({
                    "title":   c.get("title", ""),
                    "authors": c.get("authors", []),
                    "year":    c.get("year", ""),
                    "doi":     c.get("doi", ""),
                    "url":     f"https://doi.org/{c['doi']}" if c.get("doi") else "",
                    "source":  "CrossRef",
                })

        try:
            pool.extend(arx_future.result())
        except Exception:
            pass

    if not pool:
        return []

    neural_ready = _get_similarity_cross_encoder() is not None
    for cand in pool:
        if neural_ready:
            cand["neural_score"] = neural_similarity(ref.title, cand["title"])
        if cand.get("neural_score") is None:
            # Fallback keeps the feature functional without the neural stack.
            cand["neural_score"] = round(char_similarity(ref.title, cand["title"]), 4)
        cand["scoring_method"] = "cross_encoder" if neural_ready else "char_similarity_fallback"

    pool.sort(key=lambda c: c["neural_score"], reverse=True)

    # De-duplicate near-identical titles across sources, keeping the top-scored copy.
    seen, deduped = set(), []
    for c in pool:
        key = normalize(c["title"])[:80]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped[:5]


# ── Web page scraping ─────────────────────────────────────────────────────────
def scrape_metadata(url: str) -> dict:
    r = safe_get(url)
    if not r or r.status_code != 200:
        return {}
    soup = BeautifulSoup(r.text, "lxml")
    meta: dict = {}

    def og(prop: str) -> str:
        tag = soup.find("meta", property=f"og:{prop}") or \
              soup.find("meta", attrs={"name": prop})
        return tag["content"].strip() if tag and tag.get("content") else ""

    def mname(name: str) -> str:
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.get("content") else ""

    meta["title"] = (
        mname("citation_title") or mname("dc.title") or og("title") or
        (soup.title.string.strip() if soup.title else "")
    )
    author_tags = soup.find_all(
        "meta", attrs={"name": re.compile(r"citation_author|dc\.creator", re.I)}
    )
    meta["authors"] = [t["content"].strip() for t in author_tags if t.get("content")]
    date_str = (
        mname("citation_publication_date") or mname("citation_date") or
        mname("dc.date") or mname("article:published_time") or
        og("article:published_time")
    )
    year_m = YEAR_PATTERN.search(date_str)
    meta["year"] = year_m.group(0) if year_m else ""
    meta["doi"]  = (mname("citation_doi") or mname("dc.identifier") or "").replace("doi:", "").strip()

    # ── Rec 3: Additional provenance metadata from web pages ─────────────
    meta["journal"]   = mname("citation_journal_title") or mname("dc.source") or ""
    meta["publisher"] = mname("citation_publisher") or mname("dc.publisher") or ""
    meta["volume"]    = mname("citation_volume") or ""
    meta["issue"]     = mname("citation_issue") or ""
    meta["issn"]      = mname("citation_issn") or ""

    return meta


# ── Author suffix-attack guard ───────────────────────────────────────────────
def _author_first_letter_ok(cited: str, found: str) -> bool:
    """Cited name must start with the same letter as found name (blocks suffix attacks)."""
    c = cited.strip()
    f = found.strip()
    return bool(c) and bool(f) and c[0].lower() == f[0].lower()


# ── Bug fix: surname/initials-separated author comparison ────────────────────
# AUTHOR_THRESHOLD alone (applied to the whole "Surname, I." string) could not
# distinguish two very different situations that happen to produce OVERLAPPING
# similarity scores: a legitimate initials-completeness difference (e.g. cited
# "Smith, J. A." vs found "Smith, J." — same person, CrossRef just lists fewer
# initials) scores ~0.857, while an actual misspelled surname (e.g. "Vaswani"
# vs "Xaswani" or "Vasvani") scores ~0.857-0.933 — the SAME range. No single
# threshold on the combined string can separate "fewer initials known" from
# "surname is wrong" when they overlap like that; one was always going to leak
# through. Splitting the comparison fixes this: once initials aren't part of
# the compared string, matching surnames always score exactly 1.0 regardless
# of initials, while surname typos remain in the 0.857-0.933 band — a clean,
# separable gap that a strict SURNAME_THRESHOLD can sit above.
SURNAME_THRESHOLD = 0.94


def _split_surname_initials(name: str) -> tuple[str, list[str]]:
    """Splits a 'Surname, I. I.' formatted name into (surname, [initial, ...])."""
    parts = name.split(",", 1)
    surname = parts[0].strip()
    initials = re.findall(r"[A-Za-z]", parts[1]) if len(parts) > 1 else []
    return surname, initials


def author_name_similarity(cited: str, found: str) -> tuple[float, bool]:
    """
    Compares two author names by surname and initials separately rather than
    as one fuzzy string. Returns (surname_similarity, is_match).

    - Surname must clear SURNAME_THRESHOLD — a real misspelling, not just a
      difference in how many initials are recorded.
    - Initials are compared as a prefix relationship: one side is allowed to
      simply have fewer known initials than the other (a common, legitimate
      metadata-completeness difference), but whichever initials ARE present
      on both sides must agree — so "J." vs "J. A." passes, but "J." vs "B."
      does not.
    """
    surname_c, initials_c = _split_surname_initials(cited)
    surname_f, initials_f = _split_surname_initials(found)

    surname_sim = char_similarity(surname_c, surname_f)
    surname_ok  = surname_sim >= SURNAME_THRESHOLD

    if not initials_c or not initials_f:
        initials_ok = True
    else:
        shorter, longer = (
            (initials_c, initials_f) if len(initials_c) <= len(initials_f)
            else (initials_f, initials_c)
        )
        initials_ok = [i.upper() for i in longer[:len(shorter)]] == [i.upper() for i in shorter]

    return surname_sim, (surname_ok and initials_ok)


# ── Core validator ────────────────────────────────────────────────────────────
def validate_reference(ref: ParsedReference) -> ValidationResult:
    result = ValidationResult(reference=ref)

    if not ref.valid_apa:
        result.errors.append(
            f"Format error ({ref.fmt.upper()}): {ref.parse_error}"
        )
        return result

    # ── 1. Link reachability ──────────────────────────────────────────────────
    # (Recommendation 1) Previously the validator returned immediately when
    # no URL/DOI was present.  Now it continues so that the title-based
    # CrossRef search (step 2b) can still verify the reference.
    if not ref.url and not ref.doi:
        result.warnings.append(
            "No URL or DOI found — will attempt metadata search by title."
        )

    target_url = ref.url
    r = safe_get(target_url) if target_url else None

    if not target_url:
        # No URL to check — skip the link reachability block entirely.
        # link_reachable stays None (= "not checked"), which is correct.
        pass
    elif r is None:
        result.link_reachable = False
        # New Rec A: use the specific error captured by safe_get
        detail = getattr(_tls, 'request_error', '') or "Connection error / timeout"
        result.link_status = detail
        result.errors.append(f"Link unreachable ({detail}): {target_url}")
    elif r.status_code == 200:
        result.link_reachable = True
        result.link_status    = f"OK (HTTP {r.status_code})"
    elif r.status_code in (301, 302):
        result.link_reachable = True
        result.link_status    = f"Redirected (HTTP {r.status_code})"
        result.warnings.append("URL redirects — consider updating to the canonical URL.")
    elif r.status_code == 404:
        result.link_reachable = False
        result.link_status    = "404 Not Found"
        result.errors.append(f"Broken link (404): {target_url}")
    elif r.status_code in (403, 401):
        result.link_reachable = True
        result.link_status    = f"Access restricted (HTTP {r.status_code})"
        result.warnings.append(
            f"Server returned {r.status_code} — metadata cannot be fully verified."
        )
    else:
        result.link_reachable = False
        result.link_status    = f"HTTP {r.status_code}"
        result.errors.append(f"Link returned HTTP {r.status_code}: {target_url}")

    # ── 2. Metadata retrieval ─────────────────────────────────────────────────
    meta: dict = {}
    source = ""

    # ── 2a. DOI lookup (original behaviour, unchanged) ────────────────────
    if ref.doi:
        cr = crossref_lookup(ref.doi)
        if cr:
            source = "CrossRef"
            meta = {
                "title":      crossref_title(cr),
                "authors":    crossref_authors(cr),
                "year":       crossref_year(cr),
                "doi":        crossref_doi(cr),
                # ── Rec 3: extended metadata for provenance verification ──
                "publisher":  crossref_publisher(cr),
                "journal":    crossref_journal(cr),
                "type":       crossref_type(cr),
                "issn":       crossref_issn(cr),
                "isbn":       crossref_isbn(cr),
                "volume":     crossref_volume(cr),
                "issue":      crossref_issue(cr),
                "page":       crossref_page(cr),
                "conference": crossref_conference(cr),
                "subject":    crossref_subject(cr),
            }

    # ── 2b. Title-based CrossRef search (Recommendation 1) ────────────────
    #   When the DOI lookup produces no metadata (because the reference
    #   has no DOI, or the DOI was invalid), search CrossRef using the
    #   parsed title, author(s), and year.  This covers book citations,
    #   conference proceedings, and older papers that lack a DOI.
    _search_candidates = []   # Rec 5B: all scored candidates
    _best_search_score = 0.0
    if not meta and ref.title:
        cr = crossref_search_by_metadata(ref.title, ref.authors, ref.year)
        if cr:
            source = "CrossRef (title search)"
            # Rec 5B: extract the candidate list before removing private keys
            _search_candidates = cr.pop("_search_candidates", [])
            _best_search_score = cr.pop("_best_score", 0.0)
            meta = {
                "title":      crossref_title(cr),
                "authors":    crossref_authors(cr),
                "year":       crossref_year(cr),
                "doi":        crossref_doi(cr),
                # ── Rec 3: extended metadata for provenance verification ──
                "publisher":  crossref_publisher(cr),
                "journal":    crossref_journal(cr),
                "type":       crossref_type(cr),
                "issn":       crossref_issn(cr),
                "isbn":       crossref_isbn(cr),
                "volume":     crossref_volume(cr),
                "issue":      crossref_issue(cr),
                "page":       crossref_page(cr),
                "conference": crossref_conference(cr),
                "subject":    crossref_subject(cr),
            }

    # ── 2c. Web page scraping (original fallback, unchanged) ──────────────
    if not meta and result.link_reachable:
        source = "Web page metadata"
        meta = scrape_metadata(target_url)

    if not meta:
        # New Rec A: provide specific diagnostic instead of a generic message
        _diag_parts = []
        if ref.doi and getattr(_tls, 'crossref_error', ''):
            _diag_parts.append(f"DOI lookup: {getattr(_tls, 'crossref_error', '')}")
        if ref.title and not ref.doi and getattr(_tls, 'crossref_error', ''):
            _diag_parts.append(f"Title search: {getattr(_tls, 'crossref_error', '')}")
        if target_url and getattr(_tls, 'request_error', ''):
            _diag_parts.append(f"Web scraping: {getattr(_tls, 'request_error', '')}")

        if _diag_parts:
            result.warnings.append(
                "Metadata retrieval failed. Diagnostics: " +
                " | ".join(_diag_parts) +
                " — manual verification recommended."
            )
        else:
            result.warnings.append(
                "Could not retrieve metadata via DOI, title search, or web scraping. "
                "Manual verification recommended."
            )

    result.metadata_source = source

    # ── 3. Metadata field checks ──────────────────────────────────────────────
    if meta:
        # ── Title (stricter dual-threshold check) ─────────────────────────────
        found_title = meta.get("title", "")
        if found_title and ref.title:
            score, is_match, title_breakdown = title_similarity(ref.title, found_title)
            result.title_score = score
            result.title_found = found_title
            result.title_match = is_match
            # ── Rec 4: store the title breakdown ──────────────────────────
            result.similarity_breakdown["title"] = title_breakdown
            if not is_match:
                char_sim  = char_similarity(ref.title, found_title)
                word_sim  = word_jaccard(ref.title, found_title)
                # Identify whether a numeric-token mismatch was the deciding factor
                _nums_c = sorted(re.findall(r'\b\d+\b', normalize(ref.title)))
                _nums_f = sorted(re.findall(r'\b\d+\b', normalize(found_title)))
                _num_note = ""
                if (_nums_c or _nums_f) and _nums_c != _nums_f:
                    _num_note = (
                        f"\n    ⚠ Numeric mismatch — "
                        f"cited numbers {_nums_c} ≠ found numbers {_nums_f}."
                    )
                result.errors.append(
                    f"Title mismatch "
                    f"(char similarity {char_sim:.0%}, word overlap {word_sim:.0%}).\n"
                    f"    Cited : {ref.title}\n"
                    f"    Found : {found_title}\n"
                    f"    Both char similarity ≥{TITLE_CHAR_THRESHOLD:.0%} "
                    f"AND word overlap ≥{TITLE_WORD_THRESHOLD:.0%} required."
                    + _num_note
                )
        elif not found_title:
            result.warnings.append(f"Title not found in {source}.")

        # ── Authors ───────────────────────────────────────────────────────────
        found_authors = meta.get("authors", [])
        if found_authors and ref.authors:
            result.authors_found = found_authors
            matched = []
            author_details = []   # Rec 4: per-author similarity details
            for cited in ref.authors:
                # Find the best-matching found author. A candidate that
                # fully matches (surname + initials both agree) is always
                # preferred over one that merely scores higher on raw
                # surname similarity but fails the initials check.
                best_score = 0.0
                best_name  = ""
                best_passed = False
                for fa in found_authors:
                    if _author_first_letter_ok(cited, fa):
                        sim, name_ok = author_name_similarity(cited, fa)
                        is_better = (name_ok and not best_passed) or \
                                    (name_ok == best_passed and sim > best_score)
                        if is_better:
                            best_score  = sim
                            best_name   = fa
                            best_passed = name_ok
                matched.append(best_passed)
                author_details.append({
                    "cited":                  cited,
                    "best_match":             best_name,
                    "similarity":             round(best_score, 4),
                    "levenshtein_similarity":  round(levenshtein_similarity(cited, best_name), 4) if best_name else 0.0,
                    "threshold":              SURNAME_THRESHOLD,
                    "passed":                 best_passed,
                })

            # ── Count parity check ─────────────────────────────────────────────
            # The one-directional match above only verifies that each cited author
            # exists in the found list.  It does NOT catch the case where the
            # reference omits authors that CrossRef knows about.
            # Fix: if the counts differ and the raw reference has no "et al.",
            # flag a mismatch (APA 7 requires all authors listed for ≤20-author works).
            has_etal = "et al" in ref.raw.lower()
            count_ok = has_etal or (len(ref.authors) == len(found_authors))
            # Warn when et al. is used but CrossRef only lists a small author count
            # APA 7 requires et al. only for works with 21+ authors
            if has_etal and len(found_authors) < 21:
                result.warnings.append(
                    f"'et al.' used but {source} only lists {len(found_authors)} author(s). "
                    "APA 7 requires 'et al.' only for works with 21 or more authors."
                )

            result.authors_match = all(matched) and count_ok

            # ── Rec 4: store the author breakdown ─────────────────────────
            result.similarity_breakdown["authors"] = {
                "per_author":    author_details,
                "count_cited":   len(ref.authors),
                "count_found":   len(found_authors),
                "count_match":   count_ok,
                "has_et_al":     has_etal,
                "all_matched":   all(matched),
                "final_match":   result.authors_match,
                "method":        "surname/initials-separated comparison (strict surname match + prefix-compatible initials), first-letter guard",
            }

            if not result.authors_match:
                # Collect individual mismatch details
                mismatched_names = [a for a, m in zip(ref.authors, matched) if not m]
                err_parts: list[str] = []

                if mismatched_names:
                    err_parts.append(
                        f"Author mismatch. These cited authors were not found in {source}:\n"
                        + "".join(f"    • {a}\n" for a in mismatched_names)
                        + f"    Found: {', '.join(found_authors)}"
                    )

                if not count_ok:
                    # Find which found authors are completely absent from the cited list
                    missing = [
                        fa for fa in found_authors
                        if not any(author_name_similarity(c, fa)[1] for c in ref.authors)
                    ]
                    err_parts.append(
                        f"Author count mismatch: reference lists "
                        f"{len(ref.authors)} author(s) but {source} has "
                        f"{len(found_authors)}."
                        + (f"\n    Omitted author(s): {', '.join(missing)}" if missing else "")
                        + ("\n    Add missing authors or use 'et al.' as appropriate." if missing else "")
                    )

                for ep in err_parts:
                    result.errors.append(ep)

        elif not found_authors:
            result.warnings.append(f"Authors not found in {source}.")

        # ── DOI ────────────────────────────────────────────────────────────────
        # BUG (fixed): When CrossRef is the metadata source, comparing
        # ref.doi == found_doi is *circular* — CrossRef is queried WITH ref.doi
        # so it always echoes it back, making doi_match trivially True even when
        # the DOI is wrong.
        #
        # Correct approach: the DOI is "right" only if it resolves to THIS paper.
        # We already know whether CrossRef's title matches the cited title
        # (result.title_match).  Use that as the ground truth:
        #   title_match = True  →  DOI resolves to the correct paper  → doi_match True
        #   title_match = False →  DOI resolves to a DIFFERENT paper  → doi_match False
        #   title_match = None  →  can't tell                         → doi_match None
        #
        # When the source is web-page scraping the DOI *is* independently found,
        # so string comparison is legitimate there.
        found_doi = meta.get("doi", "")
        if ref.doi:
            result.doi_found = found_doi or ref.doi   # show what CrossRef echoed back

            if source == "CrossRef":
                if result.title_match is True:
                    result.doi_match = True   # DOI resolves to the correct paper ✔
                elif result.title_match is False:
                    result.doi_match = False  # DOI resolves to a different paper ✘
                    result.errors.append(
                        f"DOI is likely incorrect — it resolves to a different paper.\n"
                        f"    Cited DOI     : {ref.doi}\n"
                        f"    Cited title   : {ref.title[:120]}\n"
                        f"    CrossRef found: {result.title_found[:120]}"
                    )
                # else title_match is None → leave doi_match as None (cannot verify)
            elif found_doi:
                # Independent DOI from web-page metadata — safe to compare strings
                result.doi_match = normalize(ref.doi) == normalize(found_doi)
                if not result.doi_match:
                    result.errors.append(
                        f"DOI mismatch.\n"
                        f"    Cited : {ref.doi}\n"
                        f"    Found : {found_doi}"
                    )
            else:
                result.warnings.append(f"DOI not found in {source} for verification.")

        # ── Year (strict exact match only) ────────────────────────────────────
        found_year = meta.get("year", "")
        if found_year and ref.year:
            result.date_found = found_year
            result.date_match = False
            
            # Extract All Digits: capture every number in the year field
            cited_numbers = re.findall(r"\d+", ref.year)
            
            # The Format Guard
            if len(cited_numbers) != 1:
                result.date_match = False
                result.errors.append("Format Error: Extra characters or multiple years detected.")
            else:
                extracted_cited = cited_numbers[0]
                
                # Boundary Check: Ensure the year is exactly 4 digits
                if len(extracted_cited) != 4:
                    result.date_match = False
                    result.errors.append(f"Format Error: Year must be exactly 4 digits, got '{extracted_cited}'.")
                else:
                    # The Value Check: compare only if it's one exactly 4-digit number
                    found_year_match = re.search(r"\b(\d{4})\b", found_year.strip())
                    extracted_found = found_year_match.group(1) if found_year_match else found_year.strip()
                    
                    if extracted_cited == extracted_found:
                        result.date_match = True
                    else:
                        result.date_match = False
                        # ── Bug fix: title-search fallback can return a
                        # duplicate/mirror record with a wrong year ────────
                        # When there's no DOI, metadata comes from a
                        # bibliographic title search rather than an
                        # authoritative direct DOI lookup. CrossRef is known
                        # to carry multiple duplicate "posted-content"
                        # registrations of the same paper (mirrors/reprints
                        # registered later than the original), all sharing
                        # the same incorrect year. Verified directly against
                        # CrossRef's API: searching "Attention is all you
                        # need" returns 7 identical title/author records, all
                        # dated 2025, none carrying the real 2017 date — so
                        # there is no better candidate to fall back to; the
                        # fallback source itself is simply unauthoritative
                        # here. A cited year at or before the found year is
                        # consistent with citing the original and the search
                        # surfacing a later duplicate, so that combination is
                        # downgraded to a warning instead of a hard error.
                        # A cited year AFTER the found year is the opposite,
                        # more suspicious pattern (citing something as newer
                        # than any record found) and still reported as an
                        # error.
                        _is_unauthoritative_source = (source == "CrossRef (title search)")
                        _cited_before_or_at_found = int(extracted_cited) <= int(extracted_found)
                        if _is_unauthoritative_source and _cited_before_or_at_found:
                            result.warnings.append(
                                f"Year could not be reliably confirmed — {source} is a lower-confidence "
                                f"fallback (used because no DOI was found), and returned a duplicate/mirror "
                                f"record dated differently from the citation.\n"
                                f"    Cited : {ref.year.strip()}\n"
                                f"    Found : {found_year.strip()} (via {source})"
                            )
                        else:
                            result.errors.append(
                                f"Year mismatch.\n"
                                f"    Cited : {ref.year.strip()}\n"
                                f"    Found : {found_year.strip()}"
                            )
                
        elif ref.year and not found_year:
            result.warnings.append(f"Publication year not found in {source}.")

        # ── 3b. Provenance verification (Rec 3 + New Rec C) ─────────────────
        # New Rec C: Comprehensive multi-field cross-verification.
        # Cross-checks journal, conference, publisher, volume, issue, page,
        # and ISBN against the reference text and reports a verification
        # confidence level based on how many fields could be independently
        # confirmed.
        _vm = {}
        _ref_body = ref.rest if ref.rest else ref.raw
        _body_lower = normalize(_ref_body) if _ref_body else ""
        _body_words = set(_body_lower.split()) if _body_lower else set()

        # Track how many fields were checked and how many passed
        _cross_checks_attempted = 0
        _cross_checks_passed    = 0

        # ── Journal name cross-check ─────────────────────────────────────
        _journal = meta.get("journal", "")
        if _journal:
            _vm["journal"] = _journal
            _journal_words = set(normalize(_journal).split()) - _STOPWORDS
            if _journal_words:
                _cross_checks_attempted += 1
                _journal_overlap = len(_journal_words & _body_words) / len(_journal_words)
                _journal_match = _journal_overlap >= 0.5
                _vm["journal_in_reference"] = _journal_match
                _vm["journal_overlap"] = round(_journal_overlap, 4)
                if _journal_match:
                    _cross_checks_passed += 1
                elif len(_journal_words) >= 2:
                    result.warnings.append(
                        f"Journal name from {source} ('{_journal}') "
                        f"does not appear in the reference text. "
                        f"Verify the cited source is correct."
                    )

        # ── Conference name cross-check (NEW in Rec C) ───────────────────
        _conference = meta.get("conference", "")
        if _conference:
            _vm["conference"] = _conference
            _conf_words = set(normalize(_conference).split()) - _STOPWORDS
            if _conf_words and len(_conf_words) >= 2:
                _cross_checks_attempted += 1
                _conf_overlap = len(_conf_words & _body_words) / len(_conf_words)
                _conf_match = _conf_overlap >= 0.4
                _vm["conference_in_reference"] = _conf_match
                _vm["conference_overlap"] = round(_conf_overlap, 4)
                if _conf_match:
                    _cross_checks_passed += 1
                else:
                    result.warnings.append(
                        f"Conference name from {source} ('{_conference}') "
                        f"does not appear in the reference text."
                    )

        # ── Publisher cross-check ────────────────────────────────────────
        _publisher = meta.get("publisher", "")
        if _publisher:
            _vm["publisher"] = _publisher
            _pub_words = set(normalize(_publisher).split()) - _STOPWORDS
            if _pub_words and len(_pub_words) >= 1:
                _cross_checks_attempted += 1
                _pub_overlap = len(_pub_words & _body_words) / len(_pub_words)
                _pub_match = _pub_overlap >= 0.4
                _vm["publisher_in_reference"] = _pub_match
                _vm["publisher_overlap"] = round(_pub_overlap, 4)
                if _pub_match:
                    _cross_checks_passed += 1

        # ── Volume cross-check ───────────────────────────────────────────
        _volume = meta.get("volume", "")
        if _volume:
            _vm["volume"] = _volume
            if _ref_body:
                _cross_checks_attempted += 1
                _vol_found = _volume in _ref_body
                _vm["volume_in_reference"] = _vol_found
                if _vol_found:
                    _cross_checks_passed += 1
                else:
                    result.warnings.append(
                        f"Volume number from {source} is '{_volume}' "
                        f"but was not found in the reference text."
                    )

        # ── Issue cross-check (NEW in Rec C) ─────────────────────────────
        _issue = meta.get("issue", "")
        if _issue:
            _vm["issue"] = _issue
            if _ref_body:
                _cross_checks_attempted += 1
                _iss_found = _issue in _ref_body
                _vm["issue_in_reference"] = _iss_found
                if _iss_found:
                    _cross_checks_passed += 1

        # ── Page range cross-check (NEW in Rec C) ────────────────────────
        _page = meta.get("page", "")
        if _page:
            _vm["page"] = _page
            if _ref_body:
                _cross_checks_attempted += 1
                # Normalize dash variants for comparison
                _page_norm = _page.replace("\u2013", "-").replace("\u2014", "-")
                _body_norm = _ref_body.replace("\u2013", "-").replace("\u2014", "-")
                _pg_found = _page_norm in _body_norm
                _vm["page_in_reference"] = _pg_found
                if _pg_found:
                    _cross_checks_passed += 1

        # ── ISBN cross-check (NEW in Rec C) ──────────────────────────────
        _isbn = meta.get("isbn", [])
        if _isbn:
            _isbn_list = _isbn if isinstance(_isbn, list) else [_isbn]
            _vm["isbn"] = _isbn_list
            if _ref_body and _isbn_list:
                _cross_checks_attempted += 1
                # Check if any ISBN appears in the reference text
                _isbn_found = any(
                    isbn.replace("-", "") in _ref_body.replace("-", "")
                    for isbn in _isbn_list
                )
                _vm["isbn_in_reference"] = _isbn_found
                if _isbn_found:
                    _cross_checks_passed += 1
                else:
                    result.warnings.append(
                        f"ISBN from {source} ({', '.join(_isbn_list)}) "
                        f"was not found in the reference text."
                    )

        # ── ISSN (informational, no cross-check) ─────────────────────────
        _issn = meta.get("issn", [])
        if _issn:
            _vm["issn"] = _issn if isinstance(_issn, list) else [_issn]

        # ── Subject / type (informational) ───────────────────────────────
        _doc_type = meta.get("type", "")
        if _doc_type:
            _vm["type"] = _doc_type
        _subject = meta.get("subject", [])
        if _subject:
            _vm["subject"] = _subject

        # ── Document type verification ───────────────────────────────────
        _EXPECTED_TYPES = {
            "journal-article", "book-chapter", "book", "proceedings-article",
            "monograph", "report", "dissertation", "posted-content",
            "reference-entry", "edited-book",
        }
        if _doc_type and _doc_type not in _EXPECTED_TYPES:
            result.warnings.append(
                f"Document type from {source} is '{_doc_type}' — "
                f"this may not be a standard research paper or book chapter. "
                f"Verify the reference is citing a legitimate scholarly source."
            )

        # ── New Rec C: Verification confidence ───────────────────────────
        # Summarise how many metadata fields were independently confirmed
        # against the reference text.  This gives the user a sense of how
        # thoroughly the reference was verified beyond just title/author/year.
        _vm["cross_checks"] = {
            "attempted": _cross_checks_attempted,
            "passed":    _cross_checks_passed,
            "confidence": (
                "high"   if _cross_checks_attempted >= 3 and _cross_checks_passed >= 3 else
                "medium" if _cross_checks_attempted >= 2 and _cross_checks_passed >= 2 else
                "low"    if _cross_checks_attempted >= 1 and _cross_checks_passed >= 1 else
                "none"
            ),
            "details": (
                f"{_cross_checks_passed}/{_cross_checks_attempted} provenance fields "
                f"confirmed in the reference text"
            ),
        }

        result.verified_metadata = _vm

    # ── New Rec E: Intelligent Source Recommendation ───────────────────────
    # Only offer alternatives when verification actually failed — a title
    # mismatch, or no metadata found at all — not for references that
    # already checked out fine.
    _verification_failed = (result.title_match is False) or (not meta and ref.title)
    if _verification_failed:
        try:
            result.recommended_sources = recommend_alternative_sources(ref)
        except Exception:
            result.recommended_sources = []

    # ── Rec 4: Year, DOI, link breakdowns + overall weighted score ────────
    if ref.year:
        result.similarity_breakdown["year"] = {
            "cited":  ref.year.strip(),
            "found":  result.date_found.strip() if result.date_found else "",
            "match":  result.date_match,
            "method": "exact 4-digit comparison",
        }

    if ref.doi:
        result.similarity_breakdown["doi"] = {
            "cited":  ref.doi,
            "found":  result.doi_found,
            "match":  result.doi_match,
            "method": ("title-based verification (CrossRef echoes queried DOI, "
                       "so match is inferred from title match)"
                       if result.metadata_source in ("CrossRef", "CrossRef (title search)")
                       else "string comparison against independently found DOI"),
        }

    if ref.url:
        result.similarity_breakdown["link"] = {
            "url":       ref.url,
            "reachable": result.link_reachable,
            "status":    result.link_status,
        }

    # ── Rec 5B: Expose search candidates in the breakdown ──────────────
    # When the title-search path was used, show all candidates that
    # were considered so the user can see alternative matches.
    if _search_candidates:
        result.similarity_breakdown["search_candidates"] = {
            "total_evaluated": len(_search_candidates),
            "best_score":      round(_best_search_score, 4),
            "candidates":      _search_candidates[:10],   # top 10 for display
            "note": (
                "These are the top research papers found via CrossRef title search, "
                "ranked by combined similarity (title + authors + year + journal). "
                "The highest-scoring candidate was used for verification."
            ),
        }

    # ── Overall weighted integrity score (improved by New Rec D) ───────────
    # New Rec D: The score now incorporates provenance cross-checks alongside
    # the primary field checks.  Instead of relying only on title/authors/year/
    # DOI/link, the score also considers whether the journal, publisher,
    # conference, volume, issue, page, and ISBN matched the reference text.
    #
    # Primary checks (high weight — these confirm identity):
    #   title=0.30, authors=0.20, year=0.10, DOI=0.10, link=0.05  = 0.75
    # Provenance checks (lower weight — these confirm legitimacy):
    #   provenance=0.25 (based on cross_checks.passed / cross_checks.attempted)
    #
    # This means a reference with perfect primary checks but zero provenance
    # confirmation scores 75%, not 100%.  All fields passing scores 100%.
    _checks = []
    _weights = {}
    if result.title_match is not None:
        _checks.append(("title",   result.title_match,   0.30))
        _weights["title"] = "30%"
    if result.authors_match is not None:
        _checks.append(("authors", result.authors_match, 0.20))
        _weights["authors"] = "20%"
    if result.date_match is not None:
        _checks.append(("year",    result.date_match,    0.10))
        _weights["year"] = "10%"
    if result.doi_match is not None:
        _checks.append(("doi",     result.doi_match,     0.10))
        _weights["doi"] = "10%"
    if result.link_reachable is not None:
        _checks.append(("link",    result.link_reachable, 0.05))
        _weights["link"] = "5%"

    # ── New Rec D: Provenance checks as a combined score ─────────────────
    _prov_score = 0.0
    _prov_detail = "not available"
    _vm = result.verified_metadata
    if _vm and "cross_checks" in _vm:
        _cc = _vm["cross_checks"]
        _attempted = _cc.get("attempted", 0)
        _passed    = _cc.get("passed", 0)
        if _attempted > 0:
            _prov_score = _passed / _attempted
            _prov_detail = f"{_passed}/{_attempted} fields confirmed"
            _checks.append(("provenance", _prov_score >= 0.5, 0.25))
            _weights["provenance"] = "25%"

    if _checks:
        _total_weight = sum(w for _, _, w in _checks)
        _weighted = sum(w * (1.0 if p else 0.0) for _, p, w in _checks)
        _score = _weighted / _total_weight if _total_weight else 0.0
        result.similarity_breakdown["overall"] = {
            "weighted_integrity_score": round(_score, 4),
            "weights":    _weights,
            "components": {name: p for name, p, _ in _checks},
            "provenance_detail": _prov_detail,
            "formula": (
                "Σ(weight × pass) / Σ(weights). "
                "Primary checks: title, authors, year, DOI, link. "
                "Provenance check: journal/conference/publisher/volume/issue/page/ISBN "
                "combined — passes if ≥50% of available fields confirmed."
            ),
            "note": "Only checks that were actually performed are included.",
        }

    return result


# ── Report printer ────────────────────────────────────────────────────────────
ICON_OK   = f"{Fore.GREEN}✔{Style.RESET_ALL}"
ICON_FAIL = f"{Fore.RED}✘{Style.RESET_ALL}"
ICON_WARN = f"{Fore.YELLOW}⚠{Style.RESET_ALL}"
ICON_SKIP = f"{Fore.CYAN}–{Style.RESET_ALL}"


def fmt_check(label: str, value: Optional[bool], detail: str = "") -> str:
    if value is True:
        icon, colour = ICON_OK,   Fore.GREEN
    elif value is False:
        icon, colour = ICON_FAIL, Fore.RED
    else:
        icon, colour = ICON_SKIP, Fore.CYAN
    txt = f"  {icon} {colour}{label}{Style.RESET_ALL}"
    if detail:
        txt += f"  {Style.DIM}({detail}){Style.RESET_ALL}"
    return txt


def print_result(idx: int, vr: ValidationResult) -> None:
    ref = vr.reference
    sep = "─" * 72
    fmt_label = ref.fmt.upper()

    print(f"\n{Fore.CYAN}{Style.BRIGHT}Reference #{idx} [{fmt_label}]{Style.RESET_ALL}")
    print(sep)
    raw_display = ref.raw if len(ref.raw) <= 120 else ref.raw[:117] + "…"
    print(f"{Style.DIM}{raw_display}{Style.RESET_ALL}\n")

    if ref.doi_only and ref.formatted_citation:
        print(f"  {Fore.GREEN}✔ DOI resolved — formatted citation:{Style.RESET_ALL}")
        print(f"  {ref.formatted_citation}\n")

    print(fmt_check(f"{fmt_label} format", ref.valid_apa,
                    "" if ref.valid_apa else ref.parse_error))

    if not ref.valid_apa:
        print(f"\n  {Fore.RED}⛔ Skipping online checks — fix format first.{Style.RESET_ALL}")
        print(sep)
        return

    print(fmt_check("Link reachable", vr.link_reachable,
                    vr.link_status or (ref.url or "no URL/DOI")))

    title_detail = (f"{vr.title_score:.0%} similarity"
                    if vr.title_found else "not verified")
    print(fmt_check("Title matches",  vr.title_match,  title_detail))
    print(fmt_check("Authors match",  vr.authors_match))
    print(fmt_check("DOI matches",    vr.doi_match,
                    ref.doi if ref.doi else "no DOI in reference"))
    print(fmt_check("Year matches",   vr.date_match,
                    f"cited {ref.year}" +
                    (f" / found {vr.date_found}" if vr.date_found else "")))

    if vr.errors:
        print(f"\n  {Fore.RED}{Style.BRIGHT}Issues:{Style.RESET_ALL}")
        for e in vr.errors:
            for line_e in e.splitlines():
                print(f"  {Fore.RED}  {line_e}{Style.RESET_ALL}")

    if vr.warnings:
        print(f"\n  {Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        for w in vr.warnings:
            print(f"  {Fore.YELLOW}  {w}{Style.RESET_ALL}")

    print(sep)


def print_summary(results: list[ValidationResult]) -> None:
    total   = len(results)
    fmt_ok  = sum(1 for r in results if r.reference.valid_apa)
    link_ok = sum(1 for r in results if r.link_reachable is True)
    errors  = sum(1 for r in results if r.errors)
    clean   = sum(1 for r in results if not r.errors and r.reference.valid_apa)

    print(f"\n{'═' * 72}")
    print(f"{Fore.CYAN}{Style.BRIGHT}  SUMMARY{Style.RESET_ALL}")
    print(f"{'═' * 72}")
    print(f"  Total references checked   : {total}")
    print(f"  Valid format               : {fmt_ok}/{total}")
    print(f"  Reachable links            : {link_ok}/{total}")
    print(f"  References with issues     : {Fore.RED}{errors}{Style.RESET_ALL}/{total}")
    print(f"  Fully passing              : {Fore.GREEN}{clean}{Style.RESET_ALL}/{total}")
    print(f"{'═' * 72}\n")


# ── Input helpers ─────────────────────────────────────────────────────────────
BANNER = f"""
{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════════════╗
║     Multi-Format Reference Validator (APA 7 · MLA · Chicago) ║
║     Algorithms: Format Check · CrossRef · Link Check · Fuzzy Match   ║
╚══════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""


def collect_references_interactive() -> tuple[list[str], str]:
    fmt_prompt = (
        f"\n{Fore.CYAN}Select format [apa7 / mla / chicago] "
        f"(default: apa7):{Style.RESET_ALL} "
    )
    fmt = input(fmt_prompt).strip().lower() or FORMAT_APA7
    if fmt not in SUPPORTED_FORMATS:
        print(f"{Fore.YELLOW}Unknown format '{fmt}', defaulting to APA7.{Style.RESET_ALL}")
        fmt = FORMAT_APA7

    print(f"\n{Style.DIM}Enter references, blank line between each. "
          f"Type DONE when finished.{Style.RESET_ALL}\n")
    refs: list[str] = []
    current: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        stripped = line.strip()
        if stripped.upper() == "DONE":
            if current:
                refs.append(" ".join(current))
            break
        elif stripped == "":
            if current:
                refs.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    return refs, fmt


# ══════════════════════════════════════════════════════════════════════════════
#  CHAPTER 2 / RRL  CITATION–REFERENCE INTEGRITY CHECKER
# ══════════════════════════════════════════════════════════════════════════════

def extract_chapter2_section(text: str) -> dict:
    """Detect and extract the Chapter 2 / Review of Related Literature section."""
    result = {"found": False, "heading": "", "start": 0, "end": 0,
              "text": "", "char_count": 0}

    _CH2_PATTERNS = [
        r'(?:CHAPTER\s+(?:II|2)\s*[:\.]?\s*\n?\s*)'
        r'(?:REVIEW\s+OF\s+RELATED\s+(?:LITERATURE|STUDIES)|'
        r'RELATED\s+(?:LITERATURE|STUDIES|WORKS)|LITERATURE\s+REVIEW)',
        r'(?:CHAPTER\s+(?:II|2))\b',
        r'(?:^|\n)\s*(?:REVIEW\s+OF\s+RELATED\s+(?:LITERATURE|STUDIES)|'
        r'Review\s+of\s+Related\s+(?:Literature|Studies))',
        r'(?:^|\n)\s*(?:RELATED\s+(?:LITERATURE|STUDIES|WORKS)|'
        r'Related\s+(?:Literature|Studies|Works))',
        r'(?:^|\n)\s*(?:LITERATURE\s+REVIEW|Literature\s+Review)',
    ]

    heading_match = None
    for pat in _CH2_PATTERNS:
        m = re.search(pat, text, re.MULTILINE | re.IGNORECASE)
        if m:
            heading_match = m
            break
    if not heading_match:
        return result

    section_start = heading_match.end()
    _NEXT_CHAPTER = re.compile(
        r'\n\s*(?:CHAPTER\s+(?:III|IV|V|3|4|5)\b|'
        r'METHODOLOGY|Methodology|RESEARCH\s+DESIGN|Research\s+Design|'
        r'RESULTS?\s+AND\s+DISCUSSION|Results?\s+and\s+Discussion|'
        r'REFERENCES\s*\n|References\s*\n)', re.IGNORECASE
    )
    end_match = _NEXT_CHAPTER.search(text[section_start:])
    section_end = section_start + end_match.start() if end_match else len(text)
    section_text = text[section_start:section_end].strip()

    result.update(found=True, heading=heading_match.group(0).strip(),
                  start=section_start, end=section_end,
                  text=section_text, char_count=len(section_text))
    return result


def extract_in_text_citations(text: str) -> list[dict]:
    """
    Extract in-text citations from body text.
    Supports narrative, parenthetical, multi-cite, and IEEE bracket styles.
    Each citation includes evidence: the sentence and paragraph where it appears.
    Multiple occurrences of the same citation are tracked.
    """
    if not text:
        return []
    citations = []
    seen_keys = {}   # key -> index in citations list
    _IGNORE = {'chapter','table','figure','section','page','volume','issue',
               'edition','version','step','phase','part','item','type','group'}

    def _extract_sentence(pos, raw_len):
        start = pos
        while start > 0 and text[start - 1] not in '.!?\n':
            start -= 1
        while start < pos and text[start] in ' \t\n':
            start += 1
        end = pos + raw_len
        while end < len(text) and text[end] not in '.!?\n':
            end += 1
        if end < len(text):
            end += 1
        return re.sub(r'\s+', ' ', text[start:end]).strip()

    def _extract_paragraph(pos):
        ps = max(0, pos - 250)
        pe = min(len(text), pos + 250)
        while ps > 0 and text[ps - 1] not in '\n':
            ps -= 1
        while pe < len(text) and text[pe] not in '\n':
            pe += 1
        return re.sub(r'\s+', ' ', text[ps:pe]).strip()[:600]

    def _add(authors, year, raw, style, pos):
        # Extract the primary surname — for narrative citations like "Smith (2021)"
        # or "Later Smith (2021)", the surname is the LAST word before "et al." / "&"
        # Special case: IEEE bracket citations like "[1]" use the bracket as the key
        if style == "ieee":
            surname = authors.strip().lower()  # e.g. "[1]"
            key = (surname, year or "ieee")
        else:
            words = re.split(r'[\s,&]', authors)
            words = [w for w in words if w and w[0].isupper() and w.lower() not in ('et', 'al', 'and')]
            surname = words[-1].strip().lower() if words else ""
            key = (surname, year)
        if len(surname) < 1 or surname in _IGNORE:
            return
        sentence = _extract_sentence(pos, len(raw))
        occurrence = {"position": pos, "sentence": sentence}
        if key in seen_keys:
            idx = seen_keys[key]
            citations[idx]["occurrences"].append(occurrence)
            citations[idx]["occurrence_count"] += 1
        else:
            paragraph = _extract_paragraph(pos)
            seen_keys[key] = len(citations)
            citations.append({
                "authors": authors.strip(), "year": year,
                "citation_text": raw.strip(), "style": style,
                "context": sentence, "position": pos,
                "evidence_sentence": sentence,
                "evidence_paragraph": paragraph,
                "occurrence_count": 1,
                "occurrences": [occurrence],
            })

    _NAR = re.compile(
        r'([A-Z][a-z]+(?:-[A-Z][a-z]+)?'
        r'(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?'
        r'|&\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?'
        r'|[A-Z][a-z]+(?:-[A-Z][a-z]+)?))*'
        r')\s*\((\d{4})\)'
    )
    _HEADING_WORDS = {'writing','academic','research','study','analysis','review',
                      'method','result','discussion','conclusion','introduction',
                      'chapter','section','framework'}
    for m in _NAR.finditer(text):
        auth = m.group(1).strip()
        if len(auth) > 60:
            continue
        if any(w in auth.lower() for w in _HEADING_WORDS):
            continue
        _add(auth, m.group(2), m.group(0), "narrative", m.start())

    _PAR = re.compile(
        r'\(([A-Z][a-z]+(?:-[A-Z][a-z]+)?'
        r'(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?'
        r'|and\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?))*'
        r')\s*,\s*(\d{4})\)'
    )
    for m in _PAR.finditer(text):
        _add(m.group(1), m.group(2), m.group(0), "parenthetical", m.start())

    _MULTI = re.compile(r'\(([^)]*?\b\d{4}[^)]*?;\s*[^)]*?\b\d{4}[^)]*?)\)')
    for m in _MULTI.finditer(text):
        for part in re.split(r';\s*', m.group(1)):
            sub = re.match(r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))*)\s*,?\s*(\d{4})', part.strip())
            if sub:
                _add(sub.group(1), sub.group(2), part.strip(), "parenthetical", m.start())

    for m in re.finditer(r'\[(\d+(?:\s*[-\u2013,]\s*\d+)*)\]', text):
        for n in re.findall(r'\d+', m.group(1)):
            _add(f"[{n}]", "", m.group(0), "ieee", m.start())

    return citations

def extract_full_text_citations(text: str) -> set:
    """Quick scan of ENTIRE thesis for (surname, year) pairs.
    Handles 'Author (Year)', '(Author, Year)', and 'Author et al. (Year)'."""
    keys = set()
    _IGNORE = {'chapter','table','figure','section','page','volume','issue',
               'edition','version','step','phase','part','group','type'}
    # Pattern 1: Author (Year) or Author et al. (Year)
    for m in re.finditer(
        r'([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*(?:et\s+al\.?\s*)?\((\d{4})\)', text
    ):
        s = m.group(1).lower()
        if s not in _IGNORE:
            keys.add((s, m.group(2)))
    # Pattern 2: (Author, Year) or (Author & Author, Year)
    for m in re.finditer(
        r'\(([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*(?:et\s+al\.?\s*)?[,&]\s*(\d{4})\)', text
    ):
        s = m.group(1).lower()
        if s not in _IGNORE:
            keys.add((s, m.group(2)))
    return keys


def match_citation_to_reference(citation: dict, parsed_refs: list) -> dict:
    """
    Find the best matching reference for one in-text citation.
    Reuses existing char_similarity() for author comparison.
    """
    cite_auth = citation["authors"]
    cite_year = citation["year"]
    cite_surname = re.split(r'[\s,&]', cite_auth)[0].strip()

    if not cite_surname or len(cite_surname) < 2:
        return {"matched": False, "ref_index": None, "ref_text": "",
                "author_score": 0, "year_match": False, "title": "",
                "doi": "", "status": "UNVERIFIABLE",
                "details": "Citation has insufficient author information."}

    best_match = None
    best_score = 0.0

    for idx, ref in enumerate(parsed_refs):
        if not ref.authors:
            continue
        # Check ALL authors in the reference, not just the first
        top_sim = 0.0
        for ref_author in ref.authors:
            ref_surname = ref_author.split(",")[0].strip()
            sim = char_similarity(cite_surname, ref_surname)
            if sim > top_sim:
                top_sim = sim
        surname_sim = top_sim
        year_match = (cite_year == ref.year) if cite_year and ref.year else False
        score = surname_sim * 0.6 + (0.4 if year_match else 0.0)
        if "et al" in cite_auth.lower() and len(ref.authors) >= 3:
            score += 0.05
        if score > best_score:
            best_score = score
            best_match = (idx, ref, surname_sim, year_match)

    _NOT_CHECKED = {"label": "not_checked", "confidence": 0.0}

    if best_match is None:
        return {"matched": False, "ref_index": None, "ref_text": "",
                "author_score": 0, "year_match": False, "title": "",
                "doi": "", "status": "MISSING", "entailment": _NOT_CHECKED,
                "details": f"No reference found for {cite_auth} ({cite_year})."}

    idx, ref, surname_sim, year_match = best_match
    if surname_sim >= 0.75 and year_match:
        status, details = "MATCH", "Author and year match."
    elif surname_sim >= 0.75:
        status = "MISMATCH"
        details = f"Author matches but year differs: cited {cite_year}, reference {ref.year}."
    elif surname_sim >= 0.5 and year_match:
        status, details = "MATCH", f"Author partially matches ({int(surname_sim*100)}%), year matches."
    elif surname_sim >= 0.5:
        status = "MISMATCH"
        details = f"Author partially matches ({int(surname_sim*100)}%), year differs."
    else:
        return {"matched": False, "ref_index": None, "ref_text": "",
                "author_score": surname_sim, "year_match": False, "title": "",
                "doi": "", "status": "MISSING", "entailment": _NOT_CHECKED,
                "details": f"Best candidate too dissimilar ({int(surname_sim*100)}%)."}

    # ── New Rec E: RTE entailment check (only for confirmed matches — no
    #    point spending a Semantic Scholar call + model inference on a
    #    citation that isn't even matched to a reference yet) ─────────────
    entailment = (
        classify_citation_entailment(citation.get("evidence_sentence", ""), ref,
                                     citation_text=citation.get("citation_text", ""))
        if status == "MATCH" else _NOT_CHECKED
    )

    return {"matched": status == "MATCH", "ref_index": idx + 1,
            "ref_text": ref.raw[:200], "author_score": round(surname_sim, 4),
            "year_match": year_match, "title": ref.title,
            "doi": ref.doi, "status": status, "details": details,
            "entailment": entailment}


def analyze_thesis_integrity(full_text: str, fmt: str = FORMAT_APA7) -> dict:
    """
    CORE FUNCTION: Analyze full thesis for citation-reference integrity.
    """
    report = {"chapter2": {}, "citations": [], "references": [],
              "matches": [], "missing": [], "uncited_ch2": [],
              "uncited_anywhere": [], "mismatches": [], "summary": {},
              "errors": []}

    # Stage 1: Detect Chapter 2
    ch2 = extract_chapter2_section(full_text)
    report["chapter2"] = {"found": ch2["found"], "heading": ch2["heading"],
                          "char_count": ch2["char_count"]}
    if not ch2["found"]:
        report["errors"].append(
            "Could not detect Chapter 2 / Review of Related Literature. "
            "Looked for: Chapter II, Review of Related Literature, "
            "Literature Review, Related Studies, etc.")
        return report

    # Clean the Chapter 2 text: normalize line breaks and whitespace
    # so that citations split across lines are properly detected
    ch2_clean = re.sub(r'\r\n|\r', '\n', ch2["text"])
    ch2_clean = re.sub(r'(?<=[a-z,])\n(?=[A-Za-z])', ' ', ch2_clean)  # join wrapped lines
    ch2_clean = re.sub(r'\s+', ' ', ch2_clean)

    # Stage 2: Extract citations from Chapter 2
    ch2_citations = extract_in_text_citations(ch2_clean)
    report["citations"] = ch2_citations
    if not ch2_citations:
        report["errors"].append("No in-text citations detected in Chapter 2.")

    # Stage 3: Extract and parse References
    ref_strings = split_text_into_references(full_text, fmt)
    parsed_refs = [parse_reference(raw, fmt) for raw in ref_strings]
    report["references"] = [
        {"index": i+1, "raw": r.raw[:200], "authors": r.authors,
         "year": r.year, "title": r.title, "doi": r.doi, "valid": r.valid_apa}
        for i, r in enumerate(parsed_refs)
    ]
    if not ref_strings:
        report["errors"].append("No references extracted from References section.")

    # Stage 4: Match each citation to a reference
    matched_ref_indices = set()
    for cite in ch2_citations:
        if cite["style"] == "ieee":
            num_str = cite["authors"].strip("[]")
            try:
                num = int(num_str) - 1
                if 0 <= num < len(parsed_refs):
                    ref = parsed_refs[num]
                    mr = {"matched": True, "ref_index": num+1,
                          "ref_text": ref.raw[:200], "author_score": 1.0,
                          "year_match": True, "title": ref.title,
                          "doi": ref.doi, "status": "MATCH",
                          "details": f"IEEE [{num+1}] matched.",
                          "entailment": classify_citation_entailment(
                              cite.get("evidence_sentence", ""), ref,
                              citation_text=cite.get("citation_text", ""))}
                else:
                    mr = {"matched": False, "ref_index": None, "ref_text": "",
                          "author_score": 0, "year_match": False, "title": "",
                          "doi": "", "status": "MISSING",
                          "entailment": {"label": "not_checked", "confidence": 0.0},
                          "details": f"IEEE [{num_str}] has no reference."}
            except ValueError:
                mr = {"matched": False, "ref_index": None, "ref_text": "",
                      "author_score": 0, "year_match": False, "title": "",
                      "doi": "", "status": "UNVERIFIABLE",
                      "entailment": {"label": "not_checked", "confidence": 0.0},
                      "details": "Invalid bracket number."}
        else:
            mr = match_citation_to_reference(cite, parsed_refs)

        report["matches"].append({**cite, **mr})
        if mr["matched"] and mr["ref_index"]:
            matched_ref_indices.add(mr["ref_index"])

    # Stage 5: Categorize
    report["missing"] = [e for e in report["matches"] if e["status"] == "MISSING"]
    report["mismatches"] = [e for e in report["matches"] if e["status"] == "MISMATCH"]

    # Stage 6: Uncited references
    # Clean full text for scanning
    clean_full = re.sub(r'\r\n|\r', '\n', full_text)
    clean_full = re.sub(r'(?<=[a-z,])\n(?=[A-Za-z])', ' ', clean_full)
    all_cite_keys = extract_full_text_citations(clean_full)
    for i, ref in enumerate(parsed_refs):
        if (i + 1) not in matched_ref_indices:
            ref_surname = ref.authors[0].split(",")[0].strip().lower() if ref.authors else ""
            cited_anywhere = (ref_surname, ref.year) in all_cite_keys if ref_surname else False
            entry = {"ref_index": i+1, "ref_text": ref.raw[:200],
                     "authors": ref.authors, "year": ref.year,
                     "title": ref.title, "cited_anywhere": cited_anywhere}
            if cited_anywhere:
                report["uncited_ch2"].append(entry)
            else:
                report["uncited_anywhere"].append(entry)

    # Stage 7: Summary
    total_c = len(ch2_citations)
    matched_n = sum(1 for e in report["matches"] if e["status"] == "MATCH")
    report["summary"] = {
        "total_citations_in_ch2": total_c,
        "total_references": len(parsed_refs),
        "citations_matched": matched_n,
        "citations_missing_from_refs": len(report["missing"]),
        "citations_mismatched": len(report["mismatches"]),
        "refs_not_cited_in_ch2": len(report["uncited_ch2"]),
        "refs_not_cited_anywhere": len(report["uncited_anywhere"]),
        "integrity_score": round(matched_n / total_c * 100, 1) if total_c else 0,
    }
    return report


def _is_fragmented_pdf_text(text: str) -> bool:
    """
    Detects PDFs that extracted with one word per line — each real word
    isolated on its own line, the next word separated by a line holding
    just a single space (observed directly from a real thesis export).
    True when most non-empty lines hold only a single token.
    """
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False
    single_token_frac = sum(1 for l in non_empty if len(l.strip().split()) <= 1) / len(non_empty)
    return single_token_frac >= 0.6


def _split_fragmented_pdf_references(ref_text: str) -> list[str]:
    """
    Reference-splitting strategy for one-word-per-line PDF extractions (see
    _is_fragmented_pdf_text). There is no reliable line/whitespace signal
    for reference boundaries in this extraction style — verified directly
    against a real file: the gap between two consecutive references is a
    single blank-ish line, IDENTICAL to an ordinary word-to-word gap; only
    a coincidental page break produces a larger gap, so blank-line-run
    length can't be used as a boundary signal here.

    Instead, this fully flattens the text into one continuous string and
    scans for "Surname, Initial." shaped candidate reference starts
    wherever they occur, applying the same gate the line-based splitter
    uses for wrapped multi-author lists: a candidate only counts as a new
    reference if the entry accumulated so far already contains a (YYYY)
    year. A multi-author list inside ONE reference produces many such
    candidates before any year appears (e.g. "Agarwal, A., Arafa, M.,
    Avidor-Reiss, T., ... & Shah, R. (2023)."); a genuine new reference
    only starts once the previous entry's year — and therefore the whole
    entry — is already present.
    """
    flat = re.sub(r"\s+", " ", ref_text).strip()
    if not flat:
        return []

    _RE_IEEE     = re.compile(r"\[\d+\]")
    _RE_NUMBERED = re.compile(r"(?<=[.\s])\d{1,3}[.)]\s+(?=[A-Z])")
    _RE_NAME     = re.compile(r"[A-Z][A-Za-z'\-]+,\s+[A-Z]\.")

    # Structural markers are unambiguous regardless of position — prefer
    # them outright when present.
    for pattern in (_RE_IEEE, _RE_NUMBERED):
        starts = [m.start() for m in pattern.finditer(flat)]
        if len(starts) >= 2:
            bounds = starts + [len(flat)]
            return [
                flat[bounds[i]:bounds[i + 1]].strip()
                for i in range(len(bounds) - 1)
                if len(flat[bounds[i]:bounds[i + 1]].strip()) >= 25
            ]

    candidates = list(_RE_NAME.finditer(flat))
    if not candidates:
        return [flat] if len(flat) >= 25 else []

    starts = [candidates[0].start()]
    entry_start = candidates[0].start()
    for m in candidates[1:]:
        segment_so_far = flat[entry_start:m.start()]
        if re.search(r"\((?:19|20)\d{2}", segment_so_far):
            starts.append(m.start())
            entry_start = m.start()
    starts.append(len(flat))

    return [
        flat[starts[i]:starts[i + 1]].strip()
        for i in range(len(starts) - 1)
        if len(flat[starts[i]:starts[i + 1]].strip()) >= 25
    ]


def split_text_into_references(text: str, fmt: str = "") -> list[str]:
    """
    Extract individual references from PDF-extracted text.

    This function performs three stages:
      1. DETECT the reference/bibliography section in the full document text
      2. SPLIT the section into individual references using format-aware patterns
      3. CLEAN each reference (join wrapped lines, collapse whitespace)

    Parameters:
      text: full text extracted from a PDF (all pages concatenated)
      fmt:  optional citation format hint ("apa7", "mla", "chicago", "ieee")
            — used to improve splitting accuracy. If empty, auto-detected.

    Returns:
      list of individual reference strings, each cleaned and joined.
    """
    if not text or not text.strip():
        return []

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 1: Detect the references section
    # ══════════════════════════════════════════════════════════════════════
    # Look for common heading patterns. The heading may appear as:
    #   "References\n"  "REFERENCES\n"  "Bibliography\n"  "Works Cited\n"
    #   "Literature Cited\n"  "Reference List\n"
    # It may also have a section number: "7. References" or "VII. References"

    _HEADING_PATTERN = re.compile(
        r'(?:^|\n)\s*'
        r'(?:[IVXLC]+\.?\s+|[0-9]+\.?\s+)?'      # optional section number
        r'(REFERENCES|References|BIBLIOGRAPHY|Bibliography|'
        r'WORKS\s+CITED|Works\s+Cited|'
        r'LITERATURE\s+CITED|Literature\s+Cited|'
        r'REFERENCE\s+LIST|Reference\s+List|'
        r'CITED\s+REFERENCES|Cited\s+References|'
        r'LIST\s+OF\s+REFERENCES|List\s+of\s+References)'
        r'\s*\n',
        re.MULTILINE
    )

    # ── Bug fix: don't just take the FIRST heading match ────────────────
    # The word "references" often appears in ordinary body prose well
    # before the actual bibliography — e.g. a methodology section
    # describing how the system "detects the References or Bibliography
    # section" satisfies this exact pattern, especially on PDFs that
    # extract with one word per line (common from some export tools),
    # where a genuine newline ends up on both sides of nearly every word.
    # Taking the first match there swept up the entire rest of the
    # document into a single "reference." A genuine reference-list heading
    # is reliably followed closely by multiple "Surname, Initial."-shaped
    # entries; try candidates from last to first (the real section is
    # almost always near the end) and take the first one that's actually
    # followed by that pattern.
    _REF_START_PROBE = re.compile(r'[A-Z][a-z]+,\s+[A-Z]\.')
    heading_matches = list(_HEADING_PATTERN.finditer(text))
    heading_match = None
    for _candidate in reversed(heading_matches):
        _following = text[_candidate.end():_candidate.end() + 3000]
        if len(_REF_START_PROBE.findall(_following)) >= 2:
            heading_match = _candidate
            break
    if heading_match is None and heading_matches:
        heading_match = heading_matches[-1]

    if heading_match:
        ref_text = text[heading_match.end():]
        # Truncate at the next major section heading (if any) that comes
        # AFTER the references.  Common post-reference sections:
        # Appendix, Appendices, Author Bio, Acknowledgments, etc.
        _NEXT_SECTION = re.compile(
            r'\n\s*(?:[IVXLC]+\.?\s+|[0-9]+\.?\s+)?'
            r'(?:APPENDI[CX]|Appendi[cx]|'
            r'ACKNOWLEDGE?MENTS?|Acknowledge?ments?|'
            r'AUTHOR\s+BIO|Author\s+Bio|'
            r'ABOUT\s+THE\s+AUTHORS?|About\s+the\s+Authors?|'
            r'SUPPLEMENTARY|Supplementary|'
            r'VITA|Vita|CURRICULUM|Curriculum|'
            r'GLOSSARY|Glossary)\b',
            re.MULTILINE
        )
        next_section = _NEXT_SECTION.search(ref_text)
        if next_section:
            ref_text = ref_text[:next_section.start()]
    else:
        # No heading found — use the entire text as-is.
        # This handles PDFs that contain ONLY references (no body text).
        ref_text = text

    ref_text = ref_text.strip()
    if not ref_text:
        return []

    # ── Bug fix: one-word-per-line PDF extractions need a different
    # splitting strategy entirely — every line-based heuristic below
    # assumes a "line" can contain a full pattern like "Surname, Initial.",
    # which is never true when each word is isolated on its own line (see
    # _split_fragmented_pdf_references for why a whitespace-based reflow
    # doesn't work either — verified directly that reference-to-reference
    # gaps and ordinary word gaps are indistinguishable by blank-line count
    # in this extraction style).
    if _is_fragmented_pdf_text(ref_text):
        return _split_fragmented_pdf_references(ref_text)

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 2: Split into individual references
    # ══════════════════════════════════════════════════════════════════════
    # PDF text typically has NO blank lines between references.
    # References are separated by the start of the next reference,
    # which follows a predictable pattern depending on the format.
    #
    # Strategy: split the text into lines, then walk through the lines
    # and decide whether each line STARTS a new reference or CONTINUES
    # the previous one.

    lines = ref_text.split('\n')

    # ── Reference-start patterns ──────────────────────────────────────
    # APA:     "Surname, I." or "Surname, I. A." at start of line
    # MLA:     "Lastname, Firstname" at start of line
    # Chicago: "Lastname, Firstname." or "Lastname, Firstname. YYYY."
    # IEEE:    "[N]" bracket number at start of line
    # Generic: a line starting with an uppercase letter followed by a surname pattern

    _RE_IEEE_START = re.compile(r'^\s*\[\d+\]\s*')
    _RE_APA_START  = re.compile(
        r'^[A-Z\u00C0-\u00DC][A-Za-z\u00e0-\u00ff\'\-]+,\s+'
        r'(?:[A-Z]\.[\s\-]*)+',
    )
    _RE_NAME_START = re.compile(
        r'^[A-Z\u00C0-\u00DC][A-Za-z\u00e0-\u00ff\'\-]+,\s+[A-Z]'
    )
    _RE_NUMBERED   = re.compile(r'^\s*\d{1,3}[\.\)]\s+[A-Z]')

    def _is_strong_ref_start(line: str) -> bool:
        """
        Unambiguous structural markers — an IEEE bracket or a numbered-list
        entry can't appear anywhere except the start of a genuine new
        reference, so these always split regardless of what came before.
        """
        stripped = line.strip()
        return bool(_RE_IEEE_START.match(stripped) or _RE_NUMBERED.match(stripped))

    def _is_name_shaped(line: str) -> bool:
        """APA author pattern ('Surname, I.') or a general 'Surname, Name'
        shape. On its own this is NOT reliable evidence of a new reference —
        see _current_group_has_year() below for why."""
        stripped = line.strip()
        return bool(_RE_APA_START.match(stripped) or _RE_NAME_START.match(stripped))

    def _current_group_has_year(group: list[str]) -> bool:
        """
        A genuine reference has its year somewhere in the entry, so by the
        time a real *next* reference begins, the one being accumulated
        should already contain one. A multi-author list commonly wraps
        mid-list — e.g. '...Jones, L.,' / 'Kaiser, L., & Polosukhin, I.
        (2017)...' — and that wrap point has the exact same 'Surname,
        Initial.' shape as a new reference's opening. Requiring a year
        before honoring that shape as a split point is what tells the two
        cases apart: no year yet means still-mid-author-list, not a fresh
        entry.

        Uses the bare YEAR_PATTERN, not an APA-style '(YYYY' check — MLA
        ("...vol. 5, no. 1, 2020, pp. 1-10.") and IEEE place the year as a
        plain token with no parentheses, and a parens-only check would
        under-split those two formats (verified: it silently merged two
        separate MLA references into one until this was widened).
        """
        return bool(YEAR_PATTERN.search(' '.join(group)))

    # ── Walk through lines and group them ─────────────────────────────
    groups: list[list[str]] = []
    current_group: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty lines: if we have accumulated content, this might
            # separate references (in some PDF layouts)
            if current_group:
                # Check if the next non-empty line starts a new reference
                # For now, keep going (don't split on empty lines alone
                # since PDF extraction inserts spurious empty lines)
                pass
            continue

        starts_new = current_group and (
            _is_strong_ref_start(stripped) or
            (_is_name_shaped(stripped) and _current_group_has_year(current_group))
        )
        if starts_new:
            # This line starts a new reference — save the previous group
            groups.append(current_group)
            current_group = [stripped]
        else:
            current_group.append(stripped)

    if current_group:
        groups.append(current_group)

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 3: Clean and join each reference
    # ══════════════════════════════════════════════════════════════════════
    refs = []
    for group in groups:
        # Join the lines of this reference with spaces
        joined = " ".join(group)
        # Collapse multiple spaces
        joined = re.sub(r'\s+', ' ', joined).strip()
        # Fix broken words from line-wrap hyphenation: "refer-\nence" → "reference"
        joined = re.sub(r'(\w)-\s+(\w)', r'\1\2', joined)

        # Filter out non-reference content:
        # - Too short to be a real reference
        if len(joined) < 25:
            continue
        # - Looks like a page header/footer (page numbers, running titles)
        if re.match(r'^\d+$', joined):
            continue
        if re.match(r'^Page\s+\d+', joined, re.IGNORECASE):
            continue

        refs.append(joined)

    return refs


def detect_citation_format(text: str) -> str:
    """
    Auto-detects which of the four supported citation styles a document's
    reference list most likely uses, so a PDF upload doesn't require the
    user to correctly guess and pre-select the format themselves.

    Reference-boundary detection in split_text_into_references() doesn't
    actually depend much on which format is assumed — its line-start
    patterns check for all four formats' shapes regardless of the fmt
    argument — so this splits once, then tries parsing every resulting
    entry against each format's own parser and picks whichever format the
    most entries validate against. No new heuristics: just reuses the
    parsers that already exist for each format.
    """
    sample_refs = split_text_into_references(text, FORMAT_APA7)
    if not sample_refs:
        return FORMAT_APA7   # nothing to go on — keep the existing default

    scores = {fmt: 0 for fmt in SUPPORTED_FORMATS}
    for raw in sample_refs:
        for fmt in SUPPORTED_FORMATS:
            if parse_reference(raw, fmt).valid_apa:
                scores[fmt] += 1

    best_fmt = max(scores, key=scores.get)
    return best_fmt if scores[best_fmt] > 0 else FORMAT_APA7


def collect_references_from_file(path: str, fmt: str = FORMAT_APA7) -> tuple[list[str], str]:
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    refs = split_text_into_references(raw, fmt)
    return refs, fmt


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    print(BANNER)

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        fmt      = sys.argv[2] if len(sys.argv) > 2 else FORMAT_APA7
        print(f"Reading references from: {Fore.CYAN}{filepath}{Style.RESET_ALL}\n")
        try:
            raw_refs, fmt = collect_references_from_file(filepath)
            if len(sys.argv) > 2:
                fmt = sys.argv[2]
        except FileNotFoundError:
            print(f"{Fore.RED}File not found: {filepath}{Style.RESET_ALL}")
            sys.exit(1)
    else:
        raw_refs, fmt = collect_references_interactive()

    if not raw_refs:
        print(f"{Fore.YELLOW}No references provided. Exiting.{Style.RESET_ALL}")
        return

    print(f"\n{Fore.CYAN}Validating {len(raw_refs)} reference(s) [{fmt.upper()}]…{Style.RESET_ALL}")

    results: list[ValidationResult] = []
    for i, raw in enumerate(raw_refs, start=1):
        print(f"\n  [{i}/{len(raw_refs)}] Parsing & checking…", end="", flush=True)
        ref    = parse_reference(raw, fmt)
        result = validate_reference(ref)
        results.append(result)
        time.sleep(0.3)
        print(f"\r  [{i}/{len(raw_refs)}] Done.              ")

    for i, vr in enumerate(results, start=1):
        print_result(i, vr)

    print_summary(results)


if __name__ == "__main__":
    main()