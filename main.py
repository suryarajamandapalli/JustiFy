# main.py
import os
import io
import json
import uuid
import time
import platform
import requests
from typing import List, Optional, Dict, Any
from difflib import get_close_matches

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Optional BeautifulSoup for case page parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
HOSTNAME = os.getenv("HOSTNAME", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "300"))

STATIC_DIR = "static"
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

if platform.system() == "Windows":
    # Adjust path if tesseract installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# best-effort init of optional LLM clients
gemini_client = None
openai_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client ready")
    except Exception as e:
        gemini_client = None
        print("Gemini init failed:", e)

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client ready")
    except Exception as e:
        openai_client = None
        print("OpenAI init failed:", e)

app = FastAPI(title="JustiFy Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# load statutes.json if present (optional)
STATUTES_FILE = "statutes.json"
if os.path.exists(STATUTES_FILE):
    try:
        with open(STATUTES_FILE, "r", encoding="utf-8") as f:
            STATUTES = json.load(f)
            if not isinstance(STATUTES, list):
                STATUTES = [STATUTES]
    except Exception as e:
        print("Error loading statutes.json:", e)
        STATUTES = []
else:
    STATUTES = []

# simple in-memory cache for LLM responses
_llm_cache: Dict[str, Dict[str, Any]] = {}
def llm_cache_get(key: str) -> Optional[Dict[str, Any]]:
    rec = _llm_cache.get(key)
    if not rec:
        return None
    if time.time() - rec["ts"] > LLM_CACHE_TTL:
        del _llm_cache[key]
        return None
    return rec["value"]
def llm_cache_set(key: str, value: Dict[str, Any]):
    _llm_cache[key] = {"ts": time.time(), "value": value}

def save_upload(file: UploadFile) -> (str, str):
    ext = os.path.splitext(file.filename)[1] or ".bin"
    uid = uuid.uuid4().hex
    filename = f"{uid}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as out:
        out.write(file.file.read())
    url = f"http://{HOSTNAME}:{PORT}/static/uploads/{filename}"
    return url, path

def ocr_from_path(path: str) -> str:
    try:
        img = Image.open(path).convert("RGB")
        txt = pytesseract.image_to_string(img)
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"[OCR failed: {e}]"

def normalize_statute_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        return {
            "title": str(entry.get("title") or entry.get("name") or "").strip(),
            "keywords": entry.get("keywords") if isinstance(entry.get("keywords"), (list,tuple)) else [],
            "snippet": str(entry.get("snippet") or entry.get("summary") or "").strip(),
            "link": str(entry.get("link") or entry.get("url") or "").strip(),
            "id": str(entry.get("id") or entry.get("code") or "").strip()
        }
    if isinstance(entry, (list, tuple)):
        for it in entry:
            if isinstance(it, str) and it.strip():
                return {"title": it.strip(), "keywords": [], "snippet": "", "link": "", "id": ""}
        return {"title": "", "keywords": [], "snippet": "", "link": "", "id": ""}
    if isinstance(entry, str):
        return {"title": entry.strip(), "keywords": [], "snippet": "", "link": "", "id": ""}
    return {"title": "", "keywords": [], "snippet": "", "link": "", "id": ""}

def match_statutes(text: str, max_hits: int = 6) -> List[Dict[str, Any]]:
    text_l = (text or "").lower()
    hits: List[Dict[str, Any]] = []
    for raw in STATUTES:
        s = normalize_statute_entry(raw)
        if not s["title"] and not s["keywords"]:
            continue
        matched = False
        for kw in (s.get("keywords") or []):
            try:
                if kw and kw.lower() in text_l:
                    hits.append(s)
                    matched = True
                    break
            except Exception:
                continue
        if matched and len(hits) >= max_hits:
            break
    if len(hits) < max_hits:
        titles = [normalize_statute_entry(s).get("title","") for s in STATUTES]
        tokens = [w for w in set((text_l or "").split()) if len(w) > 4]
        for token in tokens:
            close = get_close_matches(token, titles, n=max_hits)
            for c in close:
                for raw in STATUTES:
                    s = normalize_statute_entry(raw)
                    if s.get("title") == c and s not in hits:
                        hits.append(s)
                        break
            if len(hits) >= max_hits:
                break
    return hits[:max_hits]

# SERPAPI search for similar judgments (best-effort)
def serpapi_case_search(query: str, max_results: int = 6):
    if not SERPAPI_KEY:
        return []
    try:
        params = {
            "engine": "google",
            "q": f"site:indiankanoon.org {query}",
            "api_key": SERPAPI_KEY,
            "gl": "in",
            "hl": "en",
            "num": max_results
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        hits = []
        for item in data.get("organic_results", [])[:max_results]:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")
            case_meta = {}
            if link and "indiankanoon.org" in link and BS4_AVAILABLE:
                try:
                    rr = requests.get(link, timeout=6)
                    soup = BeautifulSoup(rr.text, "html.parser")
                    h1 = soup.find("h1")
                    if h1:
                        case_meta["case_title"] = h1.get_text(strip=True)
                    meta_date = soup.find("span", class_="date")
                    if meta_date:
                        case_meta["date"] = meta_date.get_text(strip=True)
                except Exception:
                    pass
            hits.append({"title": title, "link": link, "snippet": snippet, **case_meta})
        return hits
    except Exception as e:
        print("SerpApi error:", e)
        return []

def call_llm_generate_structured(user_text: str, ocr_text: str) -> Dict[str, Any]:
    cache_key = f"{(user_text or '')[:300]}||{(ocr_text or '')[:300]}"
    cached = llm_cache_get(cache_key)
    if cached:
        return cached

    system_msg = (
        "You are JustiFy, a concise legal assistant for India. "
        "Return ONLY valid JSON with keys: explanation (string), next_steps (array of short strings), draft (string), confidence (string). "
        "Keep language simple and factual. If unsure, say 'consult a lawyer'."
    )
    user_prompt = f"User text: {user_text}\n\nOCR text: {ocr_text}\n\nProduce the JSON described above."

    llm_text = None

    # Try Gemini
    if gemini_client:
        try:
            full_prompt = system_msg + "\n\n" + user_prompt
            resp = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
            llm_text = getattr(resp, "text", str(resp))
        except Exception as e:
            print("Gemini call failed:", e)
            llm_text = None

    # OpenAI fallback
    if not llm_text and openai_client:
        try:
            resp = openai_client.responses.create(model="gpt-4o-mini", input=system_msg + "\n\n" + user_prompt, max_output_tokens=700)
            if hasattr(resp, "output_text") and resp.output_text:
                llm_text = resp.output_text
            else:
                out = ""
                for part in getattr(resp, "output", []):
                    for item in part.get("content", []):
                        if item.get("type") == "output_text":
                            out += item.get("text", "")
                llm_text = out or None
        except Exception as e:
            print("OpenAI call failed:", e)
            llm_text = None

    # If no LLM available, return a conservative fallback (useful offline)
    if not llm_text:
        fallback = {
            "explanation": (
                "Cloud LLM not reachable — providing conservative guidance. "
                "I cannot access a live LLM right now. Below is conservative, general guidance; consult a qualified lawyer for specifics."
            ),
            "next_steps": [
                "Collect evidence: names, dates, times, documents, photos.",
                "Report to local police or appropriate authority if the matter is criminal.",
                "Seek a qualified lawyer for tailored advice."
            ],
            "draft": (
                "To,\nThe Officer-in-Charge,\n[Your Local Police Station],\n[City]\n\nSubject: Complaint regarding [brief description]\n\nDear Sir/Madam,\nI wish to report that on [date], at [location], the following occurred: [short factual description].\nPlease investigate and take necessary action.\n\nYours faithfully,\n[Your Name]\n[Contact Number]"
            ),
            "confidence": "Not legal advice. Consult a lawyer."
        }
        llm_cache_set(cache_key, fallback)
        return fallback

    # try to parse JSON substring cleanly
    parsed = None
    try:
        start = llm_text.find("{")
        end = llm_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = llm_text[start:end+1]
            parsed = json.loads(candidate)
        else:
            parsed = json.loads(llm_text)
    except Exception:
        try:
            fixed = llm_text.strip().replace("'", '"')
            start = fixed.find("{"); end = fixed.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(fixed[start:end+1])
        except Exception:
            parsed = {
                "explanation": llm_text.strip()[:4000],
                "next_steps": [],
                "draft": "",
                "confidence": "Not legal advice. Consult a lawyer."
            }

    if not isinstance(parsed, dict):
        parsed = {
            "explanation": str(parsed)[:4000],
            "next_steps": [],
            "draft": "",
            "confidence": "Not legal advice. Consult a lawyer."
        }

    # Normalize keys
    parsed.setdefault("explanation", "")
    parsed.setdefault("next_steps", [])
    parsed.setdefault("draft", "")
    parsed.setdefault("confidence", "Not legal advice. Consult a lawyer.")

    # If next_steps is empty, try to extract simple bullets from explanation (server-side)
    if (not parsed.get("next_steps")) and parsed.get("explanation"):
        text = parsed["explanation"]
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        parsed["next_steps"] = sentences[:4]

    llm_cache_set(cache_key, parsed)
    return parsed

class AnalysisResult(BaseModel):
    explanation: str
    next_steps: List[str]
    draft: Dict[str, str]
    statutes: List[dict]
    case_links: List[dict]
    ocr_text: Optional[str] = None
    files: Optional[List[str]] = None
    raw_llm_text: Optional[str] = None
    related_images: Optional[List[str]] = None
    timestamp: Optional[str] = None

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(text: str = Form(...), files: List[UploadFile] = File([])):
    file_urls: List[str] = []
    ocr_parts: List[str] = []
    for f in files:
        try:
            url, path = save_upload(f)
            file_urls.append(url)
            ocr_text_single = ocr_from_path(path)
            if ocr_text_single:
                ocr_parts.append(ocr_text_single)
        except Exception as e:
            ocr_parts.append(f"[failed to process {f.filename}: {e}]")

    ocr_text = "\n\n".join([p for p in ocr_parts if p]).strip()

    combined = (text or "") + "\n\n" + ocr_text
    try:
        statutes = match_statutes(combined, max_hits=6)
    except Exception as e:
        print("match_statutes error:", e)
        statutes = []

    try:
        case_links = serpapi_case_search(text or combined, max_results=6)
    except Exception as e:
        print("serpapi_case_search error:", e)
        case_links = []

    try:
        llm_parsed = call_llm_generate_structured(text or "", ocr_text)
    except Exception as e:
        llm_parsed = {
            "explanation": f"LLM error: {e}. Providing conservative fallback guidance. Consult a lawyer.",
            "next_steps": [
                "Collect evidence: names, dates, documents.",
                "File a local police report if it's criminal.",
                "Seek legal counsel."
            ],
            "draft": "",
            "confidence": "Not legal advice."
        }

    related_images: List[str] = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    draft_obj = {"auto": llm_parsed.get("draft", "")}

    result = {
        "explanation": (llm_parsed.get("explanation", "") or "")[:4000],
        "next_steps": llm_parsed.get("next_steps", []) or [],
        "draft": draft_obj,
        "statutes": statutes,
        "case_links": case_links,
        "ocr_text": ocr_text,
        "files": file_urls,
        "raw_llm_text": json.dumps(llm_parsed, ensure_ascii=False),
        "related_images": related_images,
        "timestamp": timestamp
    }
    return result

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>JustiFy backend running. Put static/index.html there.</h3>")

@app.get("/health")
def health():
    return {"status": "ok"}
