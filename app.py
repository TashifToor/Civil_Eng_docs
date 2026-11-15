import os
import re
import io
import json
import glob
import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from groq import Groq

# OCR (images + scanned PDFs). Agar installed na ho to app phir bhi chalti hai.
try:
    import pytesseract
    from PIL import Image, ImageOps
    import fitz  # pymupdf
    pytesseract.get_tesseract_version()
    OCR_OK = True
except Exception:
    OCR_OK = False

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]   # pehla try hota hai, fail ho to doosra
FAST_MODEL = "openai/gpt-oss-20b"
CHUNK_WORDS = 180
OVERLAP = 40
TOP_K = 6
MAX_OCR_PAGES = 80
EXTS = (".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp")
INDEX_FILE = "index.json"

st.set_page_config(page_title="Civil Engineering Study Bot", page_icon="🏗️", layout="wide")

# ---------- API key ----------
def get_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

KEY = get_key()
if not KEY:
    st.error("GROQ_API_KEY nahi mila. Secrets me key daalo.")
    st.stop()
client = Groq(api_key=KEY)

# ---------- Reading + chunking ----------
STOP = set("the a an of and or to in is are was were be for on with as by at this that it from which what how why when do does did can you me please explain define".split())

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]

def split_words(name, page, text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    out = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if len(piece) > 60:
            out.append({"source": name, "page": page, "text": piece})
    return out

def pdf_to_chunks(name, data):
    chunks, ocr_pages, fdoc = [], 0, None
    reader = PdfReader(io.BytesIO(data))
    for pno, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if len(text) < 30 and OCR_OK and ocr_pages < MAX_OCR_PAGES:
            try:
                if fdoc is None:
                    fdoc = fitz.open(stream=data, filetype="pdf")
                pix = fdoc[pno - 1].get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = " ".join(pytesseract.image_to_string(img).split())
                ocr_pages += 1
            except Exception:
                text = ""
        if text:
            chunks += split_words(name, pno, text)
    return chunks

def image_to_chunks(name, data):
    if not OCR_OK:
        return []
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("L")   # WhatsApp photos ki rotation theek
    img = ImageOps.autocontrast(img)
    if img.width < 1600:
        img = img.resize((img.width * 2, img.height * 2))
    text = pytesseract.image_to_string(img)
    if len(text.strip()) < 30:
        text = pytesseract.image_to_string(img, config="--psm 6")
    text = " ".join(text.split())
    return split_words(name, 1, text) if text else []

def any_to_chunks(name, data):
    low = name.lower()
    if low.endswith(".pdf"):
        return pdf_to_chunks(name, data)
    if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return image_to_chunks(name, data)
    return split_words(name, 1, data.decode("utf-8", errors="ignore"))

# ---------- Index: ek baar banao, index.json me save, dobara na padho ----------
def safe_size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1

def doc_files():
    out = []
    for p in sorted(glob.glob("docs/**/*", recursive=True)):
        if p.lower().endswith(EXTS) and os.path.isfile(p) and safe_size(p) >= 0:
            out.append(p)
    return out

def folder_signature():
    return [[p.replace("\\", "/"), safe_size(p)] for p in doc_files()]

@st.cache_resource(show_spinner="Docs padh raha hun (sirf pehli baar, phir index.json save hoga)...")
def load_base(sig_key):
    sig = folder_signature()
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # index.json maujood hai to hamesha wahi use karo (docs dobara nahi padhe jayenge)
            return data, "loaded"
        except Exception:
            pass
    chunks, warnings = [], []
    for p in doc_files():
        name = os.path.basename(p)
        try:
            with open(p, "rb") as f:
                c = any_to_chunks(name, f.read())
        except Exception:
            c = []
        chunks += c
        if not c:
            warnings.append(f"{name}: text nahi mila" + ("" if OCR_OK else " (OCR off hai)"))
    data = {"sig": sig, "chunks": chunks, "warnings": warnings}
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass
    return data, "built"

def clean_text(t):
    t = re.sub("\u00ad\\s*", "", t)               # soft hyphen (aur uske baad ka space)
    t = re.sub(r"(\w)[-\u2010\u2011]\s+(\w)", r"\1\2", t)   # "con- tinues" -> "continues"
    t = re.sub(r"[^\x20-\x7E]+", " ", t)          # ajeeb symbols hatao
    t = re.sub(r"([.\-_~*=:;|'\"])\1{2,}", r"\1", t)  # ....... ----- etc
    return " ".join(t.split())

def is_good(t):
    """Garbage (maps, noise, OCR kachra) wale chunks hata do."""
    nospace = t.replace(" ", "")
    if len(nospace) < 60:
        return False
    letters = sum(ch.isalpha() for ch in nospace) / len(nospace)
    toks = t.split()
    wordlike = sum(1 for w in toks if len(re.sub(r"[^A-Za-z]", "", w)) >= 3) / max(len(toks), 1)
    return letters >= 0.4 and wordlike >= 0.3

@st.cache_resource(show_spinner="Search index ban raha hai...")
def build_search(sig_key, uploads):
    data, _ = load_base(sig_key)
    chunks = list(data["chunks"])
    warnings = list(data.get("warnings", []))
    for name, content in uploads:
        c = any_to_chunks(name, content)
        chunks += c
        if not c:
            warnings.append(f"{name}: text nahi mila" + ("" if OCR_OK else " (OCR off hai)"))
    cleaned = []
    for c in chunks:
        txt = clean_text(c["text"])
        if is_good(txt):
            cleaned.append({"source": c["source"], "page": c["page"], "text": txt})
    chunks = cleaned
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
    return chunks, bm25, warnings

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    style = st.selectbox("Answer style", [
        "Detailed (step by step)",
        "Short (exam revision)",
        "Numerical solve karo (formulas + steps)",
        "MCQs banao (practice)",
    ])
    lang = st.selectbox("Language", ["Jis me main likhun", "English", "Roman Urdu"])
    uploads_files = st.file_uploader("Extra PDF / image / TXT upload karo",
                                     type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp"],
                                     accept_multiple_files=True)
    if st.button("🗑️ Chat clear"):
        st.session_state.messages = []
        st.rerun()

uploads = tuple((f.name, f.getvalue()) for f in (uploads_files or []))
sig_key = str(folder_signature())
base, status = load_base(sig_key)
chunks, bm25, warns = build_search(sig_key, uploads)

with st.sidebar:
    st.caption(f"📚 {len(chunks)} chunks loaded")
    st.caption("🔍 OCR: ON" if OCR_OK else "🔍 OCR: OFF")
    st.caption("⚡ index.json se load hua (fast)" if status == "loaded" else "🛠️ Docs abhi padhe gaye")
    st.download_button("⬇️ index.json download",
                       data=json.dumps(base, ensure_ascii=False),
                       file_name="index.json", mime="application/json")
    st.caption("Ye file GitHub repo me daal do, phir cloud par dobara padhna nahi padega.")
    for w in warns:
        st.warning(w)

# ---------- Retrieval ----------
def english_keywords(q):
    try:
        r = client.chat.completions.create(
            model=FAST_MODEL,
            extra_body={"reasoning_effort": "low"},
            messages=[
                {"role": "system", "content": "Student ka sawal Roman Urdu/Urdu/English me ho sakta hai. Usay civil engineering textbook search ke liye 8-15 English keywords/phrases me convert karo. Sirf keywords output karo, aur kuch nahi."},
                {"role": "user", "content": q},
            ],
            temperature=0, max_tokens=500,
        )
        return r.choices[0].message.content
    except Exception:
        return ""

SMALL = {"hi", "hello", "hey", "hlo", "helo", "salam", "salaam", "aoa", "assalamualaikum", "asalam",
         "thanks", "thank", "shukriya", "ok", "okay", "bye", "yo"}

def is_smalltalk(q):
    words = re.findall(r"[a-zA-Z]+", q.lower())
    if not words:
        return True
    return len(words) <= 4 and words[0] in SMALL

def retrieve(q):
    if not bm25:
        return []
    query = q + " " + english_keywords(q)
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    if not top or scores[top[0]] <= 0:
        return []
    best = scores[top[0]]
    return [chunks[i] for i in top if scores[i] >= 0.4 * best]

STYLE_PROMPT = {
    "Detailed (step by step)": "Explain in detail, step by step, with headings and formulas.",
    "Short (exam revision)": "Answer in very short bullet points for exam revision.",
    "Numerical solve karo (formulas + steps)": "If it is a numerical, write the formula, substitute values, solve step by step with units, and highlight the final answer.",
    "MCQs banao (practice)": "Create 5 MCQs (4 options each) on the topic from the context, then give answers with a short explanation.",
}

LANG_RULE = {
    "Jis me main likhun": "Reply in the same language the student writes in, but ONLY English or Roman Urdu (Urdu written in English letters). If unsure, use English.",
    "English": "Reply in English.",
    "Roman Urdu": "Reply in Roman Urdu (Urdu written in English letters); keep technical terms in English.",
}
NO_SCRIPT = "NEVER use Hindi, Devanagari, Urdu/Arabic script or any non-Latin script. Use only English letters."

def stream_answer(history, hits):
    if hits:
        context = "\n\n".join(f"[{i+1}] ({h['source']}, page {h['page']})\n{h['text']}" for i, h in enumerate(hits))
        system = f"""You are a civil engineering study assistant helping a student prepare for an exam.
Answer ONLY from the CONTEXT below (the student's documents). {STYLE_PROMPT[style]}
{LANG_RULE[lang]} {NO_SCRIPT}
Mention sources like (filename, page X).
If the context does not contain the answer, say clearly that it was not found in the docs and do not guess.
The context may come from OCR, so ignore obvious garbage characters and small spelling or formula errors.

CONTEXT:
{context}"""
    else:
        system = f"""You are a friendly civil engineering study assistant. The student sent a greeting or small talk.
Reply in 1-2 short sentences and ask which topic they want to study. {LANG_RULE[lang]} {NO_SCRIPT}"""
    msgs = [{"role": "system", "content": system}] + history[-6:]
    last_err = None
    for mdl in MODELS:
        try:
            stream = client.chat.completions.create(
                model=mdl, messages=msgs, temperature=0.2, stream=True,
                extra_body={"reasoning_effort": "low"},
            )
            for part in stream:
                d = part.choices[0].delta.content
                if d:
                    yield d
            return
        except Exception as e:
            last_err = e
    raise last_err

# ---------- Chat UI ----------
st.title("🏗️ Civil Engineering Study Bot")
st.caption("Sirf tumhare docs se jawab deta hai, source page ke sath.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📖 Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['source']} (p.{s['page']})**\n\n{s['text'][:500]}...")

if prompt := st.chat_input("Sawal poochho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not chunks:
            st.warning("Koi doc load nahi hua. docs/ folder me files daalo ya sidebar se upload karo.")
        else:
            try:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                if is_smalltalk(prompt):
                    answer = st.write_stream(stream_answer(history, []))
                    hits = []
                else:
                    hits = retrieve(prompt)
                    if not hits:
                        answer = "Ye docs me nahi mila. Sawal ko thore alag alfaz me poochho."
                        st.markdown(answer)
                    else:
                        answer = st.write_stream(stream_answer(history, hits))
                        with st.expander("📖 Sources"):
                            for s_ in hits[:4]:
                                st.markdown(f"**{s_['source']} (p.{s_['page']})**\n\n{s_['text'][:300]}...")
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits[:4]})
            except Exception as e:
                st.error(f"Error: {e}\n\nRate limit ho to 1 minute ruko aur dobara poochho.")