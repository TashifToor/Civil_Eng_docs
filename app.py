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
    from PIL import Image
    import fitz  # pymupdf
    pytesseract.get_tesseract_version()
    OCR_OK = True
except Exception:
    OCR_OK = False

MODEL = "llama-3.3-70b-versatile"   # limit aaye to "llama-3.1-8b-instant" kar do
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
    img = Image.open(io.BytesIO(data)).convert("RGB")
    text = " ".join(pytesseract.image_to_string(img).split())
    return split_words(name, 1, text) if text else []

def any_to_chunks(name, data):
    low = name.lower()
    if low.endswith(".pdf"):
        return pdf_to_chunks(name, data)
    if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return image_to_chunks(name, data)
    return split_words(name, 1, data.decode("utf-8", errors="ignore"))

# ---------- Index: ek baar banao, index.json me save, dobara na padho ----------
def doc_files():
    return [p for p in sorted(glob.glob("docs/**/*", recursive=True))
            if os.path.isfile(p) and p.lower().endswith(EXTS)]

def folder_signature():
    return [[p.replace("\\", "/"), os.path.getsize(p)] for p in doc_files()]

@st.cache_resource(show_spinner="Docs padh raha hun (sirf pehli baar, phir index.json save hoga)...")
def load_base(sig_key):
    sig = folder_signature()
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # docs folder khali ho (sirf index push kiya) ya docs same hon -> index use karo
            if not sig or data.get("sig") == sig:
                return data, "loaded"
        except Exception:
            pass
    chunks, warnings = [], []
    for p in doc_files():
        name = os.path.basename(p)
        with open(p, "rb") as f:
            c = any_to_chunks(name, f.read())
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
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Student ka sawal Roman Urdu/Urdu/English me ho sakta hai. Usay civil engineering textbook search ke liye 8-15 English keywords/phrases me convert karo. Sirf keywords output karo, aur kuch nahi."},
                {"role": "user", "content": q},
            ],
            temperature=0, max_tokens=60,
        )
        return r.choices[0].message.content
    except Exception:
        return ""

def retrieve(q):
    if not bm25:
        return []
    query = q + " " + english_keywords(q)
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [chunks[i] for i in top if scores[i] > 0]

STYLE_PROMPT = {
    "Detailed (step by step)": "Detailed, step by step samjhao, headings aur formulas ke sath.",
    "Short (exam revision)": "Bohat short bullet points me do, exam revision ke liye.",
    "Numerical solve karo (formulas + steps)": "Numerical ho to formula likho, values rakho, units ke sath step by step solve karo aur final answer highlight karo.",
    "MCQs banao (practice)": "Context ke topic par 5 MCQs banao (4 options), last me answers aur chhoti explanation do.",
}

def stream_answer(history, hits):
    context = "\n\n".join(f"[{i+1}] ({h['source']}, page {h['page']})\n{h['text']}" for i, h in enumerate(hits))
    lang_rule = {"Jis me main likhun": "Student jis language me likhe usi me jawab do.",
                 "English": "Reply in English.",
                 "Roman Urdu": "Roman Urdu me jawab do (technical terms English me)."}[lang]
    system = f"""Tum ek civil engineering study assistant ho. Student ke exam ki tayyari hai.
SIRF neeche diye CONTEXT (student ke docs) se jawab do. {STYLE_PROMPT[style]} {lang_rule}
Jawab me source likho jaise (filename, page X).
Agar context me jawab na ho to saaf bolo "Docs me ye nahi mila" aur guess mat karo.
Note: context OCR se bhi aa sakta hai, to chhoti spelling/formula ki ghaltiyan ho sakti hain.

CONTEXT:
{context}"""
    msgs = [{"role": "system", "content": system}] + history[-6:]
    stream = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2, stream=True)
    for part in stream:
        d = part.choices[0].delta.content
        if d:
            yield d

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
                hits = retrieve(prompt)
                if not hits:
                    answer = "Docs me ye nahi mila. Sawal ko alag alfaz me poochho."
                    st.markdown(answer)
                else:
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    answer = st.write_stream(stream_answer(history, hits))
                    with st.expander("📖 Sources"):
                        for s in hits:
                            st.markdown(f"**{s['source']} (p.{s['page']})**\n\n{s['text'][:500]}...")
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
            except Exception as e:
                st.error(f"Error: {e}\n\nRate limit ho to 1 minute ruko ya MODEL ko llama-3.1-8b-instant kar do.")