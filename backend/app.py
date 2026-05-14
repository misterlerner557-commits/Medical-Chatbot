import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain"
)
import os
import io
import json
from flask_cors import CORS
import uuid
import time
import random
import logging
import datetime
import threading
from typing import List, Optional
from gtts import gTTS
from flask import send_file
import tempfile
from flask import abort

from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from werkzeug.utils import secure_filename

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
from PIL import Image
import cv2
from pypdf import PdfReader

import smtplib
from email.mime.text import MIMEText

# Project-specific imports (keep unchanged if present)
# ---- REQUIRED imports (FAIL FAST) ----
from src.helper import download_hugging_face_embeddings

# ---- OPTIONAL imports (ISOLATED) ----
try:
    from src.prompt import system_prompt
except Exception:
    system_prompt = "You are a helpful assistant."

try:
    from langchain_pinecone import PineconeVectorStore
except Exception:
    PineconeVectorStore = None

# Twilio
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

# ---------------- app config ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)



UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CHATS_FILE = os.getenv("CHATS_FILE", "chats.json")
chats_dir = os.path.dirname(os.path.abspath(CHATS_FILE))
if chats_dir and not os.path.exists(chats_dir):
    os.makedirs(chats_dir, exist_ok=True)

# ---------------- env & logging ----------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
TWILIO_SID = os.getenv("TWILIO_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

# Model settings (tweak these in .env)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")  # faster default
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1000"))
RAG_K = int(os.getenv("RAG_K", "1"))  # default k for retrieval (1 for speed)

# caches for HF
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "./hf_cache")

# Tesseract (adjust path on Windows)
tess_path = os.getenv("TESSERACT_PATH")
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("medical-chatbot")

# ---------------- HTTP session with retry ----------------
def make_requests_session(total_retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor,
                    status_forcelist=status_forcelist, allowed_methods=frozenset(["GET", "POST"]))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

requests_session = make_requests_session()

# ---------------- RAG lazy init (thread-safe singletons) ----------------
# NOTE: we initialize once at startup. We DO NOT re-initialize on each request.
_rag_lock = threading.Lock()
_rag_initialized = False
_rag_init_error: Optional[str] = None
rag_retriever = None
embeddings = None
pinecone_index_name = os.getenv("PINECONE_INDEX", "medical-chatbot")

conversation_topic = {}
last_user_query = {}


def initialize_rag_once(force=False):
    """
    Lazy, thread-safe initialization of embeddings and retriever.
    Use force=True to reinitialize (for debugging).
    """
    global _rag_initialized, _rag_init_error, embeddings, rag_retriever

    if _rag_initialized and not force:
        return

    with _rag_lock:
        if _rag_initialized and not force:
            return
        try:
            logger.info("🔥 Initializing RAG components...")
            if download_hugging_face_embeddings is None:
                raise RuntimeError("Embeddings loader not available. Ensure src.helper.download_hugging_face_embeddings exists.")

            # Load cached embeddings (thread-safe inside helper)
            embeddings = download_hugging_face_embeddings()
            if embeddings is None:
                raise RuntimeError("Embeddings loader returned None")

            if PineconeVectorStore is None:
                raise RuntimeError("PineconeVectorStore not available. Check your imports and environment.")

            # Create/attach to existing Pinecone index
            docsearch = PineconeVectorStore.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)
            # Use a small k by default for speed (configurable by RAG_K)
            rag_retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": RAG_K})

            _rag_initialized = True
            _rag_init_error = None
            logger.info("✅ RAG initialized successfully.")
        except Exception as e:
            _rag_init_error = str(e)
            _rag_initialized = False
            logger.exception("RAG initialization failed: %s", e)
            
# ---------------- GitHub Model caller (no OpenAI) ----------------
with app.app_context():
    initialize_rag_once()


def call_github_chat_model(system_message: str, user_message: str, model: str = CHAT_MODEL,
                           temperature: float = CHAT_TEMPERATURE, max_tokens: int = CHAT_MAX_TOKENS,
                           timeout: int = 30):
    """
    Call GitHub model inference endpoint.
    This uses the same inference path your logs showed (models.github.ai).
    The request/response shapes vary across providers; we try common fields.
    """
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set in environment.")

    url = os.getenv("GITHUB_MODELS_URL", "https://models.github.ai/inference/chat/completions")
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Add a short client name
        "User-Agent": "medical-chatbot-gh-model/1.0"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        # allow provider to stream or not; we use synchronous here
    }

    try:
        resp = requests_session.post(url, headers=headers, json=payload, timeout=timeout)
        # Try to parse JSON
        j = {}
        try:
            j = resp.json()
        except Exception:
            logger.warning("GitHub model response not JSON; returning raw text")
            return resp.text

        # Common shapes:
        # 1) { "choices": [ { "message": {"content": "..."}} ] }
        # 2) { "answer": "..." }
        # 3) provider-specific fields
        if isinstance(j, dict):
            if "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
                c = j["choices"][0]
                # try nested message.content
                if isinstance(c, dict):
                    msg = c.get("message") or c.get("delta") or {}
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    # sometimes choice has "text"
                    if "text" in c:
                        return c["text"]
                # fallback: dump the choice
                try:
                    return json.dumps(c)
                except Exception:
                    return str(c)
            if "answer" in j:
                return j["answer"]
            # some providers return 'data' or direct 'result'
            if "result" in j:
                return j["result"]
            # otherwise return stringified JSON
            return json.dumps(j)
        # fallback
        return str(j)
    except requests.RequestException as re:
        logger.exception("HTTP error calling GitHub model: %s", re)
        raise
    
def is_follow_up_question(text: str):
    followups = [
        "that", "it", "them", "those",
        "cause for that", "symptoms for that",
        "what about it", "tell me more",
        "more info", "explain more",
        "why does that happen", "how to treat that"
    ]
    t = text.lower().strip()
    return any(word in t for word in followups)

# ---------------- Intent Detection ----------------

def detect_tts_lang(text: str) -> str:
    # Kannada
    if any('\u0C80' <= ch <= '\u0CFF' for ch in text):
        return "kn"
    # Hindi
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        return "hi"
    # Tamil
    if any('\u0B80' <= ch <= '\u0BFF' for ch in text):
        return "ta"
    # Telugu
    if any('\u0C00' <= ch <= '\u0C7F' for ch in text):
        return "te"
    # Default
    return "en"




def detect_intent(text: str):
    t = text.lower().strip()

    greetings = ["hi", "hello", "hey", "hii","Namaskara"]
    if t in greetings:
        return ("greeting", None)

    # translation
    if "translate" in t or "answer in" in t:
        lang = t.replace("answer in", "").replace("translate to", "").strip()
        return ("translate", lang)

    # disease-only
    diseases = [
        "typhoid", "diabetes", "dengue", "malaria", "cholera",
        "tuberculosis", "tb", "covid", "asthma", "cancer",
        "hypertension", "bp"
    ]
    if t in diseases:
        return ("medical", None)
    # medicine / tablet questions (PRIMARY medical)
    medicine_keywords = [
        "tablet", "capsule", "medicine", "drug", "syrup",
        "injection", "b-complex", "paracetamol", "crocin",
        "azithromycin", "vitamin"
        ]
    if any(m in t for m in medicine_keywords):
        return ("medical", None)

    # follow-ups (ONLY contextual)
    followups = [
        "side effects", "sideeffect", "dose", "dosage",
        "how many", "continue", "more", "why", "safe",
        "pregnant", "children", "elderly","how to", "recover", "cure", "overcome","get rid", "treat this", "fix this"
    ]
    for f in followups:
        if f in t:
            return ("followup", None)
    

    # medical questions
    medical_words = [
    "symptom", "symptoms",
    "pain", "body pain", "chest pain", "stomach pain",
    "fever", "cold", "cough", "infection",
    "disease", "treatment", "cause", "diagnosis",
    "medicine", "tablet", "drug", "rash",
    "diarrhea", "vomiting", "nausea",
    "asthma", "diabetes", "heart",
    "skin", "typhoid", "bp",
    "blood pressure", "hypertension",
    "sugar", "headache", "migraine",
    "mental", "stress", "anxiety",
    "weakness", "fatigue",
    "exercise", "diet", "health", "wellness"
    ]
    
    symptom_patterns = [
    "i have",
    "i am having",
    "suffering from",
    "feeling",
    "my",
    "having"
    ]
    
    if any(p in t for p in symptom_patterns):
        return ("medical", None)
    
    if any(w in t for w in medical_words) or t.startswith(("what is", "explain")):
        return ("medical", None)

        # identity / chatbot info
    identity_phrases = [
        "who created you",
        "who made you",
        "who built you",
        "who are you",
        "what are you",
        "are you a doctor",
        "are you human",
        "your creator",
        "your owner"
    ]

    for p in identity_phrases:
        if p in t:
            return ("identity", None)

    return ("other", None)





# ---------------- RAG query (fast path) ----------------
def call_rag_with_retry(text, retries=3, delay=1.0, sender_id="whatsapp"):
    global conversation_topic, last_user_query

    intent, lang = detect_intent(text)
    
    if intent == "identity":
        t = text.lower()

        if "who are you" in t or "what are you" in t or "who created you" in t or "are you a doctor" in t:
            base_answer = ""
            
            if "who created you" in t:
                base_answer = (
                    "I was created by Amruth Gowda as an AI-powered medical chatbot."
                    )
                
            elif "who are you" in t or "what are you" in t:
                base_answer = (
                    "I am an AI-based medical chatbot designed to provide general health information."
                    )
            elif "are you a doctor" in t:
                base_answer = (
                    "No, I am not a doctor. I provide general medical information only."
                    )
                # ✅ If user asked to translate identity answer
            if "answer in" in t or "translate" in t:
                lang = t.replace("answer in", "").replace("translate to", "").strip()
                translated_prompt = f"Translate this to {lang}:\n{base_answer}"
                return call_github_chat_model(
                    system_message="You are a translator.",
                    user_message=translated_prompt,
                    model=CHAT_MODEL,
                    temperature=0.3,
                    max_tokens=300,
                )
            return base_answer
    # --------------------------  
    # 1️⃣ GREETING  
    # --------------------------
    if intent == "greeting":
        return "👋 Hello! How can I assist you today?"

    # --------------------------  
    # 2️⃣ TRANSLATION REQUEST  
    # User says: "Answer in Kannada", "Kannada alli heli", "Translate to Hindi"
    # --------------------------
    if intent == "translate":
        prev_q = last_user_query.get(sender_id)
        if not prev_q:
            return "Please ask a medical question first."

        translated_query = f"Answer this in {lang}:\n{prev_q}"
        return call_rag_with_retry(
            translated_query,
            retries=retries,
            delay=delay,
            sender_id=sender_id
        )

    # --------------------------  
    # 3️⃣ FOLLOW-UP QUESTION  
    # User says: "What is cause for that?", "Symptoms for that?", "Why does it happen?"
    # --------------------------
    if intent == "followup":
        topic = conversation_topic.get(sender_id)

        if not topic:
            return "Please ask a medical question first."

        followup_query = (
            f"The user previously asked about '{topic}'. "
            f"This is a follow-up question: {text}. "
            f"Provide a detailed medical explanation."
        )

        last_user_query[sender_id] = followup_query
        text = followup_query  # continue with RAG using rewritten text

    # --------------------------  
    # 4️⃣ MEDICAL MAIN QUESTION  
    # --------------------------
    if intent == "medical":
        conversation_topic[sender_id] = text         # Store topic for future follow-ups
        last_user_query[sender_id] = text            # Store for translation
        
    # 🧠 Contextual follow-up fallback
    if intent == "other":
        topic = conversation_topic.get(sender_id)
        if topic:
            followup_query = (
                f"The user previously asked about '{topic}'. "
                f"Now they are asking: {text}. "
                f"Explain treatment, recovery, and prevention."
                )
            last_user_query[sender_id] = followup_query
            text = followup_query
            intent = "followup"
        else:
            return (
                "I'm here to help with medical questions.\n"
                "Please describe your medical condition or symptoms.\n"
                "For example: fever, BP, headache, cough, diabetes."
            )

    # --------------------------
    # 6️⃣ NORMAL RAG PROCESS  
    # --------------------------
    global _rag_initialized, _rag_init_error, rag_retriever

    if not _rag_initialized:
        if _rag_init_error:
            return f"⚠ RAG initialization failed: {_rag_init_error}"
        return "⚠ RAG is loading. Try again."

    # RAG Document Retrieval
    docs = []
    try:
        if hasattr(rag_retriever, "get_relevant_documents"):
            docs = rag_retriever.get_relevant_documents(text)
        elif hasattr(rag_retriever, "retrieve"):
            docs = rag_retriever.retrieve(text)
        else:
            docs = rag_retriever(text)
    except Exception as e:
        logger.exception("Retriever error: %s", e)

    # Build context
    context_chunks = []
    for d in docs[:RAG_K]:
        c = getattr(d, "page_content", "") or getattr(d, "content", "")
        if c:
            context_chunks.append(c[:800])

    context_text = "\n\n---\n\n".join(context_chunks)

    if context_text:
        final_prompt = (
            f"Context:\n{context_text}\n\n"
            f"User Question:\n{text}\n\n"
            f"Provide a clear, medically accurate answer."
        )
    else:
        final_prompt = text

    # --------------------------
    # 7️⃣ LLM Call with Retry  
    # --------------------------
    cur_delay = delay
    for attempt in range(1, retries + 1):
        try:
            ans = call_github_chat_model(
                system_message=system_prompt,
                user_message=final_prompt,
                model=CHAT_MODEL,
                temperature=CHAT_TEMPERATURE,
                max_tokens=CHAT_MAX_TOKENS,
                timeout=25
                )
            # ✅ HARD SAFETY: never return empty or invalid text
            if not ans or not str(ans).strip():
                logger.warning("LLM returned empty response on attempt %d", attempt)
                continue
            return str(ans).strip()
        except Exception as e:
            logger.warning("RAG attempt %d failed: %s", attempt, e)
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(cur_delay)
                cur_delay *= 2
                continue
            return "⚠ Error generating response. Please try again."
        #✅ FINAL FALLBACK — NEVER EMPTY
        return "⚠ I could not generate a response. Please rephrase your question."




# ---------------- OCR / PDF utilities ----------------
def preprocess_image_for_ocr(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def extract_text_from_image(image_path: str):
    try:
        processed = preprocess_image_for_ocr(image_path)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, config=config)
        return text.strip()
    except Exception:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            return text.strip()
        except Exception as e:
            logger.exception("OCR fallback failed: %s", e)
            return ""

def extract_text_from_pdf(file_path: str):
    try:
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        logger.exception("PDF text extraction failed: %s", e)
        return ""

def extract_text_from_any(path: str) -> str:
    try:
        low = path.lower()
        if low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
            return extract_text_from_image(path)
        if low.endswith(".pdf"):
            return extract_text_from_pdf(path)
        try:
            return extract_text_from_image(path)
        except Exception:
            return ""
    except Exception as e:
        logger.exception("extract_text_from_any error: %s", e)
        return ""

# ---------------- Email helper ----------------
def send_email(to_email, subject, message):
    try:
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            logger.warning("EMAIL_ADDRESS or EMAIL_PASSWORD not set.")
            return False
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_ADDRESS, [to_email], msg.as_string())
        return True
    except Exception as e:
        logger.exception("Email send error: %s", e)
        return False

# ---------------- chats.json helpers (with lock) ----------------
chats_lock = threading.Lock()

def _ensure_chats_file():
    if not os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_chats():
    with chats_lock:
        _ensure_chats_file()
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Corrupt chats.json; resetting")
            return []

def save_chats(data):
    with chats_lock:
        tmp = CHATS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CHATS_FILE)

def find_chat(chats, chat_id):
    for c in chats:
        if c.get("id") == chat_id:
            return c
    return None


# ---------------- Web UI routes (unchanged) ----------------
@app.route("/",methods=["GET", "POST"])
def index():
    try:
        return render_template("chat.html")
    except Exception:
        return "Medical chatbot is running."

from flask import abort

@app.route('/favicon.ico')
def favicon():
    return abort(204)


@app.route("/get", methods=["POST"])
def chat_web_ui():
    try:
        msg = request.form.get("msg", "").strip()
        image = request.files.get("image")
        extracted_text = None

        if image:
            filename = secure_filename(image.filename)
            savepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
            image.save(savepath)
            try:
                extracted_text = extract_text_from_image(savepath)
                logger.info("OCR preview: %s", extracted_text[:200])
            except Exception as e:
                logger.exception("OCR error (web UI): %s", e)

        final_input = msg or ""
        if extracted_text:
            final_input = (final_input + "\n\nExtracted from image:\n" + extracted_text) if final_input else extracted_text

        if not final_input.strip():
            return "⚠ Please send a message or upload an image."

        answer = call_rag_with_retry(final_input)
        return answer
    except Exception as e:
        logger.exception("/get error: %s", e)
        return "⚠ Server error."
    
# ---------------- Text-to-Speech (TTS) ----------------
@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        lang = detect_tts_lang(text)

        tts = gTTS(text=text, lang=lang)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)

        return send_file(
            tmp.name,
            mimetype="audio/mpeg",
            as_attachment=False
        )
    except Exception as e:
        logger.exception("TTS error: %s", e)
        return jsonify({"error": "TTS failed"}), 500


# ---------------- API: Chats management (unchanged) ----------------
@app.route("/api/chats", methods=["GET", "POST"])
def api_chats():
    chats = load_chats()
    if request.method == "GET":
        view = [{"id": c["id"], "title": c.get("title", "New chat"), "created_at": c.get("created_at")} for c in chats]
        return jsonify(view)
    new_chat = {
        "id": str(uuid.uuid4()),
        "title": request.json.get("title", "New chat") if request.is_json else "New chat",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "messages": []
    }
    chats.insert(0, new_chat)
    save_chats(chats)
    return jsonify(new_chat)

@app.route("/api/chats/<chat_id>", methods=["GET", "DELETE"])
def api_chat(chat_id):
    chats = load_chats()
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    if request.method == "DELETE":
        chats = [c for c in chats if c.get("id") != chat_id]
        save_chats(chats)
        return jsonify({"ok": True})
    return jsonify(chat)

# ---------------- Add message (non-streaming fallback) ----------------
@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def api_add_message(chat_id):
    try:
        chats = load_chats()
        chat = find_chat(chats, chat_id)

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        text = request.form.get("msg", "").strip()
        file = request.files.get("image")

        if not text and not file:
            return jsonify({"error": "Empty message"}), 400

        local_image_path = None
        image_url = None

        if file:
            filename = secure_filename(
                f"{chat_id}_{int(time.time())}_{file.filename}"
            )
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            local_image_path = filepath
            image_url = f"/uploads/{filename}"
            logger.info("Saved uploaded image for chat %s -> %s", chat_id, filepath)

        # 1️⃣ Append user message FIRST
        user_msg = {
            "id": str(uuid.uuid4()),
            "type": "user",
            "text": text,
            "image_url": image_url,
            "time": datetime.datetime.utcnow().isoformat()
        }
        chat["messages"].append(user_msg)

        # 2️⃣ Set title on first message
        if len(chat["messages"]) == 1 and text:
            chat["title"] = text[:35] + "..." if len(text) > 35 else text

        # 3️⃣ Generate response USING HISTORY
        answer = process_message_for_chat_history(text,local_image_path)

        # 4️⃣ Append bot message
        bot_msg = {
            "id": str(uuid.uuid4()),
            "type": "bot",
            "text": answer,
            "image_url": None,
            "time": datetime.datetime.utcnow().isoformat()
        }
        chat["messages"].append(bot_msg)

        save_chats(chats)
        return jsonify({"chat": chat})

    except Exception as e:
        logger.exception("api_add_message error")
        return jsonify({"error": "Internal server error"}), 500

# ---------------- Helper used by both endpoints ----------------
def process_message_for_chat_history(text, image_path=None):
    extracted = ""
    if image_path:
        try:
            extracted = extract_text_from_image(image_path)
        except Exception as e:
            logger.exception("OCR error in process_message_for_chat_history: %s", e)
    final_input = text or ""
    if extracted:
        final_input = (final_input + "\n\nExtracted from image:\n" + extracted) if final_input else extracted
    return call_rag_with_retry(final_input)

# ---------------- Serve uploaded images ----------------
@app.route("/uploads/<path:filename>")
def serve_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------------- Streaming endpoint (SSE) - updated to accept files & base64 ----------------
@app.route("/api/chats/<chat_id>/stream", methods=["POST"])
def api_chat_stream(chat_id):
    try:
        logger.info("Incoming stream request: content_type=%s", request.content_type)

        chats = load_chats()
        chat = find_chat(chats, chat_id)
        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        extracted_text = None
        saved_local_image = None

        uploaded_file = None
        if request.files:
            uploaded_file = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()), None)

        if uploaded_file:
            filename = secure_filename(f"{chat_id}_{int(time.time())}_{uploaded_file.filename}")
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(filepath)
            saved_local_image = filepath
            logger.info("Saved stream-uploaded image: %s", filepath)
            try:
                extracted_text = extract_text_from_image(filepath)
                logger.info("OCR (stream) preview: %s", (extracted_text or "")[:200])
            except Exception as e:
                logger.exception("OCR error for uploaded file: %s", e)

        text = ""
        if request.is_json:
            payload = request.get_json() or {}
            text = (payload.get("message") or payload.get("msg") or payload.get("text") or "").strip()
            image_b64 = payload.get("image_base64") or payload.get("imageBase64")
            if image_b64 and not extracted_text:
                import base64, re
                m = re.match(r"data:(image/\w+);base64,(.*)", image_b64)
                if m:
                    b64 = m.group(2)
                    ext = m.group(1).split('/')[1]
                else:
                    b64 = image_b64
                    ext = "png"
                try:
                    raw = base64.b64decode(b64)
                    filename = secure_filename(f"{chat_id}_{uuid.uuid4().hex}.{ext}")
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    with open(filepath, "wb") as f:
                        f.write(raw)
                    saved_local_image = filepath
                    logger.info("Saved JSON-base64 image: %s", filepath)
                    extracted_text = extract_text_from_image(filepath)
                    logger.info("OCR (stream json base64) preview: %s", (extracted_text or "")[:200])
                except Exception as e:
                    logger.exception("Failed to decode/save OCR image from JSON: %s", e)

        if not request.is_json:
            text = (request.form.get("msg", "") or request.values.get("message", "")).strip()

        final_input = (text or "")
        if extracted_text:
            final_input = (final_input + "\n\nExtracted from image:\n" + extracted_text) if final_input else extracted_text

        if not final_input.strip():
            return jsonify({"error": "Message or image required"}), 400

        user_msg = {
            "id": str(uuid.uuid4()),
            "type": "user",
            "text": text if text else (extracted_text or ""),
            "image_url": (f"/uploads/{os.path.basename(saved_local_image)}" if saved_local_image else None),
            "time": datetime.datetime.utcnow().isoformat()
        }
        chat["messages"].append(user_msg)
        if len(chat["messages"]) == 1 and user_msg["text"]:
            chat["title"] = user_msg["text"][:35] + ("..." if len(user_msg["text"]) > 35 else "")
        save_chats(chats)

        # Generate the answer (blocking call) and save to chat before streaming
        answer = call_rag_with_retry(final_input or text)

        bot_msg = {
            "id": str(uuid.uuid4()),
            "type": "bot",
            "text": answer,
            "image_url": None,
            "time": datetime.datetime.utcnow().isoformat()
        }

        chats_after = load_chats()
        existing_chat = find_chat(chats_after, chat_id)
        if existing_chat is not None:
            existing_chat["messages"].append(bot_msg)
            save_chats(chats_after)
        else:
            chat["messages"].append(bot_msg)
            save_chats(chats)

        def generate():
            try:
                for line in answer.split("\n"):
                    yield f"data: {line}\n\n"
                    time.sleep(0.01)
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("Stream generator exception: %s", e)
                yield f"data: ⚠ Streaming error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return Response(generate(), mimetype="text/event-stream", headers=headers)

    except Exception as e:
        logger.exception("/stream error: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------------- WhatsApp webhook (async ack + background reply) ----------------
connected_users = set()
_twilio_client = None
# increase workers for concurrency
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WEBHOOK_WORKERS", "20")))

def get_twilio_client():
    global _twilio_client
    if _twilio_client is None:
        if not TWILIO_SID or not TWILIO_AUTH_TOKEN:
            logger.warning("Twilio SID/Auth not configured")
            return None
        _twilio_client = TwilioClient(TWILIO_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client

# Safe send wrapper to avoid Twilio 21619 errors (empty body) and transient failures
def safe_send_message(client, to, from_, body, max_retries=3):
    if not body or not isinstance(body, str) or not body.strip():
        body = "(response is being prepared)"  # fallback safe text

    attempt = 0
    while attempt < max_retries:
        try:
            msg = client.messages.create(body=body, from_=from_, to=to)
            return msg
        except TwilioRestException as e:
            attempt += 1
            logger.warning("Twilio send attempt %d failed: %s", attempt, e)
            # for permanent errors, don't retry
            if e.code and 20000 <= int(e.code) < 30000:
                logger.error("Permanent Twilio error %s: %s", e.code, e.msg)
                raise
            time.sleep(0.5 * attempt)
        except Exception as e:
            attempt += 1
            logger.exception("Unexpected Twilio send error attempt %d: %s", attempt, e)
            time.sleep(0.5 * attempt)
    raise RuntimeError("Failed to send Twilio message after retries")

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    twilio_resp = MessagingResponse()

    try:
        sender = request.values.get("From", "")
        incoming_msg = (request.values.get("Body", "") or "").strip()

        try:
            num_media = int(request.values.get("NumMedia", 0))
        except Exception:
            num_media = 0

        media_urls = []
        media_content_types = []

        for i in range(num_media):
            media_urls.append(request.values.get(f"MediaUrl{i}"))
            media_content_types.append(request.values.get(f"MediaContentType{i}"))

        logger.info("WhatsApp from %s: '%s' media=%d", sender, incoming_msg[:120], num_media)
        
        first_msg = incoming_msg.lower().strip()
        greetings = ["hi", "hello", "hey", "hii", "hiii", "hola"]

        if sender not in connected_users and first_msg in greetings:
            connected_users.add(sender)
            welcome = (
                "👋 Welcome to Medical Chatbot!\n\n"
                "Ask health questions or send medical images/PDFs."
            )
            twilio_resp.message(welcome)
            return Response(str(twilio_resp), content_type="application/xml; charset=utf-8")

        # immediate empty TwiML ack
        xml = str(twilio_resp)
        resp_immediate = Response(xml, content_type="application/xml; charset=utf-8")

        def background_process_and_reply(sender_local, incoming_msg_local, media_urls_local, media_content_types_local):
            try:
                client = get_twilio_client()
                twilio_from = TWILIO_WHATSAPP_NUMBER

                if not client or not twilio_from:
                    logger.error("Twilio not configured.")
                    return

                # download media
                saved_files = []
                extracted_texts = []

                for idx, url in enumerate(media_urls_local):
                    if not url:
                        continue
                    try:
                        r = requests_session.get(url, auth=(TWILIO_SID, TWILIO_AUTH_TOKEN), timeout=15)
                        if r.status_code == 200:
                            ctype = media_content_types_local[idx] if idx < len(media_content_types_local) else None
                            if ctype:
                                ext = ctype.split('/')[-1]
                            else:
                                try:
                                    ext = url.split('.')[-1].split('?')[0][:6]
                                except Exception:
                                    ext = "bin"

                            filename = secure_filename(f"{sender_local.replace(':','')}_{uuid.uuid4().hex}_{idx}.{ext}")
                            fp = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                            with open(fp, "wb") as f:
                                f.write(r.content)
                            saved_files.append(fp)
                            logger.info("Saved media: %s", fp)
                        else:
                            logger.warning("Failed to download media %s status=%d", url, r.status_code)
                    except Exception as e:
                        logger.exception("Exception downloading media in background: %s", e)

                # OCR
                for fp in saved_files:
                    try:
                        txt = extract_text_from_any(fp)
                        if txt and txt.strip():
                            extracted_texts.append(txt.strip())
                    except Exception as e:
                        logger.exception("Background OCR error: %s", e)

                # prepare input
                body_for_rag = " ".join(extracted_texts).strip() if extracted_texts else incoming_msg_local

                if not body_for_rag:
                    reply_text = "⚠ I couldn't read any text from the message."
                else:
                    reply_text = call_rag_with_retry(body_for_rag, sender_id=sender_local)

                # send final message (single final reply)
                try:
                    safe_send_message(client, sender_local, twilio_from, reply_text)
                    logger.info("Final reply sent to %s", sender_local)
                except Exception as e:
                    logger.exception("Failed to send final reply: %s", e)

            except Exception:
                logger.exception("Error in background_process_and_reply")

        executor.submit(background_process_and_reply, sender, incoming_msg, media_urls, media_content_types)
        return resp_immediate

    except Exception as e:
        logger.exception("WhatsApp webhook error: %s", e)
        twilio_resp.message("⚠ Server error. Please try again later.")
        return Response(str(twilio_resp), content_type="application/xml; charset=utf-8")

# ---------------- run ----------------
if __name__ == "__main__":
    # if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    #     try:
    #         initialize_rag_once()
    #     except Exception:
    #         logger.exception("Startup RAG init failed (continuing without it)")
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True,use_reloader=False)
