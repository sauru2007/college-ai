import os
import base64

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from groq import Groq
from openai import OpenAI

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

print(">>> Flask app started <<<")


# ---------------- ROOT ROUTE (FRONTEND) ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- MODELS ----------------
TEXT_MODEL = "llama-3.1-8b-instant"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- RAG CONFIG ----------------
CHUNK_SIZE = 1000
OVERLAP = 200


def split_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks


def get_relevant_chunks(query, chunks, k=3):
    if not chunks:
        return ""

    docs = chunks + [query]
    vectors = TfidfVectorizer().fit_transform(docs).toarray()
    scores = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()
    top = scores.argsort()[-k:][::-1]
    return "\n\n".join(chunks[i] for i in top)


# ---------------- AI PERSONALITIES ----------------
def get_personality(mode):
    if mode == "Friend":
        return "You are a friendly college buddy. Explain simply.", 0.7
    if mode == "Mentor":
        return "You are a senior mentor. Be structured and guiding.", 0.6
    return "You are a strict academic assistant.", 0.4


# ---------------- FILE UPLOAD ----------------
@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    content_type = file.content_type or ""

    # ---- PDF ----
    if content_type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        return jsonify({
            "type": "pdf",
            "chunks": split_text(text),
            "message": "PDF processed successfully"
        })

    # ---- IMAGE ----
    if content_type.startswith("image/"):
        encoded = base64.b64encode(file.read()).decode("utf-8")
        return jsonify({
            "type": "image",
            "content": encoded,
            "message": "Image received successfully"
        })

    return jsonify({"error": "Unsupported file type"}), 400


# ---------------- CHAT ----------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)

    message = data.get("message", "")
    history = data.get("history", [])
    file_ctx = data.get("fileContext")
    exam_mode = data.get("examMode", "General")
    assistant_mode = data.get("assistantMode", "Friend")

    personality, temperature = get_personality(assistant_mode)

    # -------- IMAGE → OPENAI (VISION) --------
    if file_ctx and file_ctx.get("type") == "image":
        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"response": "Image understanding is not enabled."})

        img_base64 = file_ctx["content"]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message or "Analyze this academically"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )

        return jsonify({
            "response": response.choices[0].message.content
        })

    # -------- TEXT / PDF → GROQ --------
    system_prompt = f"""
{personality}

Rules:
- Answer college academic questions only
- Start response with **[Subject: Name]**
- Exam mode: {exam_mode}
"""

    if file_ctx and file_ctx.get("type") == "pdf":
        context = get_relevant_chunks(message, file_ctx.get("chunks", []))
        system_prompt += f"\nREFERENCE MATERIAL:\n{context}\n"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1024
    )

    return jsonify({
        "response": response.choices[0].message.content
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
