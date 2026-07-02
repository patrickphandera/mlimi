"""
Flask backend for the Malawi Agriculture Chatbot.

Endpoints:
- GET  /api/health           -> simple health check
- POST /api/chat             -> non-streaming chat completion
- POST /api/chat/stream      -> Server-Sent Events streaming chat completion

Request body for /api/chat and /api/chat/stream:
{
  "messages": [ { "role": "user"|"assistant", "content": "..." }, ... ]
}
"""

import json
import os
import time

import requests
from dotenv import load_dotenv
from flask import Flask, Response, g, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS

from admin import admin_bp
from auth import auth_bp, get_db, init_db, require_auth, seed_defaults

load_dotenv()

ENGLISH_MODEL_URL = os.environ.get(
    "ENGLISH_MODEL_URL",
    "https://revenge-kelp-reword.ngrok-free.dev/chat",
)
ENGLISH_MODEL_NAME = os.environ.get("ENGLISH_MODEL_NAME", "english-agriculture-model")
MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", "120"))

# Path to the built frontend (`npm run build` output). In single-service
# Render deploys, Flask serves this so the API and SPA share an origin.
FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")),
)

BASE_SYSTEM_PROMPT = (
    "You are Mlimi AI, a friendly and knowledgeable agriculture expert helping "
    "Malawian farmers. You understand local farming conditions, smallholder "
    "realities, soils, seasons, common crops (maize, groundnuts, soybean, "
    "tobacco, cassava, sweet potato), pests, and post-harvest handling in "
    "Malawi. Provide specific, practical, and actionable advice. Prefer simple "
    "language. When useful, mention local context (e.g., rainy season timing, "
    "ADMARC, NPK/CAN/Urea fertiliser, government extension services). If a "
    "question is outside agriculture, gently steer the conversation back to "
    "farming topics. Keep replies concise unless the farmer asks for detail."
)

CHICHEWA_LANGUAGE_RULE = (
    " LANGUAGE: You MUST respond exclusively in Chichewa (Chinyanja) for every "
    "message, regardless of the language the user writes in. Do not reply in "
    "English. The only exceptions are: (a) proper nouns (names of places, "
    "people, brands, products), (b) scientific or technical terms that have no "
    "established Chichewa equivalent (in that case, use the English term as-is "
    "without translation, and you may briefly explain the meaning in Chichewa "
    "in parentheses). Numbers, units, and chemical formulas (NPK, CAN, Urea) "
    "may stay in their standard form. Never apologise in English. If you don't "
    "know a Chichewa word, write the concept in Chichewa using description "
    "rather than switching to English."
)

ENGLISH_LANGUAGE_RULE = (
    " LANGUAGE: You MUST respond exclusively in English for every message, "
    "regardless of the language the user writes in. Do not reply in Chichewa "
    "or any other language. The only exceptions are: (a) proper nouns (names "
    "of places, people, brands, products), (b) widely-used local terms with no "
    "common English equivalent (e.g. nsima, dimba) — keep them as-is and you "
    "may briefly explain in English in parentheses. Numbers, units, and "
    "chemical formulas (NPK, CAN, Urea) may stay in their standard form. Never "
    "switch languages mid-reply."
)


def system_prompt_for(language: str | None) -> str:
    code = (language or "en").lower()
    if code in {"ny", "chichewa", "chi", "nyanja"}:
        return BASE_SYSTEM_PROMPT + CHICHEWA_LANGUAGE_RULE
    return BASE_SYSTEM_PROMPT + ENGLISH_LANGUAGE_RULE

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
init_db()
seed_defaults()


@app.get("/")
def _serve_index():
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.get("/<path:path>")
def _serve_spa(path: str):
    # API routes are registered above and won't reach here. Serve a real file
    # from the built frontend if it exists, otherwise fall back to index.html
    # so React Router can handle deep links (/chat, /admin, /login, ...).
    candidate = os.path.join(FRONTEND_DIST, path)
    if os.path.isfile(candidate):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")

def call_english_model(messages) -> str:
    if not ENGLISH_MODEL_URL:
        raise RuntimeError("ENGLISH_MODEL_URL is not configured")

    response = requests.post(
        ENGLISH_MODEL_URL,
        headers={"Content-Type": "application/json"},
        json={"messages": messages},
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    reply = data.get("reply") or data.get("response")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError(f"Model returned an unexpected response: {data}")
    return reply.strip()


def build_messages(history, language=None):
    cleaned = [{"role": "system", "content": system_prompt_for(language)}]
    for m in history or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def log_chat_event(user_id, language, prompt_tokens, completion_tokens, total_tokens, model):
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO chat_events
                  (user_id, language, prompt_tokens, completion_tokens, total_tokens, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    (language or "en").lower(),
                    int(prompt_tokens or 0),
                    int(completion_tokens or 0),
                    int(total_tokens or 0),
                    model,
                    int(time.time()),
                ),
            )
    except Exception:
        # logging must never break the chat response
        pass


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": ENGLISH_MODEL_NAME, "english_model_url": ENGLISH_MODEL_URL})


@app.post("/api/chat")
@require_auth
def chat():
    body = request.get_json(silent=True) or {}
    language = body.get("language")
    messages = build_messages(body.get("messages"), language)
    if len(messages) <= 1:
        return jsonify({"error": "messages is required"}), 400

    try:
        reply = call_english_model(messages)
        log_chat_event(
            user_id=g.user["id"],
            language=language,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model=ENGLISH_MODEL_NAME,
        )
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/chat/stream")
@require_auth
def chat_stream():
    body = request.get_json(silent=True) or {}
    language = body.get("language")
    messages = build_messages(body.get("messages"), language)
    if len(messages) <= 1:
        return jsonify({"error": "messages is required"}), 400

    user_id = g.user["id"]

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    @stream_with_context
    def generate():
        prompt_tokens = completion_tokens = total_tokens = 0
        try:
            reply = call_english_model(messages)
            yield sse("token", {"text": reply})
            yield sse("done", {"ok": True})
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
        finally:
            log_chat_event(
                user_id=user_id,
                language=language,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=ENGLISH_MODEL_NAME,
            )

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=True)
