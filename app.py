import os
import io
import base64
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from PIL import Image

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, session, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import openai
import stripe

# -------------------- Setup --------------------
load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-this")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI", "sqlite:///database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# OpenRouter (OpenAI-compatible)
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)

# -------------------- DB Models --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    subscription_id = db.Column(db.String(100), nullable=True)

    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10), default="")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship("Chat", backref="user", lazy=True, cascade="all, delete-orphan")


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    title = db.Column(db.String(200), default="New Chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = db.relationship("Message", backref="chat", lazy=True, cascade="all, delete-orphan")


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey("chat.id"), nullable=False, index=True)

    role = db.Column(db.String(20), nullable=False)  # user / assistant / system
    content = db.Column(db.Text, nullable=False)

    model = db.Column(db.String(80), nullable=True)

    has_image = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(800), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


# -------------------- Model Config --------------------
FREE_MODELS = {
    "gpt-3.5-turbo": {
        "path": "openai/gpt-3.5-turbo",
        "name": "GPT-3.5 Turbo",
        "vision": False,
        "limit": None,
    },
    "claude-3-haiku": {
        "path": "anthropic/claude-3-haiku",
        "name": "Claude 3 Haiku (Free)",
        "vision": True,   # OpenRouter supports vision for many Claude models
        "limit": None,
    },
    "gemini-flash": {
        "path": "google/gemini-flash-1.5",
        "name": "Gemini Flash 1.5",
        "vision": True,
        "limit": None,
    },
    "deepseek-chat": {
        "path": "deepseek/deepseek-chat",
        "name": "DeepSeek Chat",
        "vision": False,
        "limit": 50,  # 50/day for free users
    },
}

PREMIUM_MODELS = {
    "gpt-4o": {
        "path": "openai/gpt-4o",
        "name": "GPT-4o",
        "vision": True,
    },
    "gpt-4o-mini": {
        "path": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "vision": True,
    },
    "claude-3.5-sonnet": {
        "path": "anthropic/claude-3.5-sonnet",
        "name": "Claude 3.5 Sonnet",
        "vision": True,
    },
    "claude-3-opus": {
        "path": "anthropic/claude-3-opus",
        "name": "Claude 3 Opus",
        "vision": True,
    },
    "gemini-pro": {
        "path": "google/gemini-pro-1.5",
        "name": "Gemini Pro 1.5",
        "vision": True,
    },
    "deepseek-r1": {
        "path": "deepseek/deepseek-r1",
        "name": "DeepSeek R1",
        "vision": False,
    },
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "doc", "docx"}


# -------------------- Helpers --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compress_image(image_path, max_size=(1024, 1024), quality=85):
    """Best-effort compression; safe to fail."""
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=quality)
    except Exception:
        pass


def encode_image(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def generate_chat_title(first_message: str) -> str:
    s = (first_message or "").strip()
    if not s:
        return "New Chat"
    title = s[:50]
    if len(s) > 50:
        title += "..."
    return title


def check_deepseek_limit(user: User) -> bool:
    """50/day for free users."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    return user.deepseek_count < 50


def migrate_database_sqlite():
    """
    Adds missing columns to old sqlite dbs if they exist.
    Safe no-op if DB doesn't exist.
    """
    try:
        db_path = "database.db"
        if not os.path.exists(db_path):
            return

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("PRAGMA table_info(user)")
        user_cols = [c[1] for c in cur.fetchall()]
        if "deepseek_count" not in user_cols:
            cur.execute("ALTER TABLE user ADD COLUMN deepseek_count INTEGER DEFAULT 0")
        if "deepseek_date" not in user_cols:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            cur.execute(f'ALTER TABLE user ADD COLUMN deepseek_date TEXT DEFAULT "{today}"')
        if "created_at" not in user_cols:
            cur.execute("ALTER TABLE user ADD COLUMN created_at TIMESTAMP")

        cur.execute("PRAGMA table_info(message)")
        msg_cols = [c[1] for c in cur.fetchall()]
        if "has_image" not in msg_cols:
            cur.execute("ALTER TABLE message ADD COLUMN has_image BOOLEAN DEFAULT 0")
        if "image_path" not in msg_cols:
            cur.execute("ALTER TABLE message ADD COLUMN image_path TEXT")

        conn.commit()
        conn.close()
    except Exception:
        pass


def get_model_config(model_key: str):
    if model_key in FREE_MODELS:
        return FREE_MODELS[model_key], False
    if model_key in PREMIUM_MODELS:
        return PREMIUM_MODELS[model_key], True
    return None, None


def normalize_uploaded_path(p: str | None) -> str | None:
    """
    Accepts:
    - absolute filepath returned by /upload
    - "uploads/xxx"
    - just filename
    Returns absolute filesystem path or None.
    """
    if not p:
        return None

    p = p.strip()
    if not p:
        return None

    # If frontend sends url like /uploads/filename
    if p.startswith("/uploads/"):
        p = p.replace("/uploads/", "", 1)

    # If p is absolute and exists
    if os.path.isabs(p) and os.path.exists(p):
        return p

    # If p starts with uploads/
    if p.startswith(app.config["UPLOAD_FOLDER"] + os.sep):
        abs_path = os.path.abspath(p)
        if os.path.exists(abs_path):
            return abs_path

    # Treat as filename inside uploads
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(p))
    if os.path.exists(abs_path):
        return abs_path

    return None


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/demo-chat", methods=["POST"])
def demo_chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        resp = openrouter_client.chat.completions.create(
            model=FREE_MODELS["gpt-3.5-turbo"]["path"],
            messages=[{"role": "user", "content": user_message}],
        )
        return jsonify({
            "response": resp.choices[0].message.content,
            "demo": True,
            "model": "GPT-3.5 Turbo (Demo)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------- Auth --------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        email = (data.get("email") or "").strip().lower()
        name = (data.get("name") or "").strip()
        password = data.get("password") or ""

        if not email or not name or not password:
            msg = "All fields are required"
            return jsonify({"error": msg}), 400 if request.is_json else render_template("signup.html", error=msg)

        if User.query.filter_by(email=email).first():
            msg = "Email already exists"
            return jsonify({"error": msg}), 400 if request.is_json else render_template("signup.html", error=msg)

        new_user = User(
            email=email,
            name=name,
            password=generate_password_hash(password),
            deepseek_date=datetime.utcnow().strftime("%Y-%m-%d")
        )
        db.session.add(new_user)
        db.session.commit()

        if request.is_json:
            return jsonify({"success": True, "redirect": "/login"})
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session["selected_model"] = session.get("selected_model", "gemini-flash")
            if request.is_json:
                return jsonify({"success": True, "redirect": "/dashboard"})
            return redirect(url_for("dashboard"))

        if request.is_json:
            return jsonify({"error": "Invalid credentials"}), 401
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    # dashboard.html expects models + user
    if current_user.is_premium:
        available = {**FREE_MODELS, **PREMIUM_MODELS}
    else:
        available = FREE_MODELS
    return render_template("dashboard.html", user=current_user, models=available)


@app.route("/set-model", methods=["POST"])
@login_required
def set_model():
    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "Model not specified"}), 400
    session["selected_model"] = model
    return jsonify({"success": True})


# -------- Upload --------
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(f.filename)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user.id}_{ts}_{filename}"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        compress_image(filepath)

    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "url": url_for("uploaded_file", filename=filename)
    })


# -------- Chat sidebar APIs (your dashboard JS needs these) --------
@app.route("/get-chats", methods=["GET"])
@login_required
def get_chats():
    chats = (
        Chat.query.filter_by(user_id=current_user.id)
        .order_by(Chat.updated_at.desc())
        .limit(200)
        .all()
    )
    return jsonify([
        {"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()}
        for c in chats
    ])


@app.route("/get-chat/<int:chat_id>", methods=["GET"])
@login_required
def get_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    msgs = (
        Message.query.filter_by(chat_id=chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return jsonify({
        "title": chat.title,
        "messages": [{
            "role": m.role,
            "content": m.content,
            "model": m.model,
            "has_image": m.has_image,
            "image_path": m.image_path,
            "created_at": m.created_at.isoformat()
        } for m in msgs]
    })


@app.route("/chat/new", methods=["POST"])
@login_required
def new_chat():
    c = Chat(user_id=current_user.id, title="New Chat")
    db.session.add(c)
    db.session.commit()
    return jsonify({"success": True, "chat_id": c.id, "title": c.title})


@app.route("/chat/<int:chat_id>/rename", methods=["POST"])
@login_required
def rename_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title cannot be empty"}), 400

    chat.title = title
    db.session.commit()
    return jsonify({"success": True})


@app.route("/delete-chat/<int:chat_id>", methods=["DELETE"])
@login_required
def delete_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    # best-effort delete uploaded files referenced by messages
    for m in chat.messages:
        if m.has_image and m.image_path:
            p = normalize_uploaded_path(m.image_path)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    db.session.delete(chat)
    db.session.commit()
    return jsonify({"success": True})


# (Optional compatibility endpoint; some older JS uses this)
@app.route("/chat/<int:chat_id>/messages", methods=["GET"])
@login_required
def get_chat_messages(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    msgs = (
        Message.query.filter_by(chat_id=chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return jsonify({
        "title": chat.title,
        "messages": [{
            "role": m.role,
            "content": m.content,
            "model": m.model,
            "has_image": m.has_image,
            "image_path": m.image_path,
            "created_at": m.created_at.isoformat()
        } for m in msgs]
    })


# -------- Main chat endpoint --------
@app.route("/chat", methods=["POST"])
@login_required
def chat_route():
    data = request.get_json() or {}

    user_message = (data.get("message") or "").strip()
    selected_model = data.get("model") or session.get("selected_model", "gemini-flash")
    chat_id = data.get("chat_id")
    uploaded_file = data.get("uploaded_file")  # can be filepath/filename/url

    model_cfg, is_premium_model = get_model_config(selected_model)
    if model_cfg is None:
        return jsonify({"error": "Invalid model"}), 400

    if is_premium_model and not current_user.is_premium:
        return jsonify({"error": "🔒 This model requires Premium subscription", "upgrade_url": "/checkout"}), 403

    if selected_model == "deepseek-chat" and not current_user.is_premium:
        if not check_deepseek_limit(current_user):
            return jsonify({"error": "⚠️ Daily DeepSeek limit reached (50/day). Try tomorrow or upgrade!"}), 429

    # Load / create chat
    if chat_id:
        chat_obj = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat_obj:
            return jsonify({"error": "Chat not found"}), 404
    else:
        chat_obj = Chat(user_id=current_user.id, title=generate_chat_title(user_message))
        db.session.add(chat_obj)
        db.session.flush()  # get id without full commit
        chat_id = chat_obj.id

    # Save user message (and store image reference if provided)
    img_abs = normalize_uploaded_path(uploaded_file)
    user_msg_db = Message(
        chat_id=chat_id,
        role="user",
        content=user_message if user_message else ("[Uploaded file]" if uploaded_file else ""),
        has_image=bool(img_abs),
        image_path=uploaded_file if uploaded_file else None
    )
    db.session.add(user_msg_db)

    # Update title on first message
    if chat_obj.title == "New Chat" or not chat_obj.title:
        chat_obj.title = generate_chat_title(user_message)

    # Prepare minimal context (last N messages)
    last_msgs = (
        Message.query.filter_by(chat_id=chat_id)
        .order_by(Message.created_at.desc())
        .limit(12)
        .all()
    )
    last_msgs = list(reversed(last_msgs))

    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for m in last_msgs:
        if m.role not in ("user", "assistant", "system"):
            continue
        # Keep previous context as text-only to avoid huge payloads
        messages.append({"role": m.role, "content": m.content})

    # If this request has an image and model supports vision, send it with the LAST user message
    if img_abs and model_cfg.get("vision"):
        b64 = encode_image(img_abs)
        if b64:
            messages[-1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message or "Analyze this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }

    try:
        resp = openrouter_client.chat.completions.create(
            model=model_cfg["path"],
            messages=messages
        )
        bot_response = resp.choices[0].message.content
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"AI Error: {str(e)}"}), 500

    # Save assistant message
    assistant_db = Message(
        chat_id=chat_id,
        role="assistant",
        content=bot_response,
        model=model_cfg.get("name")
    )
    db.session.add(assistant_db)

    # Deepseek counter
    if selected_model == "deepseek-chat" and not current_user.is_premium:
        current_user.deepseek_count += 1

    chat_obj.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "success": True,
        "response": bot_response,
        "model": model_cfg.get("name"),
        "chat_id": chat_id,
        "chat_title": chat_obj.title,
        "premium": current_user.is_premium,
        "deepseek_remaining": (50 - current_user.deepseek_count) if selected_model == "deepseek-chat" and not current_user.is_premium else None
    })


# -------- Stripe (optional) --------
@app.route("/checkout")
@login_required
def checkout():
    if not stripe.api_key:
        return "Stripe not configured (missing STRIPE_SECRET_KEY).", 503

    if current_user.is_premium:
        return redirect(url_for("dashboard"))

    try:
        session_obj = stripe.checkout.Session.create(
            payment_method_types=["card"],
            customer_email=current_user.email,
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": "NexaAI Premium",
                        "description": "Unlimited access to premium AI models",
                    },
                    "unit_amount": 1999,
                    "recurring": {"interval": "month"},
                },
                "quantity": 1,
            }],
            mode="subscription",
            success_url=url_for("payment_success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("dashboard", _external=True),
        )
        return redirect(session_obj.url)
    except Exception as e:
        return str(e), 500


@app.route("/payment-success")
@login_required
def payment_success():
    if not stripe.api_key:
        return redirect(url_for("dashboard"))

    session_id = request.args.get("session_id")
    if session_id:
        try:
            s = stripe.checkout.Session.retrieve(session_id)
            if s and getattr(s, "payment_status", None) == "paid":
                current_user.is_premium = True
                current_user.subscription_id = getattr(s, "subscription", None)
                db.session.commit()
        except Exception:
            pass
    return redirect(url_for("dashboard"))


@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    # Optional: you can expand this later. Keep safe default.
    return "", 200


# -------------------- Boot --------------------
if __name__ == "__main__":
    with app.app_context():
        migrate_database_sqlite()
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
