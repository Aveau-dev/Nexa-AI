# app.py
from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    session, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
import requests
import stripe
import os
import sqlite3
import base64
from PIL import Image
import traceback
import logging
import openai

# Google Generative AI for Gemini
import google.generativeai as genai

# Load environment
load_dotenv()

# -------- App setup --------
app = Flask(__name__)
CORS(app)

# Basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============ CONFIGURATION ============
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['JSON_AS_ASCII'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------- DB & login --------
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# -------- Configure AI APIs --------
# Google Generative AI
google_api_key = os.getenv('GOOGLE_API_KEY')
if google_api_key:
    genai.configure(api_key=google_api_key)
    log.info("✅ Google Generative AI configured")
else:
    log.warning("⚠️ GOOGLE_API_KEY not found")

# OpenRouter client
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if openrouter_api_key:
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    log.info("✅ OpenRouter configured")
else:
    openrouter_client = None
    log.warning("⚠️ OPENROUTER_API_KEY not found")

# ============ MODELS ============
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    subscription_id = db.Column(db.String(100), nullable=True)
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.email}>'

class Chat(db.Model):
    __tablename__ = 'chat'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    title = db.Column(db.String(200), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Chat {self.id}: {self.title}>'

class Message(db.Model):
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(200), nullable=True)
    has_image = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(1000), nullable=True)
    image_url = db.Column(db.String(1000), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'

# -------- login loader --------
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        log.exception("Error loading user")
        return None

# ============ DATABASE MIGRATION (sqlite only; safe checks) ============
def migrate_database_sqlite():
    """Attempt to add missing columns for older sqlite DBs (best-effort)."""
    try:
        uri = app.config['SQLALCHEMY_DATABASE_URI']
        if not uri.startswith('sqlite:///'):
            log.info("Skipping sqlite-specific migration (not sqlite).")
            return

        db_path = uri.replace('sqlite:///', '', 1)
        if not os.path.exists(db_path):
            log.info("Database file does not exist yet; create_all will handle creation.")
            return

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        def table_columns(table_name):
            cur.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cur.fetchall()]

        # Message table
        try:
            cols = table_columns('message')
            if 'has_image' not in cols:
                cur.execute("ALTER TABLE message ADD COLUMN has_image BOOLEAN DEFAULT 0")
                log.info("Added message.has_image")
            if 'image_path' not in cols:
                cur.execute("ALTER TABLE message ADD COLUMN image_path TEXT")
                log.info("Added message.image_path")
            if 'image_url' not in cols:
                cur.execute("ALTER TABLE message ADD COLUMN image_url TEXT")
                log.info("Added message.image_url")
        except sqlite3.OperationalError:
            log.info("Message table not present yet (will be created).")

        # User table
        try:
            cols = table_columns('user')
            if 'subscription_id' not in cols:
                cur.execute("ALTER TABLE user ADD COLUMN subscription_id TEXT")
                log.info("Added user.subscription_id")
            if 'is_premium' not in cols:
                cur.execute("ALTER TABLE user ADD COLUMN is_premium BOOLEAN DEFAULT 0")
                log.info("Added user.is_premium")
            if 'deepseek_count' not in cols:
                cur.execute("ALTER TABLE user ADD COLUMN deepseek_count INTEGER DEFAULT 0")
                log.info("Added user.deepseek_count")
            if 'deepseek_date' not in cols:
                cur.execute("ALTER TABLE user ADD COLUMN deepseek_date TEXT")
                log.info("Added user.deepseek_date")
        except sqlite3.OperationalError:
            log.info("User table not present yet (will be created).")

        conn.commit()
        conn.close()
        log.info("SQLite migration (best-effort) completed.")
    except Exception:
        log.exception("SQLite migration failed (best-effort).")

# ============ AI MODELS CONFIG ============
FREE_MODELS = {
    "gemini-flash": {
        "name": "Gemini 2.5 Flash ⚡",
        "model": "gemini-2.5-flash-lite",
        "provider": "google",
        "vision": True,
        "limit": None,
        "description": "Fast and efficient Gemini Flash model"
    },
    "gpt-3.5-turbo": {
        "name": "ChatGPT 3.5 Turbo 🤖",
        "model": "openai/gpt-3.5-turbo",
        "provider": "openrouter",
        "vision": False,
        "limit": None,
        "description": "OpenAI GPT-3.5 Turbo"
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku 🎭",
        "model": "anthropic/claude-3-haiku",
        "provider": "openrouter",
        "vision": True,
        "limit": None,
        "description": "Fast Claude model"
    },
    "deepseek-chat": {
        "name": "DeepSeek Chat 🔍",
        "model": "deepseek/deepseek-chat",
        "provider": "openrouter",
        "vision": False,
        "limit": 50,
        "description": "Powerful for code & logic (50/day for free users)"
    },
}

PREMIUM_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o 🚀",
        "model": "openai/gpt-4o",
        "provider": "openrouter",
        "vision": True,
        "description": "Most capable GPT-4 family model"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini 💎",
        "model": "openai/gpt-4o-mini",
        "provider": "openrouter",
        "vision": True,
        "description": "Efficient GPT-4 style model"
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet 🎭",
        "model": "anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "vision": True,
        "description": "Best for coding & analysis"
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus 👑",
        "model": "anthropic/claude-3-opus",
        "provider": "openrouter",
        "vision": True,
        "description": "Most capable Claude model"
    },
    "gemini-pro": {
        "name": "Gemini 1.5 Pro 👁️",
        "model": "gemini-1.5-pro",
        "provider": "google",
        "vision": True,
        "description": "Multimodal vision-capable"
    },
    "deepseek-r1": {
        "name": "DeepSeek R1 🚀",
        "model": "deepseek/deepseek-r1",
        "provider": "openrouter",
        "vision": False,
        "description": "Advanced reasoning model"
    },
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


# ============ AI API CALLING FUNCTIONS ============

def call_google_gemini(model_path, messages, timeout=60):
    """Call Google Generative AI for Gemini models"""
    try:
        if not google_api_key:
            raise Exception("Google API key not configured")
        
        model = genai.GenerativeModel(model_path)
        
        # Convert messages to Gemini format
        gemini_history = []
        last_message = None
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Skip system messages (Gemini doesn't use them explicitly)
            if role == 'system':
                continue
            
            # Convert role
            gemini_role = 'user' if role == 'user' else 'model'
            
            # Store last message separately (will be sent via send_message)
            if msg == messages[-1]:
                last_message = {'role': gemini_role, 'parts': [content]}
            else:
                gemini_history.append({'role': gemini_role, 'parts': [content]})
        
        # Start chat with history
        chat = model.start_chat(history=gemini_history)
        
        # Send last message
        if last_message:
            response = chat.send_message(last_message['parts'][0])
        else:
            response = chat.send_message(messages[-1].get('content', ''))
        
        return response.text
    
    except Exception as e:
        log.exception("Google Gemini API error")
        raise Exception(f"Google Gemini API error: {str(e)}")


def call_openrouter(model_path, messages, timeout=60):
    """Call OpenRouter API for GPT, Claude, DeepSeek models"""
    try:
        if not openrouter_client:
            raise Exception("OpenRouter API key not configured")
        
        response = openrouter_client.chat.completions.create(
            model=model_path,
            messages=messages,
            timeout=timeout
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        log.exception("OpenRouter API error")
        raise Exception(f"OpenRouter API error: {str(e)}")


def call_ai_model(model_key, messages, uploaded_file=None):
    """Universal function to call any AI model - routes to correct provider"""
    # Get model config
    if model_key in FREE_MODELS:
        model_config = FREE_MODELS[model_key]
    elif model_key in PREMIUM_MODELS:
        model_config = PREMIUM_MODELS[model_key]
    else:
        raise Exception(f"Model {model_key} not found")
    
    provider = model_config['provider']
    model_path = model_config['model']
    
    # Handle image uploads for vision models
    if uploaded_file and model_config.get('vision'):
        try:
            if os.path.exists(uploaded_file):
                # Add image context to the last user message
                messages[-1]['content'] = f"{messages[-1]['content']}\n[Image attached: {uploaded_file}]"
        except Exception as e:
            log.exception("Error processing image")
    
    # Route to appropriate provider
    if provider == 'google':
        return call_google_gemini(model_path, messages)
    elif provider == 'openrouter':
        return call_openrouter(model_path, messages)
    else:
        raise Exception(f"Unknown provider: {provider}")


# ============ HELPER FUNCTIONS ============
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        log.exception("Failed to encode image")
        return None

def compress_image(image_path, max_size=(1024, 1024), quality=85):
    try:
        img = Image.open(image_path)
        if getattr(img, "mode", None) == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        resample = getattr(Image, "Resampling", None)
        if resample:
            resample_filter = Image.Resampling.LANCZOS
        else:
            resample_filter = Image.LANCZOS
        img.thumbnail(max_size, resample_filter)
        img.save(image_path, optimize=True, quality=quality)
        log.info("Compressed image %s", image_path)
    except Exception:
        log.exception("Image compression failed for %s", image_path)

def generate_chat_title(first_message):
    title = first_message.strip()[:50]
    if len(first_message.strip()) > 50:
        title += "..."
    return title

def check_deepseek_limit(user):
    """Check and reset DeepSeek daily limit"""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    
    return user.deepseek_count < 50

def get_chat_history(chat_id, limit=10):
    """Return last `limit` messages as list for API"""
    try:
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at.desc()).limit(limit).all()
        messages = list(reversed(messages))
        
        formatted = []
        for msg in messages:
            content = msg.content
            
            # Add image reference if exists
            if msg.has_image and (msg.image_path or msg.image_url):
                img_ref = msg.image_url or msg.image_path
                content = f"{content}\n[Image: {img_ref}]"
            
            formatted.append({
                'role': msg.role,
                'content': content
            })
        
        return formatted
    except Exception:
        log.exception("Failed to fetch chat history")
        return []

def generate_image(prompt):
    """Generate image URL using Pollinations"""
    try:
        encoded_prompt = requests.utils.quote(prompt)
        seed = int(datetime.utcnow().timestamp())
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}"
    except Exception:
        log.exception("Failed to construct image URL")
        raise RuntimeError("Image generation failed")

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(e):
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

# ============ ROUTES ============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    try:
        data = request.get_json() or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Image generation check
        image_keywords = ['generate image', 'create image', 'draw', 'picture of', 'image of', 'make an image']
        if any(k in message.lower() for k in image_keywords):
            try:
                url = generate_image(message)
                return jsonify({
                    'response': 'Image generated successfully!',
                    'image_url': url,
                    'demo': True,
                    'model': 'Pollinations'
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # Use Gemini Flash for demo
        try:
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                {'role': 'user', 'content': message}
            ]
            response_text = call_ai_model('gemini-flash', messages)
            return jsonify({
                'response': response_text,
                'demo': True,
                'model': 'gemini 2.5 flash lite'
            })
        except Exception as e:
            log.exception("Demo chat failed")
            return jsonify({'error': str(e)}), 500

    except Exception:
        log.exception("Demo chat outer error")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

# -------- Auth routes --------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            email = (data.get('email') or '').strip().lower()
            name = (data.get('name') or '').strip()
            password = data.get('password') or ''

            if not email or not name or not password:
                return jsonify({'error': 'All fields are required'}), 400
            if len(password) < 6:
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already exists'}), 400

            hashed = generate_password_hash(password)
            new_user = User(
                email=email,
                name=name,
                password=hashed,
                deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
            )
            db.session.add(new_user)
            db.session.commit()
            log.info("New user registered: %s", email)

            if request.is_json:
                return jsonify({'success': True, 'redirect': url_for('login')})
            return redirect(url_for('login'))
        except Exception:
            db.session.rollback()
            log.exception("Signup failed")
            if request.is_json:
                return jsonify({'error': 'An error occurred. Please try again.'}), 500
            return render_template('signup.html', error='An error occurred. Please try again.')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            email = (data.get('email') or '').strip().lower()
            password = data.get('password') or ''
            if not email or not password:
                return jsonify({'error': 'Email and password are required'}), 400

            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                login_user(user, remember=True)
                log.info("User logged in: %s", email)
                if request.is_json:
                    return jsonify({'success': True, 'redirect': url_for('dashboard')})
                return redirect(url_for('dashboard'))

            if request.is_json:
                return jsonify({'error': 'Invalid email or password'}), 401
            return render_template('login.html', error='Invalid email or password')
        except Exception:
            log.exception("Login failed")
            if request.is_json:
                return jsonify({'error': 'An error occurred. Please try again.'}), 500
            return render_template('login.html', error='An error occurred. Please try again.')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).all()
        all_models = {**FREE_MODELS, **PREMIUM_MODELS}
        return render_template('dashboard.html', user=current_user, chats=chats, models=all_models)
    except Exception:
        log.exception("Dashboard load failed")
        return render_template('dashboard.html', user=current_user, chats=[], models={})

# -------- Upload --------
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{current_user.id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            compress_image(filepath)
        except Exception:
            pass

        file_url = url_for('uploaded_file', filename=filename, _external=True)
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'url': file_url
        })
    except Exception:
        log.exception("Upload failed")
        return jsonify({'error': 'Upload failed. Please try again.'}), 500

# -------- Chat management --------
@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    try:
        chat = Chat(user_id=current_user.id, title='New Chat')
        db.session.add(chat)
        db.session.commit()
        return jsonify({'success': True, 'chat_id': chat.id, 'title': chat.title})
    except Exception:
        db.session.rollback()
        log.exception("Create new chat failed")
        return jsonify({'error': 'Failed to create chat'}), 500

@app.route('/chat/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        payload = request.get_json() or {}
        new_title = (payload.get('title') or '').strip()
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400
        chat.title = new_title
        db.session.commit()
        return jsonify({'success': True})
    except Exception:
        db.session.rollback()
        log.exception("Rename chat failed")
        return jsonify({'error': 'Failed to rename chat'}), 500

@app.route('/chat/<int:chat_id>/delete', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Delete attached files
        for msg in chat.messages:
            if msg.has_image and msg.image_path:
                try:
                    if os.path.isabs(msg.image_path):
                        path = msg.image_path
                    else:
                        path = os.path.join(app.config['UPLOAD_FOLDER'], msg.image_path)
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    log.exception("Failed to delete attached file")
        
        db.session.delete(chat)
        db.session.commit()
        return jsonify({'success': True})
    except Exception:
        db.session.rollback()
        log.exception("Delete chat failed")
        return jsonify({'error': 'Failed to delete chat'}), 500

@app.route('/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at.asc()).all()
        out = []
        for msg in messages:
            out.append({
                'role': msg.role,
                'content': msg.content,
                'model': msg.model,
                'image_url': msg.image_url,
                'image_path': msg.image_path,
                'created_at': msg.created_at.isoformat()
            })
        return jsonify({'messages': out, 'title': chat.title})
    except Exception:
        log.exception("Get messages failed")
        return jsonify({'error': 'Failed to load messages'}), 500

# -------- Main chat endpoint (UPDATED) --------
@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        selected_model = data.get('model') or 'gemini-flash'
        chat_id = data.get('chat_id')
        uploaded_file_path = data.get('uploaded_file')

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Get or create chat
        if chat_id:
            chat_obj = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
            if not chat_obj:
                return jsonify({'error': 'Chat not found'}), 404
        else:
            chat_obj = Chat(user_id=current_user.id, title='New Chat')
            db.session.add(chat_obj)
            db.session.commit()
            chat_id = chat_obj.id

        # Get model config
        if selected_model in FREE_MODELS:
            model_info = FREE_MODELS[selected_model]
            is_premium_model = False
        elif selected_model in PREMIUM_MODELS:
            model_info = PREMIUM_MODELS[selected_model]
            is_premium_model = True
        else:
            return jsonify({'error': 'Invalid model selected'}), 400

        # Premium check
        if is_premium_model and not current_user.is_premium:
            return jsonify({
                'error': 'This model requires Premium subscription',
                'upgrade_required': True,
                'upgrade_url': url_for('checkout')
            }), 403

        # DeepSeek limit check
        if selected_model == 'deepseek-chat' and not current_user.is_premium:
            if not check_deepseek_limit(current_user):
                return jsonify({
                    'error': 'Daily limit reached for DeepSeek (50/day). Upgrade to premium for unlimited access.',
                    'upgrade_required': True,
                    'upgrade_url': url_for('checkout')
                }), 429

        # Image generation detection
        image_keywords = ['generate image', 'create image', 'draw', 'picture of', 'image of', 'make an image']
        if any(k in user_message.lower() for k in image_keywords):
            try:
                image_url = generate_image(user_message)
                
                user_msg = Message(chat_id=chat_id, role='user', content=user_message)
                db.session.add(user_msg)
                
                assistant_text = "I've generated the image for you!"
                assistant_msg = Message(
                    chat_id=chat_id,
                    role='assistant',
                    content=assistant_text,
                    model='Pollinations',
                    image_url=image_url,
                    has_image=True
                )
                db.session.add(assistant_msg)
                
                if chat_obj.title == 'New Chat':
                    chat_obj.title = generate_chat_title(user_message)
                
                chat_obj.updated_at = datetime.utcnow()
                db.session.commit()
                
                return jsonify({
                    'response': assistant_text,
                    'image_url': image_url,
                    'model': 'Pollinations',
                    'chat_id': chat_id,
                    'chat_title': chat_obj.title
                })
            except Exception as e:
                db.session.rollback()
                log.exception("Image generation failed")
                return jsonify({'error': str(e)}), 500

        # Save user message
        user_msg = Message(
            chat_id=chat_id,
            role='user',
            content=user_message,
            has_image=bool(uploaded_file_path),
            image_path=uploaded_file_path
        )
        db.session.add(user_msg)

        if chat_obj.title == 'New Chat':
            chat_obj.title = generate_chat_title(user_message)

        # Build conversation history
        history = get_chat_history(chat_id, limit=10)
        history.append({'role': 'system', 'content': 'You are a helpful AI assistant.'})
        history.append({'role': 'user', 'content': user_message})

        # Call AI model
        try:
            bot_response = call_ai_model(selected_model, history, uploaded_file_path)
        except Exception as e:
            db.session.rollback()
            log.exception("AI call failed")
            return jsonify({'error': str(e)}), 500

        # Save assistant response
        assistant_msg = Message(
            chat_id=chat_id,
            role='assistant',
            content=bot_response,
            model=model_info.get('name')
        )
        db.session.add(assistant_msg)

        # Update DeepSeek count if applicable
        if selected_model == 'deepseek-chat' and not current_user.is_premium:
            current_user.deepseek_count += 1

        # Finalize
        chat_obj.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'response': bot_response,
            'model': model_info.get('name'),
            'chat_id': chat_id,
            'chat_title': chat_obj.title,
            'premium': current_user.is_premium
        })
        
    except Exception:
        db.session.rollback()
        log.exception("Chat route failed")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

# ============ STRIPE ROUTES ============
stripe_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_key:
    stripe.api_key = stripe_key
    log.info("✅ Stripe configured")
else:
    stripe.api_key = None
    log.warning("⚠️ Stripe not configured")

@app.route('/checkout')
@login_required
def checkout():
    if not stripe.api_key:
        return "Payment system not configured", 503
    if current_user.is_premium:
        return redirect(url_for('dashboard'))
    
    try:
        session_obj = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=current_user.email,
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'NexaAI Premium',
                        'description': 'Unlimited access to premium AI models',
                    },
                    'unit_amount': 1999,
                    'recurring': {'interval': 'month'}
                },
                'quantity': 1
            }],
            mode='subscription',
            success_url=url_for('payment_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('dashboard', _external=True),
        )
        return redirect(session_obj.url)
    except Exception:
        log.exception("Stripe checkout creation failed")
        return "Failed to create checkout session", 500

@app.route('/payment-success')
@login_required
def payment_success():
    session_id = request.args.get('session_id')
    
    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status == 'paid':
                current_user.is_premium = True
                current_user.subscription_id = session.subscription
                db.session.commit()
                log.info("User upgraded to premium: %s", current_user.email)
        except Exception as e:
            log.exception("Payment verification failed")
    
    return redirect(url_for('dashboard'))

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    if not webhook_secret:
        return '', 200
    
    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=webhook_secret
        )
    except Exception:
        log.exception("Webhook signature verification failed")
        return '', 400

    # Handle events
    if event['type'] == 'checkout.session.completed':
        sess = event['data']['object']
        customer_email = sess.get('customer_email')
        sub_id = sess.get('subscription')
        if customer_email:
            user = User.query.filter_by(email=customer_email).first()
            if user:
                user.is_premium = True
                user.subscription_id = sub_id
                db.session.commit()
                log.info("User upgraded via webhook: %s", user.email)
    
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        user = User.query.filter_by(subscription_id=subscription.get('id')).first()
        if user:
            user.is_premium = False
            user.subscription_id = None
            db.session.commit()
            log.info("User subscription cancelled: %s", user.email)

    return '', 200

# ============ API ENDPOINTS ============
@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({'free': FREE_MODELS, 'premium': PREMIUM_MODELS})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'google_ai_configured': bool(google_api_key),
        'openrouter_configured': bool(openrouter_client),
        'stripe_configured': bool(stripe.api_key),
        'upload_folder': app.config['UPLOAD_FOLDER']
    })

# ============ DB INIT ============
def init_database():
    """Create tables and run migrations"""
    try:
        with app.app_context():
            migrate_database_sqlite()
            db.create_all()
            log.info("✅ Database initialized successfully")
    except Exception:
        log.exception("Database initialization failed")

init_database()

# ============ RUN ============
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
