"""
NexaAI - Advanced AI Chat Platform with Demo Mode
Complete Production Version with Supabase Support & Memory System
Author: Aarav
Date: 2026-01-09
Version: 4.1 - Fixed Flask 3.x Compatibility
"""

import os
import logging
import base64
import urllib.parse
import requests
import hashlib
import json
from datetime import datetime, timedelta
from io import BytesIO
from functools import wraps
from collections import defaultdict

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import desc, exc, text

# Import Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nexaai.log', mode='a')
    ]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# FLASK APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nexaai-secret-key-2025-change-in-production')
app.config['JSON_AS_ASCII'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Database configuration
database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')

if database_url:
    # Fix postgres:// to postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    # URL-encode password if it contains special characters
    try:
        from urllib.parse import urlparse, urlunparse, quote
        parsed = urlparse(database_url)

        if parsed.password and any(char in parsed.password for char in ['!', '@', '#', '$', '%']):
            scheme = parsed.scheme
            username = parsed.username
            password = quote(parsed.password, safe='')
            hostname = parsed.hostname
            port = parsed.port
            path = parsed.path

            if port:
                netloc = f"{username}:{password}@{hostname}:{port}"
            else:
                netloc = f"{username}:{password}@{hostname}"

            database_url = urlunparse((scheme, netloc, path, '', '', ''))
            log.info("Database password URL-encoded (contains special characters)")
    except Exception as e:
        log.warning(f"Could not parse DATABASE_URL: {e}")

    # Add SSL mode for Supabase
    if 'supabase' in database_url.lower():
        if '?' not in database_url:
            database_url += '?sslmode=require'
        elif 'sslmode' not in database_url:
            database_url += '&sslmode=require'

    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    log.info("🐘 Using Supabase PostgreSQL")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nexaai.db'
    log.info("📁 Using SQLite database (fallback)")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,
    'pool_size': 3,
    'max_overflow': 5,
    'pool_timeout': 30,
    'connect_args': {
        'connect_timeout': 10
    }
}


# File upload configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# EXTENSIONS INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'
login_manager.login_message = 'Please log in to access this page.'

# ═══════════════════════════════════════════════════════════════════
# API KEYS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

log.info("🔑 Configuring API Keys...")

# Google API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = GOOGLE_API_KEY.strip().strip('"').strip("'")
        if len(GOOGLE_API_KEY) < 10:
            log.error(f"GOOGLE_API_KEY is too short ({len(GOOGLE_API_KEY)} chars)")
            GOOGLE_API_KEY = None
        else:
            if genai:
                genai.configure(api_key=GOOGLE_API_KEY)
                log.info("✅ Google Generative AI configured successfully")
                log.info(f"   API Key length: {len(GOOGLE_API_KEY)} chars")
                log.info(f"   API Key prefix: {GOOGLE_API_KEY[:10]}...")
            else:
                log.warning("google-generativeai package not installed")
    except Exception as e:
        log.error(f"Google AI configuration failed: {e}")
        GOOGLE_API_KEY = None
else:
    log.error("❌ GOOGLE_API_KEY not set in environment variables")
    log.error("Demo mode will NOT work without this key!")
    GOOGLE_API_KEY = None

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip().strip('"').strip("'")
    log.info("✅ OpenRouter API configured")
    log.info(f"   API Key length: {len(OPENROUTER_API_KEY)} chars")
else:
    log.warning("⚠️ OPENROUTER_API_KEY not set")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

# ═══════════════════════════════════════════════════════════════════
# DEMO RATE LIMITING
# ═══════════════════════════════════════════════════════════════════

demo_rate_limit = defaultdict(list)
DEMO_RATE_LIMIT = 10  # messages per hour
DEMO_RATE_WINDOW = 3600  # 1 hour in seconds

def check_demo_rate_limit(ip_address):
    """Check if demo user has exceeded rate limit"""
    now = datetime.utcnow().timestamp()

    # Clean old entries
    demo_rate_limit[ip_address] = [
        ts for ts in demo_rate_limit[ip_address] 
        if now - ts < DEMO_RATE_WINDOW
    ]

    # Check limit
    if len(demo_rate_limit[ip_address]) >= DEMO_RATE_LIMIT:
        return False

    # Add current request
    demo_rate_limit[ip_address].append(now)
    return True

# ═══════════════════════════════════════════════════════════════════
# AI MODELS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

FREE_MODELS = [
    {
        "id": "deepseek-r1",
        "name": "DeepSeek R1 (Reasoning)",
        "model": "deepseek/deepseek-r1-0528:free",
        "provider": "openrouter",
        "vision": False,
        "tier": "free",
        "rank": 1,
        "description": "Advanced reasoning AI - Shows thought process",
        "speed": "Medium",
        "context": "64K tokens",
        "demo_enabled": True,
        "has_reasoning": True
    },
    {
        "id": "gemini-flash",
        "name": "Gemini 2.0 Flash",
        "model": "google/gemini-2.0-flash-exp:free",
        "provider": "openrouter",
        "vision": True,
        "tier": "free",
        "rank": 2,
        "description": "Fast with vision support - Image analysis",
        "speed": "Very Fast",
        "context": "1M tokens",
        "demo_enabled": True
    },
    {
        "id": "gpt-3.5-turbo",
        "name": "ChatGPT 3.5 Turbo",
        "model": "openai/gpt-3.5-turbo",
        "provider": "openrouter",
        "vision": False,
        "tier": "free",
        "rank": 3,
        "description": "OpenAI GPT-3.5 - Fast and reliable",
        "speed": "Fast",
        "context": "16K tokens",
        "demo_enabled": False
    },
    {
        "id": "claude-haiku",
        "name": "Claude 3 Haiku",
        "model": "anthropic/claude-3-haiku",
        "provider": "openrouter",
        "vision": True,
        "tier": "free",
        "rank": 4,
        "description": "Fast Claude with vision support",
        "speed": "Very Fast",
        "context": "200K tokens",
        "demo_enabled": False
    }
]

PREMIUM_MODELS = [
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "model": "openai/gpt-4o-mini",
        "provider": "openrouter",
        "vision": True,
        "tier": "pro",
        "rank": 6,
        "description": "Efficient GPT-4 level performance",
        "speed": "Fast",
        "context": "128K tokens"
    },
    {
        "id": "gemini-pro",
        "name": "Gemini 1.5 Pro",
        "model": "google/gemini-1.5-pro",
        "provider": "google",
        "vision": True,
        "tier": "pro",
        "rank": 7,
        "description": "Advanced multimodal AI",
        "speed": "Medium",
        "context": "2M tokens"
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "model": "openai/gpt-4o",
        "provider": "openrouter",
        "vision": True,
        "tier": "max",
        "rank": 8,
        "description": "Most capable GPT-4 model",
        "speed": "Medium",
        "context": "128K tokens"
    },
    {
        "id": "claude-sonnet",
        "name": "Claude 3.5 Sonnet",
        "model": "anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "vision": True,
        "tier": "max",
        "rank": 9,
        "description": "Best for coding analysis",
        "speed": "Medium",
        "context": "200K tokens"
    }
]

# ═══════════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)

    # Plan Management
    plan = db.Column(db.String(20), default='pro', nullable=False)  # basic, pro, max
    plan_started_at = db.Column(db.DateTime, default=datetime.utcnow)
    subscription_status = db.Column(db.String(20), default='trial')  # trial, active, expired
    trial_ends_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=180))  # 6 months

    # Usage Limits
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10), default='')

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')
    memories = db.relationship('Memory', backref='user', lazy=True, cascade='all, delete-orphan')
    settings = db.relationship('UserSettings', backref='user', lazy=True, uselist=False, cascade='all, delete-orphan')

    @property
    def is_premium(self):
        # Check if trial is still active
        if self.subscription_status == 'trial' and self.trial_ends_at:
            if datetime.utcnow() < self.trial_ends_at:
                return self.plan in ['pro', 'max']
        return self.plan in ['pro', 'max'] and self.subscription_status == 'active'

    def __repr__(self):
        return f'<User {self.email} - {self.plan}>'


class Chat(db.Model):
    __tablename__ = 'chats'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    title = db.Column(db.String(200), default='New Chat')
    is_demo = db.Column(db.Boolean, default=False)
    session_id = db.Column(db.String(100), nullable=True)  # For demo sessions

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan', order_by='Message.created_at')

    def __repr__(self):
        return f'<Chat {self.id}: {self.title} (Demo: {self.is_demo})>'


class Message(db.Model):
    __tablename__ = 'messages'

    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # user, assistant, system
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(200), nullable=True)

    # Image/File support
    has_image = db.Column(db.Boolean, default=False)
    image_url = db.Column(db.String(1000), nullable=True)
    image_data = db.Column(db.Text, nullable=True)  # Base64 encoded

    # Reasoning support
    has_reasoning = db.Column(db.Boolean, default=False)
    reasoning_content = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'


class Memory(db.Model):
    __tablename__ = 'memories'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    key = db.Column(db.String(100), nullable=False)  # e.g., 'favorite_color', 'preferences'
    value = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), default='general')  # general, preference, fact, etc.

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Memory {self.key}: {self.value[:50]}>'


class UserSettings(db.Model):
    __tablename__ = 'user_settings'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True, index=True)

    # UI Preferences
    theme = db.Column(db.String(20), default='dark')  # dark, light, auto
    default_model = db.Column(db.String(50), default='deepseek-r1')

    # Feature Preferences
    enable_reasoning = db.Column(db.Boolean, default=True)
    enable_memory = db.Column(db.Boolean, default=True)
    enable_web_search = db.Column(db.Boolean, default=False)

    # Privacy
    save_chat_history = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Settings for User {self.user_id}>'

# ═══════════════════════════════════════════════════════════════════
# DATABASE ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════

@app.errorhandler(exc.SQLAlchemyError)
def handle_db_error(error):
    """Handle all database errors"""
    log.error(f"Database error: {error}")
    db.session.rollback()

    error_msg = str(error).lower()

    if 'timeout' in error_msg or 'timed out' in error_msg:
        return jsonify({"error": "Database connection timeout. Please try again."}), 503
    elif 'ssl' in error_msg:
        return jsonify({"error": "Database SSL connection failed. Please contact support."}), 500
    elif 'authentication' in error_msg or 'password' in error_msg:
        return jsonify({"error": "Database authentication failed. Please contact support."}), 500
    elif 'connection' in error_msg:
        return jsonify({"error": "Cannot connect to database. Please try again later."}), 503
    else:
        return jsonify({"error": "Database error occurred. Please try again."}), 500


@app.teardown_appcontext
def shutdown_session(exception=None):
    """Remove database session after each request"""
    try:
        if exception:
            db.session.rollback()
        db.session.remove()
    except Exception as e:
        log.error(f"Session cleanup error: {e}")

# ═══════════════════════════════════════════════════════════════════
# LOGIN MANAGER
# ═══════════════════════════════════════════════════════════════════

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        log.error(f"Error loading user {user_id}: {e}")
        db.session.rollback()
        return None

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_demo_session_id():
    """Get or create demo session ID"""
    if 'demo_session_id' not in session:
        session['demo_session_id'] = hashlib.md5(
            f"{request.remote_addr}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()
    return session['demo_session_id']


def get_available_models(user=None):
    """Get list of models available to user based on their plan"""
    if user and user.is_premium:
        plan = user.plan
    else:
        plan = 'basic'

    available_free = FREE_MODELS

    if plan == 'basic':
        return available_free
    elif plan == 'pro':
        return available_free + PREMIUM_MODELS[:2]
    else:  # max
        return available_free + PREMIUM_MODELS


def check_deepseek_limit(user):
    """Check and reset daily DeepSeek limit"""
    if not user:
        return False

    today = datetime.utcnow().strftime('%Y-%m-%d')

    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()

    if user.is_premium:
        return True

    return user.deepseek_count < 50


def increment_deepseek_count(user):
    """Increment daily DeepSeek usage counter"""
    if not user:
        return

    today = datetime.utcnow().strftime('%Y-%m-%d')

    if user.deepseek_date != today:
        user.deepseek_count = 1
        user.deepseek_date = today
    else:
        user.deepseek_count += 1

    db.session.commit()


def generate_image_url(prompt):
    """Generate image using Pollinations AI"""
    try:
        clean_prompt = prompt.lower()
        for prefix in ['draw', 'generate image', 'create image', 'make image', 'show me']:
            clean_prompt = clean_prompt.replace(prefix, '').strip()

        encoded = urllib.parse.quote(clean_prompt)
        seed = int(datetime.utcnow().timestamp())

        return f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&seed={seed}"
    except Exception as e:
        log.error(f"Image generation failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════
# AI API CALL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def call_google_gemini(model_path, messages, image_data=None, timeout=90):
    """Call Google Gemini API"""
    try:
        if not GOOGLE_API_KEY or not genai:
            raise Exception("Google API Key not configured")

        model = genai.GenerativeModel(model_path)

        content_parts = []

        if messages:
            last_msg = messages[-1]
            text = last_msg.get('content', '')

            if isinstance(text, list):
                for item in text:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        break

            if text:
                content_parts.append(str(text))

        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                with Image.open(BytesIO(image_bytes)) as img:
                    img.load()
                    content_parts.append(img)
                log.info("Image attached to Gemini request")
            except Exception as e:
                log.warning(f"Failed to process image: {e}")

        if not content_parts:
            content_parts = ["Hello"]

        response = model.generate_content(
            content_parts,
            request_options={'timeout': timeout}
        )

        if response and response.text:
            return response.text
        else:
            return "No response generated. Please try again."

    except Exception as e:
        error_msg = str(e)
        log.error(f"Gemini API error: {error_msg}")

        if 'API' in error_msg.upper() or 'KEY' in error_msg.upper():
            return "Error: Invalid API key configuration."
        elif 'TIMEOUT' in error_msg.upper():
            return "Error: Request timed out. Try a shorter prompt."
        elif 'QUOTA' in error_msg.upper() or 'RATE' in error_msg.upper():
            return "Error: API rate limit exceeded. Please wait and try again."
        else:
            return f"Error: {error_msg[:150]}"


def call_openrouter(model_path, messages, image_data=None, timeout=90):
    """Call OpenRouter API"""
    try:
        if not OPENROUTER_API_KEY:
            raise Exception("OpenRouter API Key not configured")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexa-ai.onrender.com",
            "X-Title": "NexaAI"
        }

        formatted_msgs = []
        for msg in messages:
            content = msg.get('content', '')

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content = item.get('text', '')
                        break

            formatted_msgs.append({
                "role": msg.get('role', 'user'),
                "content": str(content)
            })

        # Add image if present
        if image_data and formatted_msgs:
            formatted_msgs[-1]['content'] = [
                {"type": "text", "text": formatted_msgs[-1]['content']},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]

        payload = {
            "model": model_path,
            "messages": formatted_msgs,
            "temperature": 0.7
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )

        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text)
            raise Exception(f"OpenRouter error: {error_msg[:200]}")

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Try a shorter prompt."
    except Exception as e:
        log.error(f"OpenRouter error: {str(e)}")
        return f"Error: {str(e)[:150]}"


def call_ai_model(model_id, messages, image_data=None):
    """Universal function to call any AI model"""
    all_models = FREE_MODELS + PREMIUM_MODELS
    model_config = next((m for m in all_models if m['id'] == model_id), None)

    if not model_config:
        return "Error: Model not found"

    provider = model_config['provider']
    model_path = model_config['model']

    try:
        if provider == 'google':
            return call_google_gemini(model_path, messages, image_data)
        elif provider == 'openrouter':
            return call_openrouter(model_path, messages, image_data)
        else:
            return f"Error: Unknown provider {provider}"
    except Exception as e:
        log.error(f"AI model call failed: {e}")
        return f"Error calling AI: {str(e)[:100]}"

# ═══════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def init_db():
    """Initialize database"""
    with app.app_context():
        try:
            db.create_all()
            log.info("✅ Database tables created successfully")
            return True
        except Exception as e:
            log.error(f"❌ Failed to create database tables: {e}")
            return False


def test_db_connection():
    """Test database connection at startup"""
    try:
        with app.app_context():
            db.session.execute(text('SELECT 1'))
            log.info("✅ Database connection successful")
            return True
    except Exception as e:
        log.error(f"❌ Database connection failed: {e}")
        log.error("Please check your DATABASE_URL in .env file")
        return False

# ═══════════════════════════════════════════════════════════════════
# ROUTES - PUBLIC PAGES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Landing page with demo mode"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/demo')
def demo():
    """Demo mode page - No login required"""
    return render_template('index.html')

# ═══════════════════════════════════════════════════════════════════
# ROUTES - AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')

            if not email or not password:
                return jsonify({"success": False, "error": "Email and password required"}), 400

            user = db.session.query(User).filter_by(email=email).first()

            if user and check_password_hash(user.password, password):
                login_user(user, remember=True)
                user.last_login = datetime.utcnow()

                try:
                    db.session.commit()
                except exc.SQLAlchemyError as e:
                    db.session.rollback()
                    log.error(f"Login commit error: {e}")

                log.info(f"✅ User logged in: {email}")
                return jsonify({"success": True, "redirect": url_for('dashboard')})

            log.warning(f"❌ Failed login attempt: {email}")
            return jsonify({"success": False, "error": "Invalid email or password"}), 401

        except exc.OperationalError as e:
            db.session.rollback()
            log.error(f"Database connection error during login: {e}")
            return jsonify({"success": False, "error": "Database connection error. Please try again."}), 503
        except Exception as e:
            db.session.rollback()
            log.error(f"Login error: {e}")
            return jsonify({"success": False, "error": "Login failed. Please try again."}), 500

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            name = data.get('name', '').strip()

            if not email or not password or not name:
                return jsonify({"success": False, "error": "All fields required"}), 400

            if len(password) < 6:
                return jsonify({"success": False, "error": "Password must be at least 6 characters"}), 400

            existing_user = db.session.query(User).filter_by(email=email).first()
            if existing_user:
                return jsonify({"success": False, "error": "Email already registered"}), 409

            new_user = User(
                email=email,
                password=generate_password_hash(password),
                name=name,
                plan='pro',
                plan_started_at=datetime.utcnow(),
                subscription_status='trial',
                trial_ends_at=datetime.utcnow() + timedelta(days=180),  # 6 months free
                deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
            )

            db.session.add(new_user)

            try:
                db.session.commit()
            except exc.IntegrityError:
                db.session.rollback()
                return jsonify({"success": False, "error": "Email already registered"}), 409
            except exc.OperationalError as e:
                db.session.rollback()
                log.error(f"Database connection error during signup: {e}")
                return jsonify({"success": False, "error": "Database connection error. Please try again."}), 503

            # Create default settings
            try:
                settings = UserSettings(user_id=new_user.id)
                db.session.add(settings)
                db.session.commit()
            except:
                pass  # Settings are optional

            login_user(new_user, remember=True)
            log.info(f"✅ New user registered: {email} (Pro trial)")

            return jsonify({"success": True, "redirect": url_for('dashboard')})

        except Exception as e:
            db.session.rollback()
            log.error(f"Signup error: {e}")
            return jsonify({"success": False, "error": "Registration failed. Please try again."}), 500

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

# ═══════════════════════════════════════════════════════════════════
# ROUTES - DASHBOARD
# ═══════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    try:
        chats = Chat.query.filter_by(
            user_id=current_user.id,
            is_demo=False
        ).order_by(desc(Chat.updated_at)).limit(50).all()

        available_models = get_available_models(current_user)

        if 'selected_model' not in session:
            session['selected_model'] = 'deepseek-r1'
            session['selected_model_name'] = 'DeepSeek R1 (Reasoning)'

        return render_template(
            'dashboard.html',
            user=current_user,
            chats=chats,
            models=[m['id'] for m in available_models],
            is_demo=False
        )
    except Exception as e:
        log.error(f"Dashboard error: {e}")
        return render_template(
            'dashboard.html',
            user=current_user,
            chats=[],
            models=[],
            is_demo=False
        )

# ═══════════════════════════════════════════════════════════════════
# ROUTES - DEMO MODE
# ═══════════════════════════════════════════════════════════════════

@app.route('/demo-login', methods=['POST'])
def demo_login():
    """Auto-login for demo mode (creates temporary session)"""
    try:
        session['is_demo'] = True
        session['demo_session_id'] = get_demo_session_id()
        log.info(f"Demo session created: {session['demo_session_id']}")
        return jsonify({"success": True})
    except Exception as e:
        log.error(f"Demo login error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    """Demo chat endpoint - Qwen 2.5 VL via OpenRouter - SSE Streaming (Vision + Text)"""
    try:
        from flask import Response, stream_with_context
        import json

        ip_address = request.remote_addr

        # Rate limit check
        if not check_demo_rate_limit(ip_address):
            return jsonify({
                "error": "You have reached your demo mode limit (10 messages/hour). Please sign up for unlimited access!"
            }), 429

        data = request.get_json() or {}
        message = data.get('message', '').strip()
        context = data.get('context', [])
        image_data = data.get('image')  # For vision support

        if not message and not image_data:
            return jsonify({"error": "Message or image required"}), 400

        # Check OpenRouter API key
        if not OPENROUTER_API_KEY:
            return jsonify({
                "error": "Demo mode requires OpenRouter API key. Please sign up for full access."
            }), 503

        # Build conversation history with context (last 6 messages)
        messages_history = []

        # Add system message for Qwen 2.5 VL
        messages_history.append({
            "role": "system",
            "content": "You are NexaAI, an intelligent assistant created by Aarav. Always provide well-formatted responses using markdown. Use proper formatting, code blocks, lists, tables, and emphasis to make responses clear and visually appealing. Be helpful, accurate, and concise."
        })

        # Add conversation context (last 6 messages)
        for ctx_msg in context[-6:]:
            if isinstance(ctx_msg, dict) and 'role' in ctx_msg and 'content' in ctx_msg:
                messages_history.append({
                    "role": ctx_msg['role'],
                    "content": ctx_msg['content']
                })

        # Add current user message (with or without image)
        if image_data:
            # Qwen 2.5 VL supports multi-modal input
            messages_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message or "Analyze this image"},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            })
        else:
            # Text-only message
            messages_history.append({
                "role": "user",
                "content": message
            })

        # Always use Qwen 2.5 VL (supports both text and vision)
        model = "qwen/qwen-2.5-vl-7b-instruct:free"
        messages_analyzed = len(messages_history)

        def generate():
            """SSE generator for streaming responses"""
            # Send metadata
            meta_json = json.dumps({
                "messages_analyzed": messages_analyzed,
                "model": model
            })
            yield f"event: meta\ndata: {meta_json}\n\n"

            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": request.headers.get('Referer', 'https://nexa-ai.onrender.com'),
                    "X-Title": "NexaAI Demo"
                }

                payload = {
                    "model": model,
                    "messages": messages_history,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "top_p": 0.9
                }

                resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=90)

                if resp.status_code != 200:
                    error_text = resp.text
                    logger.error(f"OpenRouter API error: {resp.status_code} - {error_text}")
                    
                    error_json = json.dumps({
                        "error": f"Demo API error ({resp.status_code}). Please try again or sign up for full access.",
                        "retryable": True
                    })
                    yield f"event: error\ndata: {error_json}\n\n"
                    return

                full_response = ""
                
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            chunk = line_str[6:].strip()
                            
                            if chunk == "[DONE]":
                                yield f"event: done\ndata: {{}}\n\n"
                                break

                            try:
                                chunk_data = json.loads(chunk)
                                delta = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')

                                if delta:
                                    full_response += delta
                                    token_json = json.dumps({"delta": delta})
                                    yield f"event: token\ndata: {token_json}\n\n"

                            except json.JSONDecodeError:
                                continue

                # Log completion
                logger.info(f"Demo chat completed: {len(full_response)} chars generated")

            except requests.exceptions.Timeout:
                logger.error("Demo chat timeout")
                error_json = json.dumps({
                    "error": "Request timeout. Please try a shorter message or sign up for full access.",
                    "retryable": True
                })
                yield f"event: error\ndata: {error_json}\n\n"

            except requests.exceptions.RequestException as e:
                logger.error(f"Demo chat request error: {str(e)}")
                err_str = str(e).lower()
                
                if any(x in err_str for x in ['quota', 'rate limit', '429', 'too many']):
                    error_json = json.dumps({
                        "error": "Demo limit reached. Sign up for unlimited access!",
                        "retryable": False
                    })
                else:
                    error_json = json.dumps({
                        "error": "Demo service temporarily unavailable. Please try again.",
                        "retryable": True
                    })
                yield f"event: error\ndata: {error_json}\n\n"

            except Exception as e:
                logger.error(f"Demo stream error: {str(e)}")
                error_json = json.dumps({
                    "error": "An unexpected error occurred. Please try again.",
                    "retryable": True
                })
                yield f"event: error\ndata: {error_json}\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Demo chat fatal error: {str(e)}")
        return jsonify({"error": "Server error. Please try again."}), 500

# ═══════════════════════════════════════════════════════════════════
# ROUTES - CHAT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

@app.route('/new-chat', methods=['POST'])
def new_chat():
    """Create new chat - Works for both authenticated and demo users"""
    try:
        is_demo = not current_user.is_authenticated

        chat = Chat(
            user_id=current_user.id if not is_demo else None,
            title='New Chat',
            is_demo=is_demo,
            session_id=get_demo_session_id() if is_demo else None
        )

        db.session.add(chat)
        db.session.commit()

        log.info(f"New chat created: {chat.id} (Demo: {is_demo})")

        return jsonify({
            "success": True,
            "chat_id": chat.id,
            "title": chat.title
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"New chat error: {e}")
        return jsonify({"error": "Failed to create chat"}), 500


@app.route('/get-chats', methods=['GET'])
def get_chats():
    """Get all chats - Works for both authenticated and demo users"""
    try:
        is_demo = not current_user.is_authenticated

        if is_demo:
            chats = Chat.query.filter_by(
                is_demo=True,
                session_id=get_demo_session_id()
            ).order_by(desc(Chat.updated_at)).limit(10).all()
        else:
            chats = Chat.query.filter_by(
                user_id=current_user.id,
                is_demo=False
            ).order_by(desc(Chat.updated_at)).limit(50).all()

        chats_data = [{
            "id": chat.id,
            "title": chat.title,
            "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
            "message_count": len(chat.messages)
        } for chat in chats]

        return jsonify({
            "success": True,
            "chats": chats_data
        })
    except Exception as e:
        log.error(f"Get chats error: {e}")
        return jsonify({"error": "Failed to load chats"}), 500


@app.route('/get-chat/<int:chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get specific chat with messages"""
    try:
        is_demo = not current_user.is_authenticated

        if is_demo:
            chat = Chat.query.filter_by(
                id=chat_id,
                is_demo=True,
                session_id=get_demo_session_id()
            ).first()
        else:
            chat = Chat.query.filter_by(
                id=chat_id,
                user_id=current_user.id
            ).first()

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        messages_data = [{
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "model": msg.model,
            "has_image": msg.has_image,
            "image_url": msg.image_url,
            "has_reasoning": msg.has_reasoning,
            "reasoning_content": msg.reasoning_content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None
        } for msg in chat.messages]

        return jsonify({
            "success": True,
            "chat_id": chat.id,
            "title": chat.title,
            "messages": messages_data
        })
    except Exception as e:
        log.error(f"Get chat error: {e}")
        return jsonify({"error": "Failed to load chat"}), 500


@app.route('/rename-chat/<int:chat_id>', methods=['POST'])
def rename_chat(chat_id):
    """Rename chat"""
    try:
        data = request.get_json() or {}
        new_title = data.get('title', '').strip()

        if not new_title:
            return jsonify({"error": "Title required"}), 400

        is_demo = not current_user.is_authenticated

        if is_demo:
            chat = Chat.query.filter_by(
                id=chat_id,
                is_demo=True,
                session_id=get_demo_session_id()
            ).first()
        else:
            chat = Chat.query.filter_by(
                id=chat_id,
                user_id=current_user.id
            ).first()

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        chat.title = new_title[:200]
        db.session.commit()

        log.info(f"Chat {chat_id} renamed to: {new_title}")

        return jsonify({
            "success": True,
            "chat_id": chat.id,
            "title": chat.title
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Rename chat error: {e}")
        return jsonify({"error": "Failed to rename chat"}), 500


@app.route('/delete-chat/<int:chat_id>', methods=['POST', 'DELETE'])
def delete_chat(chat_id):
    """Delete chat"""
    try:
        is_demo = not current_user.is_authenticated

        if is_demo:
            chat = Chat.query.filter_by(
                id=chat_id,
                is_demo=True,
                session_id=get_demo_session_id()
            ).first()
        else:
            chat = Chat.query.filter_by(
                id=chat_id,
                user_id=current_user.id
            ).first()

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        db.session.delete(chat)
        db.session.commit()

        log.info(f"Chat {chat_id} deleted (Demo: {is_demo})")

        return jsonify({
            "success": True,
            "message": "Chat deleted successfully"
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Delete chat error: {e}")
        return jsonify({"error": "Failed to delete chat"}), 500


@app.route('/chat', methods=['POST'])
def chat_route():
    """Main chat endpoint - Works for both authenticated and demo users"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        model_id = data.get('model', session.get('selected_model', 'deepseek-r1'))
        chat_id = data.get('chatId') or data.get('chat_id')
        image_data = data.get('image')
        deepthink = data.get('deepthink', False)
        web_search = data.get('web', False)

        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        is_demo = not current_user.is_authenticated

        log.info(f"Chat request - model:{model_id}, chat_id:{chat_id}, demo:{is_demo}")

        # Demo mode limitations
        if is_demo:
            return jsonify({
                "error": "Demo mode only supports simple chat. Please sign up for full features!",
                "upgrade_required": True
            }), 403

        # Get or create chat
        if chat_id:
            if is_demo:
                chat = Chat.query.filter_by(
                    id=chat_id,
                    is_demo=True,
                    session_id=get_demo_session_id()
                ).first()
            else:
                chat = Chat.query.filter_by(
                    id=chat_id,
                    user_id=current_user.id
                ).first()

            if not chat:
                return jsonify({"error": "Chat not found"}), 404
        else:
            chat = Chat(
                user_id=current_user.id if not is_demo else None,
                title=message[:50],
                is_demo=is_demo,
                session_id=get_demo_session_id() if is_demo else None
            )
            db.session.add(chat)
            db.session.flush()

        # Check for image generation command
        img_keywords = ['draw', 'generate image', 'create image', 'make image']
        if any(kw in message.lower() for kw in img_keywords):
            img_url = generate_image_url(message)

            db.session.add(Message(
                chat_id=chat.id,
                role='user',
                content=message
            ))

            db.session.add(Message(
                chat_id=chat.id,
                role='assistant',
                content="I've generated the image for you!",
                model='Pollinations AI',
                has_image=True,
                image_url=img_url
            ))

            chat.updated_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                "success": True,
                "chat_id": chat.id,
                "response": "Image generated successfully!",
                "image_url": img_url,
                "title": chat.title
            })

        # Get chat history (last 10 messages)
        history = Message.query.filter_by(
            chat_id=chat.id
        ).order_by(desc(Message.created_at)).limit(10).all()[::-1]

        messages_history = [
            {"role": m.role, "content": m.content}
            for m in history
        ]
        messages_history.append({"role": "user", "content": message})

        # Get model config
        all_models = FREE_MODELS + PREMIUM_MODELS
        model_config = next((m for m in all_models if m['id'] == model_id), FREE_MODELS[0])

        # Call AI model
        ai_response = call_ai_model(model_id, messages_history, image_data)

        # Save messages
        db.session.add(Message(
            chat_id=chat.id,
            role='user',
            content=message
        ))

        db.session.add(Message(
            chat_id=chat.id,
            role='assistant',
            content=ai_response,
            model=model_config['name']
        ))

        # Update chat
        if not chat_id:
            chat.title = message[:50]

        chat.updated_at = datetime.utcnow()

        # Increment DeepSeek counter if needed
        if not is_demo and 'deepseek' in model_id:
            increment_deepseek_count(current_user)

        db.session.commit()

        response_data = {
            "success": True,
            "chat_id": chat.id,
            "response": ai_response,
            "model": model_config['name'],
            "title": chat.title
        }

        if not is_demo and current_user:
            response_data['deepseek_count'] = current_user.deepseek_count

        return jsonify(response_data)

    except Exception as e:
        db.session.rollback()
        log.error(f"Chat error: {e}")
        return jsonify({"error": str(e)[:200]}), 500


@app.route('/set-model', methods=['POST'])
def set_model():
    """Set selected model in session"""
    try:
        data = request.get_json() or {}
        model_id = data.get('model', 'deepseek-r1')
        model_name = data.get('name', 'DeepSeek R1')

        session['selected_model'] = model_id
        session['selected_model_name'] = model_name

        log.info(f"Model selected: {model_id}")

        return jsonify({
            "success": True,
            "model": model_id,
            "name": model_name
        })
    except Exception as e:
        log.error(f"Set model error: {e}")
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════
# ROUTES - MEMORY SYSTEM (For logged-in users)
# ═══════════════════════════════════════════════════════════════════

@app.route('/memories', methods=['GET'])
@login_required
def get_memories():
    """Get all user memories"""
    try:
        memories = Memory.query.filter_by(user_id=current_user.id).all()

        memories_data = [{
            "id": m.id,
            "key": m.key,
            "value": m.value,
            "category": m.category,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None
        } for m in memories]

        return jsonify({
            "success": True,
            "memories": memories_data
        })
    except Exception as e:
        log.error(f"Get memories error: {e}")
        return jsonify({"error": "Failed to load memories"}), 500


@app.route('/memory', methods=['POST'])
@login_required
def add_memory():
    """Add or update memory"""
    try:
        data = request.get_json() or {}
        key = data.get('key', '').strip()
        value = data.get('value', '').strip()
        category = data.get('category', 'general').strip()

        if not key or not value:
            return jsonify({"error": "Key and value required"}), 400

        # Check if memory exists
        memory = Memory.query.filter_by(
            user_id=current_user.id,
            key=key
        ).first()

        if memory:
            memory.value = value
            memory.category = category
            memory.updated_at = datetime.utcnow()
        else:
            memory = Memory(
                user_id=current_user.id,
                key=key,
                value=value,
                category=category
            )
            db.session.add(memory)

        db.session.commit()

        return jsonify({
            "success": True,
            "memory": {
                "id": memory.id,
                "key": memory.key,
                "value": memory.value,
                "category": memory.category
            }
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Add memory error: {e}")
        return jsonify({"error": "Failed to save memory"}), 500


@app.route('/memory/<int:memory_id>', methods=['DELETE'])
@login_required
def delete_memory(memory_id):
    """Delete memory"""
    try:
        memory = Memory.query.filter_by(
            id=memory_id,
            user_id=current_user.id
        ).first()

        if not memory:
            return jsonify({"error": "Memory not found"}), 404

        db.session.delete(memory)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Memory deleted"
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Delete memory error: {e}")
        return jsonify({"error": "Failed to delete memory"}), 500

# ═══════════════════════════════════════════════════════════════════
# ROUTES - USER SETTINGS
# ═══════════════════════════════════════════════════════════════════

@app.route('/settings', methods=['GET'])
@login_required
def get_settings():
    """Get user settings"""
    try:
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()

        if not settings:
            settings = UserSettings(user_id=current_user.id)
            db.session.add(settings)
            db.session.commit()

        return jsonify({
            "success": True,
            "settings": {
                "theme": settings.theme,
                "default_model": settings.default_model,
                "enable_reasoning": settings.enable_reasoning,
                "enable_memory": settings.enable_memory,
                "enable_web_search": settings.enable_web_search,
                "save_chat_history": settings.save_chat_history
            }
        })
    except Exception as e:
        log.error(f"Get settings error: {e}")
        return jsonify({"error": "Failed to load settings"}), 500


@app.route('/settings', methods=['POST'])
@login_required
def update_settings():
    """Update user settings"""
    try:
        data = request.get_json() or {}

        settings = UserSettings.query.filter_by(user_id=current_user.id).first()

        if not settings:
            settings = UserSettings(user_id=current_user.id)
            db.session.add(settings)

        # Update settings
        if 'theme' in data:
            settings.theme = data['theme']
        if 'default_model' in data:
            settings.default_model = data['default_model']
        if 'enable_reasoning' in data:
            settings.enable_reasoning = data['enable_reasoning']
        if 'enable_memory' in data:
            settings.enable_memory = data['enable_memory']
        if 'enable_web_search' in data:
            settings.enable_web_search = data['enable_web_search']
        if 'save_chat_history' in data:
            settings.save_chat_history = data['save_chat_history']

        settings.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Settings updated"
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Update settings error: {e}")
        return jsonify({"error": "Failed to update settings"}), 500

# ═══════════════════════════════════════════════════════════════════
# ROUTES - STATIC FILES
# ═══════════════════════════════════════════════════════════════════

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    if request.path.startswith('/api'):
        return jsonify({"error": "Not found"}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """500 error handler"""
    db.session.rollback()
    log.error(f"Internal error: {e}")
    if request.path.startswith('/api'):
        return jsonify({"error": "Internal server error"}), 500
    return render_template('500.html'), 500


@app.errorhandler(403)
def forbidden(e):
    """403 error handler"""
    if request.path.startswith('/api'):
        return jsonify({"error": "Forbidden"}), 403
    return render_template('403.html'), 403

# ═══════════════════════════════════════════════════════════════════
# API VALIDATION & STARTUP
# ═══════════════════════════════════════════════════════════════════

def validate_api_keys():
    """Validate API keys at startup"""
    issues = []

    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY not set")
    if not OPENROUTER_API_KEY:
        issues.append("OPENROUTER_API_KEY not set")

    if issues:
        log.warning(f"⚠️ API Configuration Issues: {', '.join(issues)}")
        log.warning("Some AI models may not work")
    else:
        log.info("✅ All API keys configured")

    return len(issues) == 0

# Test database on startup
with app.app_context():
    try:
        db.session.execute("SELECT 1")
        log.info("✓ Database connection successful!")
    except Exception as e:
        log.error(f"✗ Database connection failed: {e}")
        log.error("Check your DATABASE_URL in .env file")


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Test database connection
    test_db_connection()

    # Initialize database
    init_db()

    # Validate API keys
    validate_api_keys()

    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    log.info("=" * 70)
    log.info(f"🚀 Starting NexaAI on port {port}")
    log.info(f"📊 Debug mode: {debug}")
    log.info(f"🎭 Demo mode: Available at /demo and /")
    log.info(f"🤖 Demo AI Model: DeepSeek R1 (Reasoning)")
    log.info(f"👁️ Vision Support: Gemini 2.0 Flash (for images)")
    log.info(f"⚡ Demo rate limit: {DEMO_RATE_LIMIT} messages per hour")
    log.info("=" * 70)

    app.run(host='0.0.0.0', port=port, debug=debug)



