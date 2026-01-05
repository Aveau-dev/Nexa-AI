"""
NexaAI - Advanced AI Chat Platform with Demo Mode
Complete Production Version with Supabase Support & Local AI
Author: Aarav
Date: 2026-01-05
Version: 3.0 - Production Ready with Demo Routes
"""

import os
import logging
import base64
import urllib.parse
import requests
import hashlib
from datetime import datetime, timedelta
from io import BytesIO
from functools import wraps
from collections import defaultdict

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    session, send_from_directory, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import desc, exc

# Google Generative AI
import google.generativeai as genai

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nexaai.log', mode='a')
    ]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nexaai-secret-key-2025-change-in-production')
app.config['JSON_AS_ASCII'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE DATABASE CONFIGURATION - FIXED
# ═══════════════════════════════════════════════════════════════════════════

database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')

if database_url:
    # Fix postgres:// to postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    # ⚠️ CRITICAL: Add SSL mode for Supabase
    if 'supabase' in database_url.lower():
        if '?' not in database_url:
            database_url += '?sslmode=require'
        elif 'sslmode' not in database_url:
            database_url += '&sslmode=require'

    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    log.info(f"🐘 Using Supabase PostgreSQL")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nexaai.db'
    log.info("📁 Using SQLite database (fallback)")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ✅ SUPABASE-OPTIMIZED CONNECTION POOL SETTINGS
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'connect_args': {
        'connect_timeout': 10,
        'options': '-c statement_timeout=30000'
    }
}

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Extensions Initialization
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'
login_manager.login_message = 'Please log in to access this page.'

# ═══════════════════════════════════════════════════════════════════════════
# API KEYS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

log.info("🔑 Configuring API Keys...")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY.strip())
        log.info("✅ Google Generative AI configured")
    except Exception as e:
        log.error(f"❌ Google AI configuration failed: {e}")
        GOOGLE_API_KEY = None
else:
    log.warning("⚠️ GOOGLE_API_KEY not set")

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()
    log.info("✅ OpenRouter API configured")
else:
    log.warning("⚠️ OPENROUTER_API_KEY not set")

# Nexa AI Configuration (Local AI)
NEXA_API_URL = os.getenv('NEXA_API_URL', 'http://localhost:8000/v1')
NEXA_ENABLED = os.getenv('NEXA_ENABLED', 'false').lower() == 'true'

if NEXA_ENABLED:
    log.info(f"✅ Nexa AI Local Server configured: {NEXA_API_URL}")
else:
    log.info("⚠️ Nexa AI disabled (set NEXA_ENABLED=true in .env to enable)")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITING FOR DEMO MODE
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# AI MODELS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

FREE_MODELS = [
    {
        'id': 'gemini-flash',
        'name': 'Gemini 2.5 Flash',
        'model': 'gemini-2.0-flash-exp',
        'provider': 'google',
        'vision': True,
        'tier': 'free',
        'rank': 1,
        'description': '⚡ Fast and efficient with vision',
        'speed': 'Very Fast',
        'context': '1M tokens',
        'demo_enabled': True
    },
    {
        'id': 'nexa-llama',
        'name': 'Nexa Llama 3.2 (Local)',
        'model': 'llama-3.2-1b-instruct',
        'provider': 'nexa',
        'vision': False,
        'tier': 'free',
        'rank': 2,
        'description': '🖥️ Private local AI - No API key needed',
        'speed': 'Fast',
        'context': '128K tokens',
        'requires_nexa': True,
        'demo_enabled': False
    },
    {
        'id': 'gpt-3.5-turbo',
        'name': 'ChatGPT 3.5 Turbo',
        'model': 'openai/gpt-3.5-turbo',
        'provider': 'openrouter',
        'vision': False,
        'tier': 'free',
        'rank': 3,
        'description': '🤖 OpenAI GPT-3.5 - Fast and reliable',
        'speed': 'Fast',
        'context': '16K tokens',
        'demo_enabled': False
    },
    {
        'id': 'claude-haiku',
        'name': 'Claude 3 Haiku',
        'model': 'anthropic/claude-3-haiku',
        'provider': 'openrouter',
        'vision': True,
        'tier': 'free',
        'rank': 4,
        'description': '🎨 Fast Claude with vision support',
        'speed': 'Very Fast',
        'context': '200K tokens',
        'demo_enabled': False
    },
    {
        'id': 'deepseek-chat',
        'name': 'DeepSeek Chat',
        'model': 'deepseek/deepseek-chat',
        'provider': 'openrouter',
        'vision': False,
        'tier': 'free',
        'limit': 50,
        'rank': 5,
        'description': '💡 Powerful for code & reasoning (50/day)',
        'speed': 'Fast',
        'context': '64K tokens',
        'demo_enabled': False
    }
]

PREMIUM_MODELS = [
    {
        'id': 'gpt-4o-mini',
        'name': 'GPT-4o Mini',
        'model': 'openai/gpt-4o-mini',
        'provider': 'openrouter',
        'vision': True,
        'tier': 'pro',
        'rank': 6,
        'description': '⚡ Efficient GPT-4 level performance',
        'speed': 'Fast',
        'context': '128K tokens'
    },
    {
        'id': 'gemini-pro',
        'name': 'Gemini 1.5 Pro',
        'model': 'gemini-1.5-pro',
        'provider': 'google',
        'vision': True,
        'tier': 'pro',
        'rank': 7,
        'description': '🎯 Advanced multimodal AI',
        'speed': 'Medium',
        'context': '2M tokens'
    },
    {
        'id': 'gpt-4o',
        'name': 'GPT-4o',
        'model': 'openai/gpt-4o',
        'provider': 'openrouter',
        'vision': True,
        'tier': 'max',
        'rank': 8,
        'description': '🏆 Most capable GPT-4 model',
        'speed': 'Medium',
        'context': '128K tokens'
    },
    {
        'id': 'claude-sonnet',
        'name': 'Claude 3.5 Sonnet',
        'model': 'anthropic/claude-3.5-sonnet',
        'provider': 'openrouter',
        'vision': True,
        'tier': 'max',
        'rank': 9,
        'description': '🏆 Best for coding & analysis',
        'speed': 'Medium',
        'context': '200K tokens'
    }
]

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)

    # Plan Management
    plan = db.Column(db.String(20), default='basic', nullable=False)
    plan_started_at = db.Column(db.DateTime, default=datetime.utcnow)
    subscription_status = db.Column(db.String(20), default='none')

    # Usage Limits
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10), default='')

    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

    @property
    def is_premium(self):
        return self.plan in ['pro', 'max']

    def __repr__(self):
        return f'<User {self.email} - {self.plan}>'


class Chat(db.Model):
    __tablename__ = 'chats'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    title = db.Column(db.String(200), default='New Chat')
    is_demo = db.Column(db.Boolean, default=False)
    session_id = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan', order_by='Message.created_at')

    def __repr__(self):
        return f'<Chat {self.id}: {self.title} (Demo: {self.is_demo})>'


class Message(db.Model):
    __tablename__ = 'messages'

    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(200), nullable=True)

    has_image = db.Column(db.Boolean, default=False)
    image_url = db.Column(db.String(1000), nullable=True)
    image_data = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@app.errorhandler(exc.SQLAlchemyError)
def handle_db_error(error):
    """Handle all database errors"""
    log.error(f"❌ Database error: {error}")
    db.session.rollback()

    error_msg = str(error).lower()

    if 'timeout' in error_msg or 'timed out' in error_msg:
        return jsonify({'error': 'Database connection timeout. Please try again.'}), 503
    elif 'ssl' in error_msg:
        return jsonify({'error': 'Database SSL connection failed. Please contact support.'}), 500
    elif 'authentication' in error_msg or 'password' in error_msg:
        return jsonify({'error': 'Database authentication failed. Please contact support.'}), 500
    elif 'connection' in error_msg:
        return jsonify({'error': 'Cannot connect to database. Please try again later.'}), 503
    else:
        return jsonify({'error': 'Database error occurred. Please try again.'}), 500


@app.teardown_appcontext
def shutdown_session(exception=None):
    """Remove database session after each request"""
    try:
        if exception:
            db.session.rollback()
        db.session.remove()
    except Exception as e:
        log.error(f"Session cleanup error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# LOGIN MANAGER
# ═══════════════════════════════════════════════════════════════════════════

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        log.error(f"Error loading user {user_id}: {e}")
        db.session.rollback()
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_demo_session_id():
    """Get or create demo session ID"""
    if 'demo_session_id' not in session:
        session['demo_session_id'] = hashlib.md5(
            f"{request.remote_addr}_{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()
    return session['demo_session_id']


def get_available_models(user=None):
    """Get list of models available to user based on their plan"""
    if user and user.is_premium:
        plan = user.plan
    else:
        plan = 'basic'

    # Filter out Nexa models if not enabled
    available_free = [m for m in FREE_MODELS if not m.get('requires_nexa') or NEXA_ENABLED]

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


# ═══════════════════════════════════════════════════════════════════════════
# AI MODEL API CALLS
# ═══════════════════════════════════════════════════════════════════════════

def call_nexa_ai(model_path, messages, timeout=60):
    """Call Nexa AI Local Server (OpenAI-compatible API)"""
    try:
        if not NEXA_ENABLED:
            raise Exception("Nexa AI is not enabled. Set NEXA_ENABLED=true in .env")

        headers = {"Content-Type": "application/json"}

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

        payload = {
            "model": model_path,
            "messages": formatted_msgs,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        response = requests.post(
            f"{NEXA_API_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )

        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            error_msg = f"Nexa AI error: {response.status_code}"
            log.error(error_msg)
            raise Exception(error_msg)

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Nexa AI local server. Make sure it's running."
    except requests.exceptions.Timeout:
        return "Error: Nexa AI request timed out."
    except Exception as e:
        log.error(f"Nexa AI error: {str(e)}")
        return f"Error: {str(e)[:150]}"


def call_google_gemini(model_path, messages, image_data=None, timeout=90):
    """Call Google Gemini API"""
    try:
        if not GOOGLE_API_KEY:
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

        if "API" in error_msg.upper() or "KEY" in error_msg.upper():
            return "Error: Invalid API key configuration."
        elif "TIMEOUT" in error_msg.upper():
            return "Error: Request timed out. Try a shorter prompt."
        elif "QUOTA" in error_msg.upper() or "RATE" in error_msg.upper():
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
        elif provider == 'nexa':
            return call_nexa_ai(model_path, messages)
        else:
            return f"Error: Unknown provider '{provider}'"
    except Exception as e:
        log.error(f"AI model call failed: {e}")
        return f"Error calling AI: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - PUBLIC
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - DEMO MODE (NO LOGIN REQUIRED)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/demo-login', methods=['POST'])
def demo_login():
    """Auto-login for demo mode (creates temporary session)"""
    try:
        session['is_demo'] = True
        session['demo_session_id'] = get_demo_session_id()
        log.info(f"Demo session created: {session['demo_session_id']}")
        return jsonify({'success': True})
    except Exception as e:
        log.error(f"Demo login error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    """Demo chat endpoint - Rate limited, Gemini only"""
    try:
        # Check rate limit
        ip_address = request.remote_addr
        if not check_demo_rate_limit(ip_address):
            return jsonify({
                'error': '⏰ Demo rate limit reached (10 messages/hour). Please sign up for unlimited access!'
            }), 429

        data = request.get_json() or {}
        message = (data.get('message') or '').strip()

        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Limit message length for demo
        if len(message) > 500:
            message = message[:500]
            log.info(f"Demo message truncated to 500 chars")

        # Only use Gemini Flash for demo (fastest & free)
        model_config = next((m for m in FREE_MODELS if m['id'] == 'gemini-flash'), None)

        if not model_config or not GOOGLE_API_KEY:
            return jsonify({
                'error': 'Demo mode temporarily unavailable. Please try again later or sign up for full access.'
            }), 503

        # Simple conversation history (last message only for demo)
        messages_history = [{'role': 'user', 'content': message}]

        # Call Gemini API
        ai_response = call_google_gemini(
            model_config['model'],
            messages_history,
            timeout=30  # Shorter timeout for demo
        )

        log.info(f"Demo chat completed for IP: {ip_address}")

        return jsonify({
            'success': True,
            'response': ai_response,
            'model': model_config['name']
        })

    except Exception as e:
        log.error(f"Demo chat error: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login - FIXED"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            email = (data.get('email') or '').strip().lower()
            password = data.get('password') or ''

            if not email or not password:
                return jsonify(success=False, error="Email and password required"), 400

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
                return jsonify(success=True, redirect=url_for('dashboard'))

            log.warning(f"⚠️ Failed login attempt: {email}")
            return jsonify(success=False, error="Invalid email or password"), 401

        except exc.OperationalError as e:
            db.session.rollback()
            log.error(f"❌ Database connection error during login: {e}")
            return jsonify(success=False, error="Database connection error. Please try again."), 503

        except Exception as e:
            db.session.rollback()
            log.error(f"❌ Login error: {e}")
            return jsonify(success=False, error="Login failed. Please try again."), 500

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration - FIXED"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            email = (data.get('email') or '').strip().lower()
            password = data.get('password') or ''
            name = (data.get('name') or '').strip()

            if not email or not password or not name:
                return jsonify(success=False, error="All fields required"), 400

            if len(password) < 6:
                return jsonify(success=False, error="Password must be at least 6 characters"), 400

            existing_user = db.session.query(User).filter_by(email=email).first()
            if existing_user:
                return jsonify(success=False, error="Email already registered"), 409

            new_user = User(
                email=email,
                password=generate_password_hash(password),
                name=name,
                plan="pro",
                plan_started_at=datetime.utcnow(),
                subscription_status='trial',
                deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
            )

            db.session.add(new_user)

            try:
                db.session.commit()
            except exc.IntegrityError:
                db.session.rollback()
                return jsonify(success=False, error="Email already registered"), 409
            except exc.OperationalError as e:
                db.session.rollback()
                log.error(f"❌ Database connection error during signup: {e}")
                return jsonify(success=False, error="Database connection error. Please try again."), 503

            login_user(new_user, remember=True)
            log.info(f"✅ New user registered: {email} (Pro trial)")

            return jsonify(success=True, redirect=url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            log.error(f"❌ Signup error: {e}")
            return jsonify(success=False, error="Registration failed. Please try again."), 500

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    log.info(f"User logged out: {current_user.email}")
    logout_user()
    return redirect(url_for('index'))


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

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
            session['selected_model'] = 'gemini-flash'
            session['selected_model_name'] = 'Gemini 2.5 Flash'

        return render_template(
            'dashboard.html',
            user=current_user,
            chats=chats,
            models={m['id']: m for m in available_models},
            is_demo=False
        )
    except Exception as e:
        log.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', user=current_user, chats=[], models={}, is_demo=False)


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/set-model', methods=['POST'])
def set_model():
    """Set selected model in session"""
    try:
        data = request.get_json() or {}
        model_id = data.get('model', 'gemini-flash')
        model_name = data.get('name', 'Gemini 2.5 Flash')

        session['selected_model'] = model_id
        session['selected_model_name'] = model_name

        log.info(f"Model selected: {model_id}")
        return jsonify({
            'success': True,
            'model': model_id,
            'name': model_name
        })
    except Exception as e:
        log.error(f"Set model error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - CHAT MANAGEMENT (WORKS FOR BOTH AUTH & DEMO)
# ═══════════════════════════════════════════════════════════════════════════

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
            'success': True,
            'chat_id': chat.id,
            'title': chat.title
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"New chat error: {e}")
        return jsonify({'error': 'Failed to create chat'}), 500


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
            'id': chat.id,
            'title': chat.title,
            'updated_at': chat.updated_at.isoformat() if chat.updated_at else None,
            'message_count': len(chat.messages)
        } for chat in chats]

        return jsonify({
            'success': True,
            'chats': chats_data
        })
    except Exception as e:
        log.error(f"Get chats error: {e}")
        return jsonify({'error': 'Failed to load chats'}), 500


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
            return jsonify({'error': 'Chat not found'}), 404

        messages_data = [{
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'model': msg.model,
            'has_image': msg.has_image,
            'image_url': msg.image_url,
            'created_at': msg.created_at.isoformat() if msg.created_at else None
        } for msg in chat.messages]

        return jsonify({
            'success': True,
            'chat_id': chat.id,
            'title': chat.title,
            'messages': messages_data
        })
    except Exception as e:
        log.error(f"Get chat error: {e}")
        return jsonify({'error': 'Failed to load chat'}), 500


@app.route('/rename-chat/<int:chat_id>', methods=['POST'])
def rename_chat(chat_id):
    """Rename chat"""
    try:
        data = request.get_json() or {}
        new_title = (data.get('title') or '').strip()

        if not new_title:
            return jsonify({'error': 'Title required'}), 400

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
            return jsonify({'error': 'Chat not found'}), 404

        chat.title = new_title[:200]
        db.session.commit()

        log.info(f"Chat {chat_id} renamed to: {new_title}")
        return jsonify({
            'success': True,
            'chat_id': chat.id,
            'title': chat.title
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Rename chat error: {e}")
        return jsonify({'error': 'Failed to rename chat'}), 500


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
            return jsonify({'error': 'Chat not found'}), 404

        db.session.delete(chat)
        db.session.commit()

        log.info(f"Chat {chat_id} deleted (Demo: {is_demo})")
        return jsonify({
            'success': True,
            'message': 'Chat deleted successfully'
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Delete chat error: {e}")
        return jsonify({'error': 'Failed to delete chat'}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - CHAT MESSAGING (WORKS FOR BOTH AUTH & DEMO)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/chat', methods=['POST'])
def chat_route():
    """Main chat endpoint - Works for both authenticated and demo users"""
    try:
        data = request.get_json() or {}
        message = (data.get('message') or '').strip()
        model_id = data.get('model', session.get('selected_model', 'gemini-flash'))
        chat_id = data.get('chat_id') or data.get('chatid')
        image_data = data.get('image')
        deepthink = data.get('deepthink', False)
        web_search = data.get('web', False)

        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        is_demo = not current_user.is_authenticated

        log.info(f"Chat request: model={model_id}, chat_id={chat_id}, demo={is_demo}")

        # Demo mode limitations
        if is_demo:
            return jsonify({
                'error': '⚠️ Demo mode only supports simple chat. Please sign up for full features!',
                'upgrade_required': True
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
                return jsonify({'error': 'Chat not found'}), 404
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
                content="I've generated the image for you! 🎨",
                model='Pollinations AI',
                has_image=True,
                image_url=img_url
            ))

            chat.updated_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                'success': True,
                'chat_id': chat.id,
                'response': "Image generated successfully!",
                'image_url': img_url,
                'title': chat.title
            })

        # Get model config
        all_models = FREE_MODELS + PREMIUM_MODELS
        model_config = next((m for m in all_models if m['id'] == model_id), None)

        if not model_config:
            return jsonify({'error': 'Invalid model selected'}), 400

        # Check model access for authenticated users
        if not is_demo and model_config['tier'] != 'free' and not current_user.is_premium:
            return jsonify({
                'error': 'This model requires Ultimate plan. Please upgrade.',
                'upgrade_required': True
            }), 403

        # Check DeepSeek limit
        if 'deepseek' in model_id:
            if is_demo:
                return jsonify({
                    'error': 'DeepSeek requires an account. Please sign up!',
                    'upgrade_required': True
                }), 403
            elif not check_deepseek_limit(current_user):
                return jsonify({
                    'error': 'Daily DeepSeek limit reached (50/day). Upgrade to Ultimate for unlimited access.',
                    'upgrade_required': True,
                    'deepseek_count': current_user.deepseek_count
                }), 429

        # Check Nexa AI availability
        if model_config.get('requires_nexa') and not NEXA_ENABLED:
            return jsonify({
                'error': 'Nexa AI local server is not running. Start it with: nexa server start',
                'nexa_setup_required': True
            }), 503

        # Save user message
        user_msg = Message(
            chat_id=chat.id,
            role='user',
            content=message,
            model=model_id,
            has_image=bool(image_data),
            image_data=image_data
        )
        db.session.add(user_msg)

        # Get chat history (last 10 messages)
        history = Message.query.filter_by(chat_id=chat.id)\
            .order_by(desc(Message.created_at))\
            .limit(10)\
            .all()[::-1]

        messages_history = [
            {'role': m.role, 'content': m.content}
            for m in history
        ]
        messages_history.append({'role': 'user', 'content': message})

        # Call AI
        ai_response = call_ai_model(model_id, messages_history, image_data)

        # Save AI response
        ai_msg = Message(
            chat_id=chat.id,
            role='assistant',
            content=ai_response,
            model=model_config['name']
        )
        db.session.add(ai_msg)

        # Update chat
        if not chat_id:
            chat.title = message[:50]
        chat.updated_at = datetime.utcnow()

        # Increment DeepSeek counter if needed
        if not is_demo and 'deepseek' in model_id:
            increment_deepseek_count(current_user)

        db.session.commit()

        response_data = {
            'success': True,
            'chat_id': chat.id,
            'response': ai_response,
            'model': model_config['name'],
            'title': chat.title
        }

        if not is_demo and current_user:
            response_data['deepseek_count'] = current_user.deepseek_count

        return jsonify(response_data)

    except Exception as e:
        db.session.rollback()
        log.error(f"Chat error: {e}")
        return jsonify({'error': str(e)[:200]}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - SETTINGS & CHECKOUT
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/checkout')
@login_required
def checkout():
    """Checkout page for upgrades"""
    return render_template('checkout.html', user=current_user)


@app.route('/upgrade', methods=['POST'])
@login_required
def upgrade():
    """Handle upgrade (payment integration placeholder)"""
    try:
        data = request.get_json() or {}
        plan = data.get('plan', 'pro')

        current_user.plan = plan
        current_user.subscription_status = 'active'
        db.session.commit()

        log.info(f"User {current_user.email} upgraded to {plan}")
        return jsonify({
            'success': True,
            'plan': plan,
            'message': f'Successfully upgraded to {plan.upper()}!'
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Upgrade error: {e}")
        return jsonify({'error': 'Upgrade failed'}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - STATIC FILES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """500 error handler"""
    db.session.rollback()
    log.error(f"Internal error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


@app.errorhandler(403)
def forbidden(e):
    """403 error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Forbidden'}), 403
    return render_template('403.html'), 403


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def test_db_connection():
    """Test database connection at startup"""
    try:
        with app.app_context():
            db.session.execute('SELECT 1')
            log.info("✅ Database connection successful")
            return True
    except Exception as e:
        log.error(f"❌ Database connection failed: {e}")
        log.error("Please check your DATABASE_URL in .env file")
        return False


def init_db():
    """Initialize database"""
    with app.app_context():
        try:
            db.create_all()
            log.info("✅ Database initialized successfully")
        except Exception as e:
            log.error(f"❌ Database initialization failed: {e}")


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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Test database connection first
    if not test_db_connection():
        log.error("⚠️ Starting app without database connection!")
        log.error("Login/Signup will NOT work until DATABASE_URL is fixed")

    # Initialize database tables
    init_db()

    # Validate API keys
    validate_api_keys()

    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    log.info(f"🚀 Starting NexaAI on port {port}")
    log.info(f"🔧 Debug mode: {debug}")
    log.info(f"🎭 Demo mode: Available at /demo and /")
    log.info(f"🤖 Nexa AI Local: {'Enabled' if NEXA_ENABLED else 'Disabled'}")
    log.info(f"📊 Demo rate limit: {DEMO_RATE_LIMIT} messages per hour")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
