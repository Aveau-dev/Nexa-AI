"""
NexaAI - Advanced AI Chat Platform
Complete Fixed Version with ALL Features Working
Author: Aarav
Date: 2026-01-04
"""

import os
import logging
import base64
import urllib.parse
import requests
from datetime import datetime, timedelta
from io import BytesIO

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
from sqlalchemy import desc

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

# Database Configuration
database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')
if database_url:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    log.info("🐘 Using PostgreSQL database")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nexaai.db'
    log.info("📁 Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 10,
    'max_overflow': 20
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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

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
        'description': 'Fast and efficient Gemini Flash model with vision',
        'speed': 'Very Fast',
        'context': '1M tokens'
    },
    {
        'id': 'gpt-3.5-turbo',
        'name': 'ChatGPT 3.5 Turbo',
        'model': 'openai/gpt-3.5-turbo',
        'provider': 'openrouter',
        'vision': False,
        'tier': 'free',
        'rank': 2,
        'description': 'OpenAI GPT-3.5 Turbo - Fast and reliable',
        'speed': 'Fast',
        'context': '16K tokens'
    },
    {
        'id': 'claude-haiku',
        'name': 'Claude 3 Haiku',
        'model': 'anthropic/claude-3-haiku',
        'provider': 'openrouter',
        'vision': True,
        'tier': 'free',
        'rank': 3,
        'description': 'Fast Claude model with vision support',
        'speed': 'Very Fast',
        'context': '200K tokens'
    },
    {
        'id': 'deepseek-chat',
        'name': 'DeepSeek Chat',
        'model': 'deepseek/deepseek-chat',
        'provider': 'openrouter',
        'vision': False,
        'tier': 'free',
        'limit': 50,
        'rank': 4,
        'description': 'Powerful for code & reasoning (50/day free)',
        'speed': 'Fast',
        'context': '64K tokens'
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
        'rank': 5,
        'description': 'Efficient GPT-4 level performance',
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
        'rank': 6,
        'description': 'Advanced multimodal AI with vision',
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
        'rank': 7,
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
        'rank': 8,
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
    def ispremium(self):
        return self.plan in ['pro', 'max']

    def __repr__(self):
        return f'<User {self.email} - {self.plan}>'


class Chat(db.Model):
    __tablename__ = 'chats'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(200), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan', order_by='Message.created_at')

    def __repr__(self):
        return f'<Chat {self.id}: {self.title}>'


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
# LOGIN MANAGER
# ═══════════════════════════════════════════════════════════════════════════

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        log.error(f"Error loading user {user_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_available_models(user):
    """Get list of models available to user based on their plan"""
    plan = user.plan if user.ispremium else 'basic'

    if plan == 'basic':
        return FREE_MODELS
    elif plan == 'pro':
        return FREE_MODELS + PREMIUM_MODELS[:2]  # Include first 2 premium
    else:  # max
        return FREE_MODELS + PREMIUM_MODELS


def check_deepseek_limit(user):
    """Check and reset daily DeepSeek limit"""
    today = datetime.utcnow().strftime('%Y-%m-%d')

    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()

    if user.ispremium:
        return True

    return user.deepseek_count < 50


def increment_deepseek_count(user):
    """Increment daily DeepSeek usage counter"""
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
                img = Image.open(BytesIO(image_bytes))
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
    """Landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            email = (data.get('email') or '').strip().lower()
            password = data.get('password') or ''

            if not email or not password:
                return jsonify(success=False, error="Email and password required"), 400

            user = User.query.filter_by(email=email).first()

            if user and check_password_hash(user.password, password):
                login_user(user, remember=True)
                user.last_login = datetime.utcnow()
                db.session.commit()
                log.info(f"✅ User logged in: {email}")
                return jsonify(success=True, redirect=url_for('dashboard'))

            log.warning(f"⚠️ Failed login attempt: {email}")
            return jsonify(success=False, error="Invalid email or password"), 401

        except Exception as e:
            log.error(f"Login error: {e}")
            return jsonify(success=False, error="Login failed"), 500

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
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

            if User.query.filter_by(email=email).first():
                return jsonify(success=False, error="Email already registered"), 409

            new_user = User(
                email=email,
                password=generate_password_hash(password),
                name=name,
                plan="pro",  # Free Pro trial
                plan_started_at=datetime.utcnow(),
                subscription_status='trial',
                deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
            )

            db.session.add(new_user)
            db.session.commit()

            login_user(new_user, remember=True)
            log.info(f"✅ New user registered: {email} (Pro trial)")

            return jsonify(success=True, redirect=url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            log.error(f"Signup error: {e}")
            return jsonify(success=False, error="Registration failed"), 500

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
        chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).limit(50).all()
        available_models = get_available_models(current_user)

        # Set default model in session
        if 'selected_model' not in session:
            session['selected_model'] = 'gemini-flash'
            session['selected_model_name'] = 'Gemini 2.5 Flash'

        return render_template(
            'dashboard.html',
            user=current_user,
            chats=chats,
            models={m['id']: m for m in available_models}
        )
    except Exception as e:
        log.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', user=current_user, chats=[], models={})


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/set-model', methods=['POST'])
@login_required
def set_model():
    """Set selected model in session"""
    try:
        data = request.get_json() or {}
        model_id = data.get('model', 'gemini-flash')
        model_name = data.get('name', 'Gemini 2.5 Flash')

        session['selected_model'] = model_id
        session['selected_model_name'] = model_name

        log.info(f"User {current_user.email} selected model: {model_id}")
        return jsonify({
            'success': True,
            'model': model_id,
            'name': model_name
        })
    except Exception as e:
        log.error(f"Set model error: {e}")
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - CHAT MANAGEMENT (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/new-chat', methods=['POST'])
@login_required
def new_chat():
    """Create new chat - FIXED"""
    try:
        chat = Chat(
            user_id=current_user.id,
            title='New Chat'
        )
        db.session.add(chat)
        db.session.commit()

        log.info(f"New chat created: {chat.id} for user {current_user.email}")
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
@login_required
def get_chats():
    """Get all chats for user - FIXED"""
    try:
        chats = Chat.query.filter_by(user_id=current_user.id)\
            .order_by(desc(Chat.updated_at))\
            .limit(50)\
            .all()

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
@login_required
def get_chat(chat_id):
    """Get specific chat with messages - FIXED"""
    try:
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
@login_required
def rename_chat(chat_id):
    """Rename chat - FIXED"""
    try:
        data = request.get_json() or {}
        new_title = (data.get('title') or '').strip()

        if not new_title:
            return jsonify({'error': 'Title required'}), 400

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
@login_required
def delete_chat(chat_id):
    """Delete chat - FIXED"""
    try:
        chat = Chat.query.filter_by(
            id=chat_id,
            user_id=current_user.id
        ).first()

        if not chat:
            return jsonify({'error': 'Chat not found'}), 404

        db.session.delete(chat)
        db.session.commit()

        log.info(f"Chat {chat_id} deleted by user {current_user.email}")
        return jsonify({
            'success': True,
            'message': 'Chat deleted successfully'
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Delete chat error: {e}")
        return jsonify({'error': 'Failed to delete chat'}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - CHAT MESSAGING
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    """Main chat endpoint - FIXED"""
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

        log.info(f"Chat request: user={current_user.email}, model={model_id}, chat_id={chat_id}")

        # Get or create chat
        if chat_id:
            chat = Chat.query.filter_by(
                id=chat_id,
                user_id=current_user.id
            ).first()
            if not chat:
                return jsonify({'error': 'Chat not found'}), 404
        else:
            chat = Chat(
                user_id=current_user.id,
                title=message[:50]
            )
            db.session.add(chat)
            db.session.flush()

        # Check for image generation command
        img_keywords = ['draw', 'generate image', 'create image', 'make image']
        if any(kw in message.lower() for kw in img_keywords):
            img_url = generate_image_url(message)

            # Save user message
            db.session.add(Message(
                chat_id=chat.id,
                role='user',
                content=message
            ))

            # Save AI response with image
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

        # Check model access
        if model_config['tier'] != 'free' and not current_user.ispremium:
            return jsonify({
                'error': 'This model requires Ultimate plan. Please upgrade.',
                'upgrade_required': True
            }), 403

        # Check DeepSeek limit
        if 'deepseek' in model_id and not check_deepseek_limit(current_user):
            return jsonify({
                'error': 'Daily DeepSeek limit reached (50/day). Upgrade to Ultimate for unlimited access.',
                'upgrade_required': True,
                'deepseek_count': current_user.deepseek_count
            }), 429

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
            .order_by(Message.created_at)\
            .limit(10)\
            .all()

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
        if 'deepseek' in model_id:
            increment_deepseek_count(current_user)

        db.session.commit()

        return jsonify({
            'success': True,
            'chat_id': chat.id,
            'response': ai_response,
            'model': model_config['name'],
            'title': chat.title,
            'deepseek_count': current_user.deepseek_count
        })

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

        # In production, integrate with Stripe/Razorpay here
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

def init_db():
    """Initialize database"""
    with app.app_context():
        try:
            db.create_all()
            log.info("✅ Database initialized successfully")
        except Exception as e:
            log.error(f"❌ Database initialization failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Initialize database
    init_db()

    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    log.info(f"🚀 Starting NexaAI on port {port}")
    log.info(f"🔧 Debug mode: {debug}")
    log.info(f"📊 Database: {app.config['SQLALCHEMY_DATABASE_URI'][:30]}...")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
