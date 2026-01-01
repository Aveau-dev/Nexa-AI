"""
NexaAI - Advanced AI Chat Platform
Fixed and working version with all features
Author: Aarav
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
from sqlalchemy import desc, text as sql_text

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
# PLAN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PLANS = {
    "basic": {
        "name": "Basic",
        "display_name": "Basic (Free)",
        "description": "Access to free AI models only. No advanced features.",
        "price_monthly": 0,
        "allowed_model_types": {"free"},
        "allow_deepthink": False,
        "allow_web_scraper": False,
        "deepseek_daily_limit": 50,
        "features": [
            "4 Free AI Models",
            "Basic Chat Features",
            "Image Generation (Basic)",
            "Chat History",
            "50 DeepSeek messages/day"
        ]
    },
    "pro": {
        "name": "Pro",
        "display_name": "Pro (Free 6 Months)",
        "description": "Free for 6 months. Access to Pro models, DeepThink & Web Scraper.",
        "price_monthly": 1999,
        "allowed_model_types": {"free", "pro"},
        "allow_deepthink": True,
        "allow_web_scraper": True,
        "trial_months": 6,
        "deepseek_daily_limit": None,
        "features": [
            "All Free Models",
            "2 Premium Pro Models",
            "DeepThink AI Reasoning",
            "Web Scraper",
            "Unlimited DeepSeek",
            "Priority Support"
        ]
    },
    "max": {
        "name": "Max",
        "display_name": "Max (Premium)",
        "description": "Top 5 AI models, Pro image generation, all features unlocked.",
        "price_monthly": 4999,
        "allowed_model_types": {"free", "pro", "max"},
        "allow_deepthink": True,
        "allow_web_scraper": True,
        "deepseek_daily_limit": None,
        "features": [
            "Top 5 Ranked AI Models",
            "Advanced Image Generation",
            "DeepThink AI Reasoning",
            "Web Scraper with AI Analysis",
            "Unlimited Everything",
            "24/7 Priority Support",
            "Early Access to New Models"
        ]
    }
}

# AI Models Configuration
FREE_MODELS = [
    {'id': 'gemini-flash', 'name': 'Gemini 2.5 Flash', 'model': 'gemini-2.0-flash-exp', 'provider': 'google', 'vision': True, 'tier': 'free', 'rank': 4, 'description': 'Fast and efficient Gemini Flash model with vision capabilities', 'speed': 'Very Fast', 'context': '1M tokens'},
    {'id': 'gpt-3.5-turbo', 'name': 'ChatGPT 3.5 Turbo', 'model': 'openai/gpt-3.5-turbo', 'provider': 'openrouter', 'vision': False, 'tier': 'free', 'rank': 5, 'description': 'OpenAI GPT-3.5 Turbo - Fast and reliable', 'speed': 'Fast', 'context': '16K tokens'},
    {'id': 'claude-haiku', 'name': 'Claude 3 Haiku', 'model': 'anthropic/claude-3-haiku', 'provider': 'openrouter', 'vision': True, 'tier': 'free', 'rank': 6, 'description': 'Fast Claude model with vision support', 'speed': 'Very Fast', 'context': '200K tokens'},
    {'id': 'deepseek-chat', 'name': 'DeepSeek Chat', 'model': 'deepseek/deepseek-chat', 'provider': 'openrouter', 'vision': False, 'tier': 'free', 'limit': 50, 'rank': 7, 'description': 'Powerful for code & reasoning (50/day free, unlimited for Pro+)', 'speed': 'Fast', 'context': '64K tokens'}
]

PREMIUM_MODELS = [
    {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'model': 'openai/gpt-4o-mini', 'provider': 'openrouter', 'vision': True, 'tier': 'pro', 'rank': 3, 'description': 'Efficient GPT-4 level performance', 'speed': 'Fast', 'context': '128K tokens'},
    {'id': 'gemini-pro', 'name': 'Gemini 1.5 Pro', 'model': 'gemini-1.5-pro', 'provider': 'google', 'vision': True, 'tier': 'pro', 'rank': 2, 'description': 'Advanced multimodal AI with vision', 'speed': 'Medium', 'context': '2M tokens'},
    {'id': 'gpt-4o', 'name': 'GPT-4o', 'model': 'openai/gpt-4o', 'provider': 'openrouter', 'vision': True, 'tier': 'max', 'rank': 1, 'description': '🏆 #1 Most capable GPT-4 model', 'speed': 'Medium', 'context': '128K tokens'},
    {'id': 'claude-sonnet', 'name': 'Claude 3.5 Sonnet', 'model': 'anthropic/claude-3.5-sonnet', 'provider': 'openrouter', 'vision': True, 'tier': 'max', 'rank': 1, 'description': '🏆 #1 Best for coding & analysis', 'speed': 'Medium', 'context': '200K tokens'},
    {'id': 'deepseek-r1', 'name': 'DeepSeek R1', 'model': 'deepseek/deepseek-r1', 'provider': 'openrouter', 'vision': False, 'tier': 'max', 'rank': 2, 'description': 'Advanced reasoning model', 'speed': 'Slow', 'context': '64K tokens'}
]

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
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
    settings = db.relationship('UserSettings', backref='user', uselist=False, cascade='all, delete-orphan')
    
    @property
    def ispremium(self):
        """Legacy compatibility"""
        return self.plan in ['pro', 'max']
    
    def __repr__(self):
        return f'<User {self.email} - {self.plan}>'


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
    image_url = db.Column(db.String(1000), nullable=True)
    image_data = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'


class UserSettings(db.Model):
    """Store user preferences in database"""
    __tablename__ = 'user_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True, index=True)
    
    # UI Settings
    compact_mode = db.Column(db.Boolean, default=False)
    enter_to_send = db.Column(db.Boolean, default=True)
    theme = db.Column(db.String(20), default='dark')
    
    # Feature Settings
    memory_enabled = db.Column(db.Boolean, default=False)
    memory_text = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserSettings for user {self.user_id}>'


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

def get_user_plan(user):
    """Get effective user plan with trial expiration check"""
    plan_key = user.plan or "basic"
    cfg = PLANS.get(plan_key, PLANS["basic"])
    
    if plan_key == "pro" and user.plan_started_at:
        trial_months = cfg.get("trial_months", 6)
        trial_end = user.plan_started_at + timedelta(days=trial_months * 30)
        
        if datetime.utcnow() > trial_end:
            if not user.subscription_status or user.subscription_status != 'active':
                log.info(f"Pro trial expired for user {user.email}, downgrading to basic")
                user.plan = "basic"
                user.subscription_status = 'expired'
                db.session.commit()
                cfg = PLANS["basic"]
    
    return cfg


def check_deepseek_limit(user):
    """Check and reset daily DeepSeek limit"""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    
    plan_cfg = get_user_plan(user)
    limit = plan_cfg.get('deepseek_daily_limit')
    
    if limit is None:
        return True
    
    return user.deepseek_count < limit


def increment_deepseek_count(user):
    """Increment daily DeepSeek usage counter"""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    if user.deepseek_date != today:
        user.deepseek_count = 1
        user.deepseek_date = today
    else:
        user.deepseek_count += 1
    
    db.session.commit()


def get_available_models(user):
    """Get list of models available to user based on their plan"""
    plan_cfg = get_user_plan(user)
    allowed_tiers = plan_cfg['allowed_model_types']
    
    available = []
    all_models = FREE_MODELS + PREMIUM_MODELS
    
    for model in all_models:
        tier = model.get('tier', 'free')
        if tier in allowed_tiers:
            available.append(model)
    
    available.sort(key=lambda x: x.get('rank', 99))
    
    return available


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compress_image(image_path, max_size=(1024, 1024), quality=85):
    """Compress and resize uploaded images"""
    try:
        img = Image.open(image_path)
        
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=quality)
        log.info(f"Compressed image: {image_path}")
    except Exception as e:
        log.error(f"Image compression failed: {e}")


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


def extract_text_content(content):
    """Extract text from message content"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
    return str(content)


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
            text = extract_text_content(last_msg.get('content', ''))
            if text:
                content_parts.append(text)
        
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
            content = extract_text_content(msg.get('content', ''))
            formatted_msgs.append({
                "role": msg.get('role', 'user'),
                "content": content
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
                plan="pro",
                plan_started_at=datetime.utcnow(),
                subscription_status='trial',
                deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
            )
            
            db.session.add(new_user)
            db.session.flush()
            
            # Create default settings
            settings = UserSettings(user_id=new_user.id)
            db.session.add(settings)
            
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
        chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
        plan_cfg = get_user_plan(current_user)
        available_models = get_available_models(current_user)
        
        # Set default model in session
        if 'selected_model' not in session:
            session['selected_model'] = 'gemini-flash'
            session['selected_model_name'] = 'Gemini 2.5 Flash'
        
        return render_template(
            'dashboard.html',
            user=current_user,
            chats=chats,
            models={m['id']: m for m in available_models},
            plan=plan_cfg,
            plans_config=PLANS
        )
    except Exception as e:
        log.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', user=current_user, chats=[], models={}, plan=PLANS['basic'], plans_config=PLANS)


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - VIEW LOADER (CRITICAL - MISSING IN ORIGINAL)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/view/<view_name>')
@login_required
def load_view(view_name):
    """Load view HTML for router.js - THIS WAS MISSING!"""
    allowed_views = ['chat', 'files', 'memory', 'projects', 'canvas', 'voice', 'settings']
    
    if view_name not in allowed_views:
        view_name = 'chat'
    
    try:
        template_name = f'views/{view_name}.html'
        return render_template(template_name, user=current_user, plan=get_user_plan(current_user))
    except Exception as e:
        log.error(f"View load error for {view_name}: {e}")
        return f'<div class="welcome-section"><h1>View not found: {view_name}</h1></div>', 404


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - MODEL SELECTION (CRITICAL - MISSING IN ORIGINAL)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/set-model', methods=['POST'])
@login_required
def set_model():
    """Set selected model in session - THIS WAS MISSING!"""
    try:
        data = request.get_json() or {}
        model_id = data.get('model', 'gemini-flash')
        
        # Find model name
        all_models = FREE_MODELS + PREMIUM_MODELS
        model_config = next((m for m in all_models if m['id'] == model_id), None)
        
        if model_config:
            session['selected_model'] = model_id
            session['selected_model_name'] = model_config['name']
            log.info(f"User {current_user.email} selected model: {model_id}")
            return jsonify({'success': True, 'model': model_id, 'name': model_config['name']})
        
        return jsonify({'error': 'Invalid model'}), 400
    except Exception as e:
        log.error(f"Set model error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - CHAT
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    """Main chat endpoint"""
    try:
        data = request.get_json() or {}
        message = (data.get('message') or '').strip()
        model_id = data.get('model', session.get('selected_model', 'gemini-flash'))
        chat_id = data.get('chatid') or data.get('chat_id')  # Support both naming conventions
        image_data = data.get('image')
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        log.info(f"Chat request: user={current_user.email}, model={model_id}")
        
        # Get or create chat
        if chat_id:
            chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
            if not chat:
                return jsonify({'error': 'Chat not found'}), 404
        else:
            chat = Chat(user_id=current_user.id, title=message[:50])
            db.session.add(chat)
            db.session.flush()
        
        # Check for image generation command
        img_keywords = ['draw', 'generate image', 'create image', 'make image']
        if any(kw in message.lower() for kw in img_keywords):
            img_url = generate_image_url(message)
            
            db.session.add(Message(chat_id=chat.id, role='user', content=message))
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
                'chatid': chat.id,
                'response': "Image generated successfully!",
                'image_url': img_url,
                'chattitle': chat.title
            })
        
        # Get model config and check permissions
        all_models = FREE_MODELS + PREMIUM_MODELS
        model_config = next((m for m in all_models if m['id'] == model_id), None)
        
        if not model_config:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        plan_cfg = get_user_plan(current_user)
        allowed_tiers = plan_cfg['allowed_model_types']
        
        if model_config['tier'] not in allowed_tiers:
            return jsonify({
                'error': f'Your {plan_cfg["name"]} plan cannot access this model. Please upgrade.',
                'upgrade_required': True
            }), 403
        
        # Check DeepSeek limit
        if 'deepseek' in model_id and not check_deepseek_limit(current_user):
            remaining = plan_cfg.get('deepseek_daily_limit', 50) - current_user.deepseek_count
            return jsonify({
                'error': 'Daily DeepSeek limit reached (50/day). Upgrade to Pro for unlimited access.',
                'upgrade_required': True,
                'deepseekremaining': max(0, remaining)
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
        
        # Get chat history
        history = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
        messages_history = [{'role': m.role, 'content': m.content} for m in history[-10:]]
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
        
        # Calculate DeepSeek remaining
        plan_cfg = get_user_plan(current_user)
        limit = plan_cfg.get('deepseek_daily_limit')
        deepseek_remaining = (limit - current_user.deepseek_count) if limit else 999
        
        return jsonify({
            'success': True,
            'chatid': chat.id,
            'response': ai_response,
            'model': model_config['name'],
            'chattitle': chat.title,
            'deepseekremaining': deepseek_remaining
        })
    
    except Exception as e:
        db.session.rollback()
        log.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    """Create new chat"""
    try:
        chat = Chat(user_id=current_user.id, title='New Chat')
        db.session.add(chat)
        db.session.commit()
        return jsonify({'success': True, 'chatid': chat.id, 'title': chat.title})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to create chat'}), 500


@app.route('/chat/<int:chat_id>')
@login_required
def chat_view(chat_id):
    """View specific chat"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
    plan_cfg = get_user_plan(current_user)
    available_models = get_available_models(current_user)
    
    return render_template(
        'dashboard.html',
        user=current_user,
        chats=chats,
        active_chat=chat,
        messages=chat.messages,
        models={m['id']: m for m in available_models},
        plan=plan_cfg,
        plans_config=PLANS
    )


@app.route('/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    """Get messages for a chat"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    
    messages = [{
        'id': m.id,
        'role': m.role,
        'content': m.content,
        'model': m.model,
        'has_image': m.has_image,
        'image_url': m.image_url,
        'created_at': m.created_at.isoformat()
    } for m in chat.messages]
    
    return jsonify({'messages': messages, 'title': chat.title})


@app.route('/chat/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    """Rename chat - THIS WAS MISSING!"""
    try:
        data = request.get_json() or {}
        new_title = (data.get('title') or '').strip()
        
        if not new_title:
            return jsonify({'error': 'Title required'}), 400
        
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
        chat.title = new_title[:200]
        db.session.commit()
        
        return jsonify({'success': True, 'title': chat.title})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to rename chat'}), 500


@app.route('/chat/<int:chat_id>/delete', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    """Delete chat"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
        db.session.delete(chat)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to delete chat'}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - SETTINGS (SAVE TO DATABASE)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def user_settings():
    """Get or update user settings in database"""
    try:
        settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        
        if not settings:
            settings = UserSettings(user_id=current_user.id)
            db.session.add(settings)
            db.session.commit()
        
        if request.method == 'GET':
            return jsonify({
                'compact_mode': settings.compact_mode,
                'enter_to_send': settings.enter_to_send,
                'theme': settings.theme,
                'memory_enabled': settings.memory_enabled,
                'memory_text': settings.memory_text or ''
            })
        
        elif request.method == 'POST':
            data = request.get_json() or {}
            
            if 'compact_mode' in data:
                settings.compact_mode = bool(data['compact_mode'])
            if 'enter_to_send' in data:
                settings.enter_to_send = bool(data['enter_to_send'])
            if 'theme' in data:
                settings.theme = data['theme']
            if 'memory_enabled' in data:
                settings.memory_enabled = bool(data['memory_enabled'])
            if 'memory_text' in data:
                settings.memory_text = data['memory_text']
            
            settings.updated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Settings saved'})
    
    except Exception as e:
        db.session.rollback()
        log.error(f"Settings error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES - FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """File upload endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
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
        except:
            pass
        
        file_url = url_for('uploaded_file', filename=filename, _external=True)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'url': file_url
        })
    
    except Exception as e:
        log.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ═════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return "<h1>404 - Page Not Found</h1><a href='/dashboard'>Go to Dashboard</a>", 404


@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    log.error(f"Server error: {e}")
    if request.path.startswith('/api'):
        return jsonify({'error': 'Internal server error'}), 500
    return "<h1>500 - Internal Server Error</h1><a href='/dashboard'>Go to Dashboard</a>", 500


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("=" * 60)
            print("✅ NexaAI Database Initialized Successfully")
            print("=" * 60)
        except Exception as e:
            print("=" * 60)
            print(f"❌ Database Initialization Failed: {e}")
            print("=" * 60)
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print(f"🚀 Starting NexaAI Server")
    print(f"📡 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"🤖 Google AI: {'✅' if GOOGLE_API_KEY else '❌'}")
    print(f"🔗 OpenRouter: {'✅' if OPENROUTER_API_KEY else '❌'}")
    print("=" * 60)
    
    app.run(debug=debug, host='0.0.0.0', port=port)
