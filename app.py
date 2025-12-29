import os
import logging
import base64
import io
import urllib.parse
import requests
import stripe
import google.generativeai as genai
from datetime import datetime, timedelta
from io import BytesIO

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    session, send_from_directory
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

# -------------------------------------------------------------------
# 1. APP CONFIGURATION & SETUP
# -------------------------------------------------------------------

# Load Environment Variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-2025')
app.config['JSON_AS_ASCII'] = False

# Database Config
database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')
if database_url and database_url.startswith('postgres'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///nexa-ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 300}

# File Upload Config
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# API Keys Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
STRIPE_KEY = os.getenv('STRIPE_SECRET_KEY')

# Configure Google AI
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        log.info("Google Generative AI configured")
    except Exception as e:
        log.error(f"Google AI config error: {e}")

# Configure Stripe
if STRIPE_KEY and STRIPE_KEY.startswith('sk_'):
    stripe.api_key = STRIPE_KEY
else:
    stripe.api_key = None
    log.warning("Stripe key missing or invalid")


# -------------------------------------------------------------------
# 2. CONSTANTS & PLAN CONFIGURATION
# -------------------------------------------------------------------

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

PLANS = {
    "basic": {
        "name": "Basic",
        "description": "Free plan. Access to free AI models only.",
        "allowed_model_types": {"free"},
        "allow_deepthink": False,
        "allow_web_scraper": False,
    },
    "pro": {
        "name": "Pro",
        "description": "Free for 6 months. Access to Pro models.",
        "allowed_model_types": {"free", "pro"},
        "allow_deepthink": True,
        "allow_web_scraper": True,
        "trial_months": 6,
    },
    "max": {
        "name": "Max",
        "description": "Top ranked AIs and advanced features.",
        "allowed_model_types": {"free", "pro", "max"},
        "allow_deepthink": True,
        "allow_web_scraper": True,
    },
}

FREE_MODELS = [
    {'name': 'Gemini 2.5 Flash', 'model': 'gemini-2.5-flash', 'provider': 'google', 'vision': True, 'tier': 'free', 'description': 'Fast and efficient Gemini Flash model'},
    {'name': 'ChatGPT 3.5 Turbo', 'model': 'openai/gpt-3.5-turbo', 'provider': 'openrouter', 'vision': False, 'tier': 'free', 'description': 'OpenAI GPT-3.5 Turbo'},
    {'name': 'Claude 3 Haiku', 'model': 'anthropic/claude-3-haiku', 'provider': 'openrouter', 'vision': True, 'tier': 'free', 'description': 'Fast Claude model'},
    {'name': 'DeepSeek Chat', 'model': 'deepseek/deepseek-chat', 'provider': 'openrouter', 'vision': False, 'limit': 50, 'tier': 'free', 'description': 'Powerful code & logic (50/day)'},
]

PREMIUM_MODELS = [
    {'name': 'GPT-4o Mini', 'model': 'openai/gpt-4o-mini', 'provider': 'openrouter', 'vision': True, 'tier': 'pro', 'description': 'Efficient GPT-4 family model'},
    {'name': 'Gemini 1.5 Pro', 'model': 'gemini-1.5-pro', 'provider': 'google', 'vision': True, 'tier': 'pro', 'description': 'Multimodal vision-capable'},
    {'name': 'GPT-4o', 'model': 'openai/gpt-4o', 'provider': 'openrouter', 'vision': True, 'tier': 'max', 'description': 'Most capable GPT-4 model'},
    {'name': 'Claude 3.5 Sonnet', 'model': 'anthropic/claude-3.5-sonnet', 'provider': 'openrouter', 'vision': True, 'tier': 'max', 'description': 'Best for coding & analysis'},
    {'name': 'DeepSeek R1', 'model': 'deepseek/deepseek-r1', 'provider': 'openrouter', 'vision': False, 'tier': 'max', 'description': 'Advanced reasoning model'},
]


# -------------------------------------------------------------------
# 3. DATABASE MODELS
# -------------------------------------------------------------------

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    
    # Plan info
    plan = db.Column(db.String(20), default='basic')
    plan_started_at = db.Column(db.DateTime, default=datetime.utcnow)
    subscription_id = db.Column(db.String(100), nullable=True)
    
    # Limits
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10), default='')
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

    # --- KEY FIX: Added ispremium property ---
    @property
    def ispremium(self):
        return self.plan in ['pro', 'max']

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

class Message(db.Model):
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    has_image = db.Column(db.Boolean, default=False)
    image_url = db.Column(db.String(1000), nullable=True)
    image_data = db.Column(db.Text, nullable=True) # Base64 storage if needed
    model = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


# -------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------------------------------------------

def init_database():
    """Initializes the database and tables."""
    with app.app_context():
        try:
            db.create_all()
            log.info("Database tables verified/created.")
            return True
        except Exception as e:
            log.error(f"DB Initialization failed: {e}")
            return False

def get_user_plan(user: User):
    """Calculates effective plan (handles trial expiration)."""
    plan_key = user.plan or "basic"
    cfg = PLANS.get(plan_key, PLANS["basic"])

    if plan_key == "pro" and user.plan_started_at:
        trial_months = cfg.get("trial_months", 6)
        # If trial expired (> 6 months) and no paid sub, downgrade
        if datetime.utcnow() - user.plan_started_at > timedelta(days=trial_months * 30):
            if not user.subscription_id:
                user.plan = "basic"
                db.session.commit()
                cfg = PLANS["basic"]
    return cfg

def check_deepseek_limit(user):
    """Resets daily limit if new day, returns True if allowed."""
    today = str(datetime.utcnow().date())
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    
    # 50 messages per day limit for deepseek
    return user.deepseek_count < 50

def increment_deepseek_count(user):
    today = str(datetime.utcnow().date())
    if user.deepseek_date != today:
        user.deepseek_count = 1
        user.deepseek_date = today
    else:
        user.deepseek_count += 1
    db.session.commit()

def generate_image_prompt(prompt):
    """Generates image URL using Pollinations.ai"""
    try:
        clean_prompt = prompt.lower()
        for prefix in ['draw', 'generate image', 'create image', 'make an image of']:
            clean_prompt = clean_prompt.replace(prefix, '')
        encoded = urllib.parse.quote(clean_prompt.strip())
        return f"https://image.pollinations.ai/prompt/{encoded}?nologo=true"
    except Exception:
        return None

def extract_text_content(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
    return str(content)

# --- AI API CALLS ---

def call_google_gemini(model_path, messages, image_data=None):
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not configured."
    
    try:
        model = genai.GenerativeModel(model_path)
        content_parts = []
        
        # Add text
        if messages:
            last_msg = messages[-1]
            text = extract_text_content(last_msg.get('content', ''))
            content_parts.append(text)
            
        # Add image if present
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(image_bytes))
                content_parts.append(img)
            except Exception as e:
                log.error(f"Image processing error: {e}")

        if not content_parts: return "Please say something."

        response = model.generate_content(content_parts)
        return response.text if response and response.text else "No response generated."

    except Exception as e:
        log.error(f"Gemini API Error: {e}")
        return f"Error connecting to AI: {str(e)[:100]}"

def call_openrouter(model_path, messages, image_data=None):
    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API Key not configured."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nexa-ai.onrender.com",
        "X-Title": "NexaAI"
    }

    # Format messages
    formatted_msgs = []
    for msg in messages:
        content = extract_text_content(msg.get('content', ''))
        formatted_msgs.append({"role": msg.get('role', 'user'), "content": content})

    # Handle Image for GPT/Claude
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

    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        else:
            return f"Error {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        log.error(f"OpenRouter Error: {e}")
        return "Error connecting to AI Provider."


# -------------------------------------------------------------------
# 5. ROUTES
# -------------------------------------------------------------------

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))

    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form.to_dict()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            return jsonify(success=True, redirect=url_for('dashboard'))
        
        return jsonify(success=False, error="Invalid email or password"), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))

    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form.to_dict()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password')
        name = (data.get('name') or '').strip()

        if User.query.filter_by(email=email).first():
            return jsonify(success=False, error="Email already exists"), 409

        # New users get 'pro' trial
        new_user = User(
            email=email,
            password=generate_password_hash(password),
            name=name,
            plan="pro",
            plan_started_at=datetime.utcnow()
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            return jsonify(success=True, redirect=url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            return jsonify(success=False, error="Registration failed"), 500

    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
    plan_cfg = get_user_plan(current_user)
    
    # Filter models based on plan
    allowed_tiers = plan_cfg["allowed_model_types"]
    available_models = [m for m in FREE_MODELS + PREMIUM_MODELS if m.get("tier") in allowed_tiers]

    return render_template('dashboard.html',
                           chats=chats,
                           models={m['model']: m for m in available_models}, # Pass as dict for template lookups
                           user=current_user,
                           plan=plan_cfg)

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        model_key = data.get('model', 'gemini-2.5-flash')
        chat_id = data.get('chat_id')
        image_data = data.get('image') # Base64 string

        if not message: return jsonify({'error': 'Empty message'}), 400

        # Handle Image Generation Command
        if message.lower().startswith(('draw', 'generate image', 'create image')):
            img_url = generate_image_prompt(message)
            if img_url:
                # Create chat if needed
                if not chat_id:
                    chat = Chat(user_id=current_user.id, title=message[:30])
                    db.session.add(chat)
                    db.session.flush()
                    chat_id = chat.id
                
                # Save User Msg
                db.session.add(Message(chat_id=chat_id, role='user', content=message, model='image-gen'))
                # Save AI Msg
                db.session.add(Message(chat_id=chat_id, role='assistant', content="Here is your image:", 
                                       model='image-gen', has_image=True, image_url=img_url))
                db.session.commit()
                
                return jsonify({
                    'success': True, 'chat_id': chat_id, 'response': "Image Generated",
                    'image_url': img_url, 'url': url_for('chat_view', chat_id=chat_id)
                })

        # --- Regular AI Chat ---
        
        # Check Plan Permission
        plan_cfg = get_user_plan(current_user)
        all_models = FREE_MODELS + PREMIUM_MODELS
        model_cfg = next((m for m in all_models if m['model'] == model_key), None)

        if not model_cfg: return jsonify({'error': 'Invalid Model'}), 400
        if model_cfg['tier'] not in plan_cfg['allowed_model_types']:
            return jsonify({'error': 'Upgrade required for this model'}), 403

        # Check DeepSeek Limit
        if 'deepseek' in model_key.lower():
            if not check_deepseek_limit(current_user):
                return jsonify({'error': 'Daily limit reached (50/day).'}), 429

        # Get/Create Chat
        if chat_id:
            chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        else:
            chat = Chat(user_id=current_user.id, title=message[:50])
            db.session.add(chat)
            db.session.flush()

        # Save User Message
        user_msg = Message(chat_id=chat.id, role='user', content=message, model=model_key, image_data=image_data)
        db.session.add(user_msg)
        
        # Get History
        history_objs = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
        history = [{'role': m.role, 'content': m.content} for m in history_objs]

        # Call API
        if model_cfg['provider'] == 'google':
            ai_text = call_google_gemini(model_cfg['model'], history, image_data)
        else:
            ai_text = call_openrouter(model_cfg['model'], history, image_data)

        # Save AI Response
        ai_msg = Message(chat_id=chat.id, role='assistant', content=ai_text, model=model_key)
        db.session.add(ai_msg)
        
        chat.updated_at = datetime.utcnow()
        if 'deepseek' in model_key.lower(): increment_deepseek_count(current_user)
        
        db.session.commit()

        return jsonify({
            'success': True,
            'chat_id': chat.id,
            'response': ai_text,
            'chat_title': chat.title,
            'url': url_for('chat_view', chat_id=chat.id)
        })

    except Exception as e:
        db.session.rollback()
        log.error(f"Chat Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/<int:chat_id>')
@login_required
def chat_view(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    # Pass necessary data to template (reusing logic from dashboard)
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
    plan_cfg = get_user_plan(current_user)
    allowed_tiers = plan_cfg["allowed_model_types"]
    available_models = [m for m in FREE_MODELS + PREMIUM_MODELS if m.get("tier") in allowed_tiers]
    
    return render_template('dashboard.html', # Re-using dashboard template effectively
                           chats=chats,
                           active_chat=chat,
                           messages=chat.messages,
                           models={m['model']: m for m in available_models},
                           user=current_user,
                           plan=plan_cfg)

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    chat = Chat(user_id=current_user.id, title="New Chat")
    db.session.add(chat)
    db.session.commit()
    return jsonify({'success': True, 'chat_id': chat.id, 'url': url_for('chat_view', chat_id=chat.id)})

@app.route('/api/messages/<int:chat_id>')
@login_required
def get_messages(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    msgs = [{
        'role': m.role,
        'content': m.content,
        'has_image': m.has_image,
        'image_url': m.image_url
    } for m in chat.messages]
    return jsonify({'messages': msgs, 'title': chat.title})

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    """Public demo for landing page"""
    data = request.get_json() or {}
    msg = data.get('message', '')
    if not msg: return jsonify({'error': 'Empty'}), 400
    
    # Demo only uses Gemini Flash
    resp = call_google_gemini('gemini-2.5-flash', [{'role':'user', 'content':msg}])
    return jsonify({'response': resp, 'demo': True})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------------------------------------------------------
# 6. MAIN EXECUTION
# -------------------------------------------------------------------

if __name__ == '__main__':
    # Initialize DB before running
    with app.app_context():
        try:
            db.create_all()
            print(">>> Database Initialized Successfully")
        except Exception as e:
            print(f">>> Database Error: {e}")

    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    print(f">>> Starting NexaAI on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)
