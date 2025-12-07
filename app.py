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
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
import requests
import stripe
import os
import base64
from PIL import Image
import traceback
import logging
import json

# Google Generative AI for Gemini
import google.generativeai as genai

# Load environment
load_dotenv()

# -------- App setup --------
app = Flask(__name__)

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ============ CONFIGURATION ============
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-2025')

# Database Configuration
database_url = os.getenv('DATABASE_URL')
if database_url:
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    log.info("🐘 Using PostgreSQL database")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    log.info("📁 Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}
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

# ============ Configure AI APIs ============
log.info("🤖 Configuring AI APIs...")

# Google Generative AI
google_api_key = os.getenv('GOOGLE_API_KEY')
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        log.info("✅ Google Generative AI configured")
    except Exception as e:
        log.error(f"❌ Google AI config error: {e}")
else:
    log.warning("⚠️ GOOGLE_API_KEY not set")

# OpenRouter API Key
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if openrouter_api_key:
    log.info("✅ OpenRouter configured")
else:
    log.warning("⚠️ OPENROUTER_API_KEY not set")

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
    deepseek_date = db.Column(db.String(10), default='')
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

# ============ DATABASE INITIALIZATION ============
def init_database():
    """Initialize database with proper error handling"""
    with app.app_context():
        try:
            db.create_all()
            log.info("✅ Database tables created/verified")
            return True
        except Exception as e:
            log.error(f"❌ Database initialization failed: {e}")
            log.exception("Database error details:")
            return False

# Initialize database
init_database()

# ============ AI MODELS CONFIG ============
FREE_MODELS = {
    "gemini-flash": {
        "name": "Gemini 2.0 Flash",
        "model": "gemini-2.0-flash-exp",
        "provider": "google",
        "vision": True,
        "limit": None,
        "description": "Fast and efficient Gemini Flash model"
    },
    "gpt-3.5-turbo": {
        "name": "ChatGPT 3.5 Turbo",
        "model": "openai/gpt-3.5-turbo",
        "provider": "openrouter",
        "vision": False,
        "limit": None,
        "description": "OpenAI GPT-3.5 Turbo"
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku",
        "model": "anthropic/claude-3-haiku",
        "provider": "openrouter",
        "vision": True,
        "limit": None,
        "description": "Fast Claude model"
    },
    "deepseek-chat": {
        "name": "DeepSeek Chat",
        "model": "deepseek/deepseek-chat",
        "provider": "openrouter",
        "vision": False,
        "limit": 50,
        "description": "Powerful for code & logic (50/day for free users)"
    },
}

PREMIUM_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "model": "openai/gpt-4o",
        "provider": "openrouter",
        "vision": True,
        "description": "Most capable GPT-4 family model"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "model": "openai/gpt-4o-mini",
        "provider": "openrouter",
        "vision": True,
        "description": "Efficient GPT-4 style model"
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "model": "anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "vision": True,
        "description": "Best for coding & analysis"
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "model": "anthropic/claude-3-opus",
        "provider": "openrouter",
        "vision": True,
        "description": "Most capable Claude model"
    },
    "gemini-pro": {
        "name": "Gemini 1.5 Pro",
        "model": "gemini-1.5-pro",
        "provider": "google",
        "vision": True,
        "description": "Multimodal vision-capable"
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "model": "deepseek/deepseek-r1",
        "provider": "openrouter",
        "vision": False,
        "description": "Advanced reasoning model"
    },
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

# ============ AI API CALLING FUNCTIONS ============
def encode_image_to_base64(image_path):
    """Convert image to base64 for vision models"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        log.exception(f"Failed to encode image: {image_path}")
        return None

def call_google_gemini(model_path, messages, image_path=None, timeout=60):
    """Call Google Generative AI for Gemini models with vision support"""
    try:
        if not google_api_key:
            raise Exception("Google API key not configured")
        
        model = genai.GenerativeModel(model_path)
        
        # Extract the last user message
        last_message = messages[-1]['content'] if messages else ""
        
        # Build conversation history (exclude last message and system messages)
        gemini_history = []
        for msg in messages[:-1]:
            if msg.get('role') == 'system':
                continue
            gemini_role = 'user' if msg.get('role') == 'user' else 'model'
            gemini_history.append({'role': gemini_role, 'parts': [msg.get('content', '')]})
        
        # Start chat with history
        chat = model.start_chat(history=gemini_history)
        
        # Handle vision input
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                response = chat.send_message([last_message, img])
            except Exception as e:
                log.error(f"Vision input failed, using text only: {e}")
                response = chat.send_message(last_message)
        else:
            response = chat.send_message(last_message)
        
        return response.text
    
    except Exception as e:
        log.exception("Google Gemini API error")
        raise Exception(f"Gemini API error: {str(e)}")

def call_openrouter(model_path, messages, image_path=None, timeout=60):
    """Call OpenRouter API with vision support"""
    try:
        if not openrouter_api_key:
            raise Exception("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("APP_URL", "https://nexaai.app"),
            "X-Title": "NexaAI"
        }
        
        # Handle vision models (Claude, GPT-4 Vision)
        formatted_messages = []
        for msg in messages:
            if msg.get('role') == 'system':
                formatted_messages.append({"role": "system", "content": msg['content']})
            else:
                content = msg.get('content', '')
                
                # Add image for last user message if provided
                if msg == messages[-1] and image_path and os.path.exists(image_path):
                    base64_image = encode_image_to_base64(image_path)
                    if base64_image:
                        ext = os.path.splitext(image_path)[1].lower().replace('.', '')
                        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'image/jpeg'
                        
                        formatted_messages.append({
                            "role": msg.get('role', 'user'),
                            "content": [
                                {"type": "text", "text": content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        })
                    else:
                        formatted_messages.append({"role": msg.get('role', 'user'), "content": content})
                else:
                    formatted_messages.append({"role": msg.get('role', 'user'), "content": content})
        
        payload = {
            "model": model_path,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        response.raise_for_status()
        data = response.json()
        
        if 'choices' not in data or len(data['choices']) == 0:
            raise Exception("No response from model")
        
        return data['choices'][0]['message']['content']
    
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        status_code = response.status_code if 'response' in locals() else 0
        if status_code == 429:
            raise Exception("Rate limit reached. Please try again later.")
        elif status_code == 401:
            raise Exception("Invalid API key.")
        else:
            raise Exception(f"API error (Status {status_code})")
    except Exception as e:
        log.exception("OpenRouter API call failed")
        raise Exception(f"API error: {str(e)}")

def call_ai_model(model_key, messages, uploaded_file=None):
    """Universal function to call any AI model"""
    if model_key in FREE_MODELS:
        model_config = FREE_MODELS[model_key]
    elif model_key in PREMIUM_MODELS:
        model_config = PREMIUM_MODELS[model_key]
    else:
        raise Exception(f"Model {model_key} not found")
    
    provider = model_config['provider']
    model_path = model_config['model']
    
    if provider == 'google':
        return call_google_gemini(model_path, messages, uploaded_file)
    elif provider == 'openrouter':
        return call_openrouter(model_path, messages, uploaded_file)
    else:
        raise Exception(f"Unknown provider: {provider}")

# ============ HELPER FUNCTIONS ============
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    """Return last messages as list for API"""
    try:
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at.desc()).limit(limit).all()
        messages = list(reversed(messages))
        
        formatted = []
        for msg in messages:
            formatted.append({
                'role': msg.role,
                'content': msg.content
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
    log.exception("Internal server error")
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Internal server error. Please try again.'}), 500
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
        image_keywords = ['generate image', 'create image', 'draw', 'picture of', 'image of', 'make an image', 'show me']
        if any(k in message.lower() for k in image_keywords):
            try:
                url = generate_image(message)
                return jsonify({
                    'response': "Here's your generated image!",
                    'image_url': url,
                    'has_image': True,
                    'demo': True,
                    'model': 'Pollinations AI'
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
                'model': 'Gemini 2.0 Flash',
                'has_image': False
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
                error_msg = 'All fields are required'
                if request.is_json:
                    return jsonify({'error': error_msg}), 400
                return render_template('signup.html', error=error_msg)
            
            if len(password) < 6:
                error_msg = 'Password must be at least 6 characters'
                if request.is_json:
                    return jsonify({'error': error_msg}), 400
                return render_template('signup.html', error=error_msg)
            
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                error_msg = 'Email already exists'
                if request.is_json:
                    return jsonify({'error': error_msg}), 400
                return render_template('signup.html', error=error_msg)

            hashed = generate_password_hash(password)
            new_user = User(
                email=email,
                name=name,
                password=hashed,
                deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
            )
            
            db.session.add(new_user)
            db.session.commit()
            log.info(f"✅ New user registered: {email}")

            if request.is_json:
                return jsonify({'success': True, 'redirect': url_for('login')})
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            log.exception(f"Signup failed: {e}")
            error_msg = 'Signup failed. Please try again.'
            if request.is_json:
                return jsonify({'error': error_msg}), 500
            return render_template('signup.html', error=error_msg)

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            email = (data.get('email') or '').strip().lower()
            password = data.get('password') or ''
            
            if not email or not password:
                error_msg = 'Email and password are required'
                if request.is_json:
                    return jsonify({'error': error_msg}), 400
                return render_template('login.html', error=error_msg)

            user = User.query.filter_by(email=email).first()
            
            if user and check_password_hash(user.password, password):
                login_user(user, remember=True)
                log.info(f"✅ User logged in: {email}")
                if request.is_json:
                    return jsonify({'success': True, 'redirect': url_for('dashboard')})
                return redirect(url_for('dashboard'))

            log.warning(f"Failed login attempt for: {email}")
            error_msg = 'Invalid email or password'
            if request.is_json:
                return jsonify({'error': error_msg}), 401
            return render_template('login.html', error=error_msg)
            
        except Exception as e:
            log.exception(f"Login failed: {e}")
            error_msg = 'Login failed. Please try again.'
            if request.is_json:
                return jsonify({'error': error_msg}), 500
            return render_template('login.html', error=error_msg)

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
        chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.updated_at.desc()).limit(20).all()
        all_models = {**FREE_MODELS, **PREMIUM_MODELS}
        return render_template('dashboard.html', user=current_user, chats=chats, models=all_models)
    except Exception:
        log.exception("Dashboard load failed")
        return render_template('dashboard.html', user=current_user, chats=[], models={**FREE_MODELS, **PREMIUM_MODELS})

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
            return jsonify({'error': 'File type not allowed. Supported: PNG, JPG, JPEG, GIF, WEBP, PDF'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{current_user.id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Compress images
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
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

# -------- Main chat endpoint --------
@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        selected_model = data.get('model') or 'gemini-flash'
        chat_id = data.get('chat_id') or data.get('chatid')
        uploaded_file_path = data.get('uploaded_file') or data.get('uploadedfile')

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
    try:
        db.session.execute(db.text('SELECT 1'))
        db_status = True
    except:
        db_status = False
    
    return jsonify({
        'status': 'healthy' if db_status else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'database_connected': db_status,
        'google_ai_configured': bool(google_api_key),
        'openrouter_configured': bool(openrouter_api_key),
        'stripe_configured': bool(stripe.api_key)
    })

# ============ RUN ============
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    log.info(f"🚀 Starting NexaAI on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
