from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory, abort, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy import text as sql_text, desc
import requests
import urllib.parse
import stripe
import os
import base64
from PIL import Image
import logging
import io
import google.generativeai as genai
from io import BytesIO

# Load Environment
load_dotenv()

TITLE = "Nexa-AI"
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# App Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-2025')

# Database Configuration
database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')
if database_url:
    if database_url.startswith('postgres'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    log.info("Using PostgreSQL database")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nexa-ai.db'
    log.info("Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 300}
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['JSON_AS_ASCII'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Setup
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

log.info("Configuring AI APIs...")

# Configure AI APIs
google_api_key = os.getenv('GOOGLE_API_KEY')
if google_api_key:
    try:
        google_api_key = google_api_key.strip()
        genai.configure(api_key=google_api_key)
        log.info("Google Generative AI configured")
    except Exception as e:
        log.error(f"Google AI config error: {e}")
        google_api_key = None
else:
    log.warning("GOOGLE_API_KEY not set")

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if openrouter_api_key:
    openrouter_api_key = openrouter_api_key.strip()
    log.info("OpenRouter configured")
else:
    log.warning("OPENROUTER_API_KEY not set")

stripe_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_key:
    stripe_key = stripe_key.strip()
    if stripe_key.startswith('sk_'):
        stripe.api_key = stripe_key
        log.info("Stripe configured")
    else:
        log.error("INVALID Stripe key format")
        stripe.api_key = None
else:
    stripe.api_key = None
    log.warning("STRIPE_SECRET_KEY not set")

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'user'  # Quoted for PostgreSQL
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10), default='')
    subscription_id = db.Column(db.String(100), nullable=True)
    is_premium = db.Column(db.Boolean, default=False)
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
        return f'<Chat {self.id} - {self.title}>'

class Message(db.Model):
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    has_image = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(1000), nullable=True)
    image_url = db.Column(db.String(1000), nullable=True)
    image_data = db.Column(db.Text, nullable=True)  # Base64
    model = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        log.exception("Error loading user")
        return None

def validate_api_keys():
    issues = []
    if not google_api_key or len(google_api_key) < 20:
        issues.append("GOOGLE_API_KEY missing or invalid")
        log.warning("GOOGLE_API_KEY not properly configured")
    else:
        log.info("Google API key validated")
    
    if not openrouter_api_key or len(openrouter_api_key) < 20:
        issues.append("OPENROUTER_API_KEY missing or invalid")
        log.warning("OPENROUTER_API_KEY not properly configured")
    else:
        log.info("OpenRouter API key validated")
    
    if issues:
        log.warning(f"API Configuration Issues: {', '.join(issues)}")
    return len(issues) == 0

def init_database():
    with app.app_context():
        try:
            db.create_all()
            log.info("Database tables created/verified")
            validate_api_keys()
            return True
        except Exception as e:
            log.error(f"Database initialization failed: {e}")
            log.exception("Database error details")
            return False

init_database()

# AI Models Configuration
FREE_MODELS = [
    {'name': 'Gemini 2.5 Flash', 'model': 'gemini-2.5-flash', 'provider': 'google', 'vision': True, 'limit': None, 'description': 'Fast and efficient Gemini Flash model'},
    {'name': 'ChatGPT 3.5 Turbo', 'model': 'openai/gpt-3.5-turbo', 'provider': 'openrouter', 'vision': False, 'limit': None, 'description': 'OpenAI GPT-3.5 Turbo'},
    {'name': 'Claude 3 Haiku', 'model': 'anthropic/claude-3-haiku', 'provider': 'openrouter', 'vision': True, 'limit': None, 'description': 'Fast Claude model'},
    {'name': 'DeepSeek Chat', 'model': 'deepseek/deepseek-chat', 'provider': 'openrouter', 'vision': False, 'limit': 50, 'description': 'Powerful for code & logic (50/day for free users)'}
]

PREMIUM_MODELS = [
    {'name': 'GPT-4o', 'model': 'openai/gpt-4o', 'provider': 'openrouter', 'vision': True, 'description': 'Most capable GPT-4 family model'},
    {'name': 'GPT-4o Mini', 'model': 'openai/gpt-4o-mini', 'provider': 'openrouter', 'vision': True, 'description': 'Efficient GPT-4 style model'},
    {'name': 'Claude 3.5 Sonnet', 'model': 'anthropic/claude-3.5-sonnet', 'provider': 'openrouter', 'vision': True, 'description': 'Best for coding & analysis'},
    {'name': 'Claude 3 Opus', 'model': 'anthropic/claude-3-opus', 'provider': 'openrouter', 'vision': True, 'description': 'Most capable Claude model'},
    {'name': 'Gemini 1.5 Pro', 'model': 'gemini-1.5-pro', 'provider': 'google', 'vision': True, 'description': 'Multimodal vision-capable'},
    {'name': 'DeepSeek R1', 'model': 'deepseek/deepseek-r1', 'provider': 'openrouter', 'vision': False, 'description': 'Advanced reasoning model'}
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

def generate_image_prompt(prompt):
    """Generate image using Pollinations.ai"""
    try:
        log.info(f"Generating image for prompt: {prompt}")
        clean_prompt = prompt.lower()
        for prefix in ['draw', 'generate image', 'create image', 'make an image of']:
            if clean_prompt.startswith(prefix):
                clean_prompt = clean_prompt.replace(prefix, '')
        encoded = urllib.parse.quote(clean_prompt.strip())
        return f"https://image.pollinations.ai/prompt/{encoded}?nologo=true"
    except Exception as e:
        log.error(f"Image gen error: {e}")
        return None

def extract_text_content(content):
    """Extract text from various content formats"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        log.exception(f"Failed to encode image {image_path}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_google_gemini(model_path, messages, image_data=None, image_path=None, timeout=90):
    """Call Google Generative AI (Gemini) - Enhanced Error Handling"""
    try:
        if not google_api_key:
            log.error("Google API key not configured")
            raise Exception("Google API key not configured. Please set GOOGLE_API_KEY in environment.")
        
        log.info(f"Calling Gemini model: {model_path}")
        model = genai.GenerativeModel(model_path)
        
        content = []
        if messages:
            last_msg = messages[-1]
            text = extract_text_content(last_msg.get('content', ''))
            if text:
                content.append(text)
                log.info(f"Message length: {len(text)} chars")
        
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                content.append(image)
                log.info("Image attached to request")
            except Exception as e:
                log.warning(f"Failed to process image: {e}")
        
        if not content:
            content = ["Please respond."]
        
        log.info(f"Sending request with timeout: {timeout}...")
        response = model.generate_content(
            content,
            request_options={'timeout': timeout}
        )
        
        if response and response.text:
            log.info(f"Gemini response received: {len(response.text)} chars")
            return response.text
        else:
            log.warning("Empty response from Gemini")
            return "No response generated. Please try again."
            
    except Exception as e:
        error_msg = str(e)
        log.error(f"Gemini API error: {error_msg}")
        
        if "APIKEY" in error_msg.upper() or "INVALID_ARGUMENT" in error_msg.upper():
            return "Error: Invalid or missing Google API key. Please check your configuration."
        elif "TIMEOUT" in error_msg.upper() or "DEADLINE_EXCEEDED" in error_msg.upper():
            return "Error: Request timed out. The model took too long to respond. Try again with a shorter prompt."
        elif "QUOTA" in error_msg.upper() or "RATE" in error_msg.upper():
            return "Error: API rate limit exceeded. Please wait a moment and try again."
        elif "SAFETY" in error_msg.upper() or "BLOCK" in error_msg.upper():
            return "Error: Content blocked by safety filters. Please rephrase your request."
        elif "NOT_FOUND" in error_msg.upper():
            return "Error: Model not found. Please select a different model."
        else:
            return f"Error: {error_msg[:150]}"

def call_openrouter(model_path, messages, image_data=None, image_path=None):
    """Call OpenRouter API - Enhanced Error Handling"""
    try:
        if not openrouter_api_key:
            log.error("OpenRouter API key not configured")
            raise Exception("OpenRouter API key not configured. Please set OPENROUTER_API_KEY.")
        
        log.info(f"Calling OpenRouter model: {model_path}")
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexa-ai-2d8d.onrender.com",
            "X-Title": "NexaAI"
        }
        
        formatted_messages = []
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                formatted_messages.append({"role": msg.get('role', 'user'), "content": content})
            else:
                formatted_messages.append({"role": msg.get('role', 'user'), "content": extract_text_content(content)})
        
        log.info(f"Message history: {len(formatted_messages)} messages")
        
        if image_data and ('gpt-4' in model_path.lower() or 'claude' in model_path.lower()):
            formatted_messages[-1]['content'] = [
                {"type": "text", "text": formatted_messages[-1]['content']},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
            log.info("Image attached to OpenRouter request")
        
        payload = {
            "model": model_path,
            "messages": formatted_messages,
            "temperature": 0.7,
        }
        
        log.info("Sending OpenRouter request...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )
        
        log.info(f"OpenRouter status: {response.status_code}")
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text)[:200]
            log.error(f"OpenRouter error: {error_msg}")
            raise Exception(f"OpenRouter API error {response.status_code}: {error_msg}")
        
        result = response.json()
        ai_response = result['choices'][0]['message']['content']
        log.info(f"OpenRouter response: {len(ai_response)} chars")
        return ai_response
        
    except requests.exceptions.Timeout:
        log.error("OpenRouter request timed out")
        return "Error: Request timed out after 90 seconds. Try a shorter prompt or different model."
    except requests.exceptions.ConnectionError:
        log.error("OpenRouter connection failed")
        return "Error: Could not connect to OpenRouter API. Check your internet connection."
    except Exception as e:
        error_msg = str(e)
        log.error(f"OpenRouter error: {error_msg}")
        
        if "APIKEY" in error_msg.upper() or "401" in error_msg:
            return "Error: Invalid OpenRouter API key. Please check your configuration."
        elif "RATE" in error_msg.upper() or "429" in error_msg:
            return "Error: Rate limit exceeded. Please wait and try again."
        elif "INSUFFICIENT_QUOTA" in error_msg.upper():
            return "Error: API quota exceeded. Please add credits to your OpenRouter account."
        else:
            return f"Error: {error_msg[:150]}"

def is_stripe_configured():
    """Check if Stripe is properly configured"""
    if not stripe.api_key:
        return False
    try:
        stripe.Account.retrieve()
        return True
    except Exception as e:
        log.error(f"Stripe API test failed: {e}")
        return False

def check_deepseek_limit(user):
    """Check if user has exceeded DeepSeek daily limit"""
    today = str(datetime.utcnow().date())
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    if user.is_premium:
        return True
    return user.deepseek_count < 50

def increment_deepseek_count(user):
    """Increment user's DeepSeek usage count"""
    today = str(datetime.utcnow().date())
    if user.deepseek_date != today:
        user.deepseek_count = 1
        user.deepseek_date = today
    else:
        user.deepseek_count += 1
    db.session.commit()

# Routes
@app.route('/')
def index():
    """Home page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            log.info(f"User logged in: {email}")
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        log.warning(f"Failed login attempt: {email}")
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    """Public demo endpoint used on the landing page. Supports image generation via Pollinations and text via Gemini Flash."""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Check if message is image generation request
        if message.lower().startswith(('draw', 'generate image', 'create image', 'make an image of')):
            try:
                url = generate_image_prompt(message)
                if url:
                    return jsonify({
                        'response': f"Here is your generated image!",
                        'has_image': True,
                        'image_url': url,
                        'demo': True
                    })
                else:
                    return jsonify({'response': "Sorry, I couldn't generate that image right now."})
            except Exception as e:
                log.exception("Demo image generation failed")
                return jsonify({'error': str(e)}), 500
        
        # Text response via Gemini
        try:
            if not google_api_key:
                return jsonify({'response': "I am in Demo Mode, but no API key is configured. Please sign up!"})
            
            messages = [{"role": "user", "content": message}]
            response_text = call_google_gemini('gemini-2.5-flash', messages, image_data=None)
            
            return jsonify({
                'response': response_text,
                'demo': True,
                'model': 'Gemini Flash 2.5 lite'
            })
        except Exception as e:
            log.error(f"Demo chat error: {e}")
            try:
                # Fallback if the helper fails (e.g., rate limit)
                model = genai.GenerativeModel('gemini-2.5-flash')
                resp = model.generate_content(message)
                return jsonify({'response': resp.text})
            except:
                return jsonify({'response': "I am currently overloaded in Demo Mode. Please sign up for dedicated access!"})
                
    except Exception as e:
        log.exception("Demo chat outer error")
        return jsonify({'error': str(e)}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        
        if not email or not password or not name:
            return jsonify({'success': False, 'error': 'All fields required'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already exists'}), 409
        
        user = User(email=email, password=generate_password_hash(password), name=name)
        try:
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            log.info(f"New user registered: {email}")
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            db.session.rollback()
            log.error(f"Signup error: {e}")
            return jsonify({'success': False, 'error': 'Registration failed'}), 500
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    log.info(f"User logged out: {current_user.email}")
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard - shows chat list"""
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
    available_models = FREE_MODELS
    if current_user.is_premium:
        available_models += PREMIUM_MODELS
    return render_template('dashboard.html', chats=chats, models=available_models, user=current_user)

@app.route('/chat/<int:chat_id>')
@login_required
def chat_view(chat_id):
    """View specific chat with unique URL"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    all_chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
    available_models = FREE_MODELS
    if current_user.is_premium:
        available_models += PREMIUM_MODELS
    return render_template('chat.html', chat=chat, chats=all_chats, messages=chat.messages, models=available_models, user=current_user)

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    """Create new chat"""
    try:
        chat = Chat(user_id=current_user.id, title='New Chat')
        db.session.add(chat)
        db.session.commit()
        log.info(f"New chat created {chat.id} by {current_user.email}")
        return jsonify({
            'success': True,
            'chat_id': chat.id,
            'title': chat.title,
            'url': url_for('chat_view', chat_id=chat.id)
        })
    except Exception as e:
        db.session.rollback()
        log.error(f"Create chat error: {e}")
        return jsonify({'error': 'Failed to create chat'}), 500

@app.route('/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    """Get chat messages"""
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    messages = [{
        'id': m.id,
        'role': m.role,
        'content': m.content,
        'model': m.model,
        'has_image': m.has_image,
        'image_url': m.image_url,
        'created_at': m.created_at.isoformat() if m.created_at else None
    } for m in chat.messages]
    return jsonify({'messages': messages, 'title': chat.title})

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    """Send message and get AI response"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        model_key = data.get('model', 'gemini-2.0-flash-exp')
        chat_id = data.get('chat_id')
        image_data = data.get('image')
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        log.info(f"Chat request: user={current_user.email}, model={model_key}")
        
        # Get or create chat
        if chat_id:
            chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        else:
            chat = Chat(user_id=current_user.id, title=message[:50])
            db.session.add(chat)
            db.session.flush()
        
        # Handle image generation requests
        if message.lower().startswith(('draw', 'generate image', 'create image')):
            image_url = generate_image_prompt(message)
            if image_url:
                response = f"Here is your image for: {message}"
            else:
                response = "Sorry, I couldn't generate that image."
            
            # Save user msg
            user_msg = Message(chat_id=chat.id, role='user', content=message, model='image-gen')
            db.session.add(user_msg)
            
            # Save AI msg
            ai_msg = Message(chat_id=chat.id, role='assistant', content=response, model='image-gen')
            db.session.add(ai_msg)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'chat_id': chat.id,
                'response': response,
                'model': 'image-gen',
                'chat_title': chat.title,
                'url': url_for('chat_view', chat_id=chat.id)
            })
        
        # Check DeepSeek limit
        if 'deepseek' in model_key.lower():
            if not check_deepseek_limit(current_user):
                return jsonify({'error': 'Daily DeepSeek limit reached (50 messages). Upgrade to Premium.'}), 429
        
        # Prepare history for AI
        history = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
        messages = [{'role': m.role, 'content': m.content} for m in history]
        
        # Save user message
        user_msg = Message(
            chat_id=chat.id,
            role='user',
            content=message,
            model=model_key,
            has_image=bool(image_data),
            image_data=image_data if image_data else None
        )
        db.session.add(user_msg)
        db.session.commit()
        
        # Find model config
        all_models = FREE_MODELS + PREMIUM_MODELS if current_user.is_premium else FREE_MODELS
        model_config = next((m for m in all_models if m['model'] == model_key), None)
        
        if not model_config:
            return jsonify({'error': 'Model not found'}), 400
        
        # Check premium requirement
        if not current_user.is_premium and model_config in PREMIUM_MODELS:
            return jsonify({'error': 'This model requires Premium subscription'}), 403
        
        # Call AI API
        if model_config['provider'] == 'google':
            response = call_google_gemini(model_config['model'], messages, image_data=image_data)
        else:
            response = call_openrouter(model_config['model'], messages, image_data=image_data)
        
        # Save AI response
        ai_msg = Message(chat_id=chat.id, role='assistant', content=response, model=model_key)
        db.session.add(ai_msg)
        
        # Update title
        if not chat_id:
            chat.title = message[:50]
        
        chat.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Increment DeepSeek count
        if 'deepseek' in model_key.lower():
            increment_deepseek_count(current_user)
        
        return jsonify({
            'success': True,
            'chat_id': chat.id,
            'response': response,
            'model': model_key,
            'chat_title': chat.title,
            'url': url_for('chat_view', chat_id=chat.id)
        })
        
    except Exception as e:
        db.session.rollback()
        log.exception("Chat error")
        return jsonify({'error': str(e)}), 500

# IMAGE GENERATION ROUTES - GEMINI LOW & HIGH QUALITY
@app.route('/api/generate-gemini-low', methods=['POST'])
def generate_gemini_low():
    """Gemini LOW QUALITY - Fast generation (512x512)"""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400
    
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(
            [prompt],
            generation_config=genai.types.GenerationConfig(
                response_mime_type='image/png',
                response_modalities=['image']
            )
        )
        
        image_data = response.parts[0].inline_data.data
        return send_file(
            BytesIO(image_data),
            mimetype='image/png',
            as_attachment=False,
            download_name='gemini-low.png'
        )
    except Exception as e:
        log.error(f"Gemini LOW image error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-gemini-high', methods=['POST'])
def generate_gemini_high():
    """Gemini HIGH QUALITY - Detailed generation (1024x1024)"""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400
    
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(
            [prompt],
            generation_config=genai.types.GenerationConfig(
                response_mime_type='image/png',
                response_modalities=['image']
            )
        )
        
        image_data = response.parts[0].inline_data.data
        return send_file(
            BytesIO(image_data),
            mimetype='image/png',
            as_attachment=False,
            download_name='gemini-high.png'
        )
    except Exception as e:
        log.error(f"Gemini HIGH image error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stripe-status', methods=['GET'])
def stripe_status():
    """Debug endpoint to check Stripe configuration"""
    return jsonify({
        'stripe_configured': bool(stripe.api_key),
        'has_api_key': bool(os.getenv('STRIPE_SECRET_KEY')),
        'key_format_valid': stripe.api_key.startswith('sk_') if stripe.api_key else False,
        'webhook_secret_set': bool(os.getenv('STRIPE_WEBHOOK_SECRET'))
    })

@app.route('/api/models', methods=['GET'])
@login_required
def get_available_models():
    """Get available AI models for user"""
    models = FREE_MODELS
    if current_user.is_premium:
        models += PREMIUM_MODELS
    return jsonify({'models': models})

@app.route('/api/user', methods=['GET'])
@login_required
def get_user_info():
    """Get current user info"""
    return jsonify({
        'email': current_user.email,
        'name': current_user.name,
        'is_premium': current_user.is_premium,
        'deepseek_count': current_user.deepseek_count,
        'deepseek_limit': 50
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if db.engine else 'disconnected',
        'google_api': 'configured' if google_api_key else 'missing',
        'openrouter_api': 'configured' if openrouter_api_key else 'missing',
        'stripe': 'configured' if stripe.api_key else 'missing',
        'timestamp': datetime.utcnow().isoformat()
    })

# Continue with remaining routes...
@app.context_processor
def utility_processor():
    """Make utilities available in templates"""
    return {
        'now': datetime.utcnow(),
        'stripe_configured': bool(stripe.api_key)
    }

@app.route('/set-model', methods=['POST'])
@login_required
def set_model():
    """Set user's default model"""
    data = request.get_json()
    session['selected_model'] = data.get('model', 'gemini-2.0-flash-exp')
    return jsonify({'success': True})

@app.route('/upload-image', methods=['POST'])
@login_required
def upload_image():
    """Upload image for vision models"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            filename = f"{current_user.id}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image_base64 = encode_image_to_base64(filepath)
            log.info(f"Image uploaded: {filename}")
            return jsonify({'success': True, 'image_data': image_base64, 'filename': filename})
        
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        log.error(f"Image upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    """Rename chat"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
        data = request.get_json()
        new_title = data.get('title', '').strip()
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400
        
        chat.title = new_title
        db.session.commit()
        log.info(f"Chat renamed {chat_id} - {new_title}")
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        log.error(f"Rename chat error: {e}")
        return jsonify({'error': 'Failed to rename chat'}), 500

@app.route('/chat/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    """Delete chat"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
        db.session.delete(chat)
        db.session.commit()
        log.info(f"Chat deleted: {chat_id}")
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        log.error(f"Delete chat error: {e}")
        return jsonify({'error': 'Failed to delete chat'}), 500

@app.route('/checkout')
@login_required
def checkout():
    """Stripe checkout"""
    if not stripe.api_key:
        return jsonify({'error': 'Payment system not configured'}), 503
    
    if current_user.is_premium:
        return redirect(url_for('dashboard'))
    
    try:
        session_obj = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=current_user.email,
            line_items=[
                {
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': 'NexaAI Premium',
                            'description': 'Unlimited access to premium AI models',
                        },
                        'unit_amount': 1999,
                        'recurring': {
                            'interval': 'month',
                        },
                    },
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=url_for('payment_success', external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('dashboard', external=True),
        )
        
        log.info(f"Stripe checkout created for {current_user.email}")
        return redirect(session_obj.url)
    except Exception as e:
        log.error(f"Checkout error: {e}")
        return jsonify({'error': 'Failed to create checkout session'}), 500

@app.route('/payment-success')
@login_required
def payment_success():
    """Payment success callback"""
    session_id = request.args.get('session_id')
    if not session_id or not stripe.api_key:
        return redirect(url_for('dashboard'))
    
    try:
        session_obj = stripe.checkout.Session.retrieve(session_id)
        if session_obj.payment_status == 'paid':
            current_user.is_premium = True
            current_user.subscription_id = session_obj.subscription
            db.session.commit()
            log.info(f"User became premium: {current_user.email}")
    except Exception as e:
        log.error(f"Payment verification error: {e}")
    
    return redirect(url_for('dashboard'))

@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    if not webhook_secret:
        return jsonify({'error': 'Webhook secret not configured'}), 500
    
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        
        if event['type'] == 'customer.subscription.deleted':
            subscription_id = event['data']['object']['id']
            user = User.query.filter_by(subscription_id=subscription_id).first()
            if user:
                user.is_premium = False
                user.subscription_id = None
                db.session.commit()
                log.info(f"Subscription cancelled: {user.email}")
        
        return jsonify({'success': True})
    except Exception as e:
        log.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 400

# Error Handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors without templates"""
    if request.path.startswith('/api'):
        return jsonify({'error': f'Endpoint not found: {request.path}'}), 404
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>404 - Page Not Found - NexaAI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); }
            h1 { font-size: 72px; margin: 0; }
            p { font-size: 20px; margin: 20px 0; }
            a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 12px 24px; border-radius: 8px; display: inline-block; margin-top: 20px; transition: all 0.3s; }
            a:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>404</h1>
            <p>Page Not Found</p>
            <p style="font-size: 16px; opacity: 0.8;">The page you're looking for doesn't exist.</p>
            <a href="/">Go to Homepage</a>
        </div>
    </body>
    </html>
    ''', 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors without templates"""
    log.error(f"Server error: {e}")
    if request.path.startswith('/api'):
        return jsonify({'error': 'Internal server error', 'message': str(e)[:200]}), 500
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>500 - Server Error - NexaAI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif; text-align: center; padding: 50px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); }
            h1 { font-size: 72px; margin: 0; }
            p { font-size: 20px; margin: 20px 0; }
            a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 12px 24px; border-radius: 8px; display: inline-block; margin-top: 20px; transition: all 0.3s; }
            a:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>500</h1>
            <p>Something Went Wrong</p>
            <p style="font-size: 16px; opacity: 0.8;">We're working to fix this issue.</p>
            <a href="/">Go to Homepage</a>
        </div>
    </body>
    </html>
    ''', 500

@app.errorhandler(403)
def forbidden(e):
    """Handle 403 errors"""
    return jsonify({'error': 'Access forbidden', 'message': str(e)}), 403

@app.errorhandler(Exception)
def handle_exception(e):
    """Catch-all exception handler"""
    log.exception("Unhandled exception")
    if request.path.startswith('/api'):
        return jsonify({'error': 'Server error', 'type': type(e).__name__, 'message': str(e)[:200]}), 500
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error - NexaAI</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; background: #f5f5f5; }
            .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }
            h1 { color: #e74c3c; }
            a { color: #3498db; text-decoration: none; display: inline-block; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Unexpected Error</h1>
            <p>Please try again or contact support.</p>
            <a href="/">Go Home</a>
        </div>
    </body>
    </html>
    ''', 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    log.info(f"Starting NexaAI on port {port}, debug={debug}")
    app.run(debug=debug, host='0.0.0.0', port=port)
