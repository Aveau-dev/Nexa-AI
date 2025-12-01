import os
import sqlite3
import base64
from datetime import datetime
import io
import urllib.parse

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, session, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import stripe
import hashlib
import json

# ================== ENV & FLASK SETUP ==================

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-this')

# DB URL
uri = os.getenv("DATABASE_URL")
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = uri if uri else 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEMO_SESSIONS_FOLDER'] = 'demo_sessions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEMO_SESSIONS_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# API Keys
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')  # For Claude only
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Free from AI Studio
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Free $5 credit
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')  # Optional

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ================== DATABASE MODELS ==================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    subscription_id = db.Column(db.String(100))
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(50))
    has_image = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ================== MIGRATION ==================

def migrate_database():
    try:
        db_path = 'database.db'
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user)")
        columns = [c[1] for c in cursor.fetchall()]
        if 'deepseek_count' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN deepseek_count INTEGER DEFAULT 0")
        if 'deepseek_date' not in columns:
            today = datetime.utcnow().strftime('%Y-%m-%d')
            cursor.execute(f"ALTER TABLE user ADD COLUMN deepseek_date TEXT DEFAULT '{today}'")
        if 'created_at' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN created_at TIMESTAMP")
        cursor.execute("PRAGMA table_info(message)")
        msg_columns = [c[1] for c in cursor.fetchall()]
        if 'has_image' not in msg_columns:
            cursor.execute("ALTER TABLE message ADD COLUMN has_image BOOLEAN DEFAULT 0")
        if 'image_path' not in msg_columns:
            cursor.execute("ALTER TABLE message ADD COLUMN image_path TEXT")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Migration warning: {str(e)}")

# ================== MODEL CONFIG ==================

FREE_MODELS = {
    'gpt-4o-mini': {
        'name': 'GPT-4o Mini',
        'provider': 'openai',
        'vision': True,
        'image_gen': True,
        'free': True
    },
    'gemini-flash': {
        'name': 'Gemini Flash 1.5',
        'provider': 'google',
        'vision': True,
        'image_gen': True,
        'free': True
    },
    'claude-3-haiku': {
        'name': 'Claude 3 Haiku',
        'provider': 'openrouter',
        'path': 'anthropic/claude-3-haiku',
        'vision': False,
        'image_gen': False,
        'free': True
    },
    'deepseek-chat': {
        'name': 'DeepSeek Chat',
        'provider': 'openrouter',
        'path': 'deepseek/deepseek-chat',
        'limit': 50,
        'vision': False,
        'image_gen': False,
        'free': True
    }
}

PREMIUM_MODELS = {
    'gpt-4o': {
        'name': 'GPT-4o',
        'provider': 'openai',
        'vision': True,
        'image_gen': True
    },
    'claude-3.5-sonnet': {
        'name': 'Claude 3.5 Sonnet',
        'provider': 'openrouter',
        'path': 'anthropic/claude-3.5-sonnet',
        'vision': True,
        'image_gen': False
    },
    'gemini-pro': {
        'name': 'Gemini Pro 1.5',
        'provider': 'google',
        'vision': True,
        'image_gen': True
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ================== AI API CALLS ==================

def call_openai_api(model_name, messages, uploaded_file=None):
    """Call OpenAI directly (free $5 credit)"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Handle vision
    formatted_messages = []
    for msg in messages:
        if isinstance(msg.get('content'), list):
            formatted_messages.append(msg)
        else:
            formatted_messages.append(msg)
    
    payload = {
        "model": model_name,
        "messages": formatted_messages,
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def call_google_api(model_name, messages, uploaded_file=None):
    """Call Google AI Studio directly (FREE forever)"""
    import google.generativeai as genai
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Choose model
    if 'pro' in model_name:
        model = genai.GenerativeModel('gemini-1.5-pro')
    else:
        model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Convert messages to Gemini format
    prompt_parts = []
    for msg in messages:
        if msg['role'] == 'system':
            continue
        if isinstance(msg.get('content'), list):
            # Handle vision
            for part in msg['content']:
                if part['type'] == 'text':
                    prompt_parts.append(part['text'])
                elif part['type'] == 'image_url':
                    # Add image
                    image_data = part['image_url']['url'].split(',')[1]
                    import PIL.Image
                    import io
                    img = PIL.Image.open(io.BytesIO(base64.b64decode(image_data)))
                    prompt_parts.append(img)
        else:
            prompt_parts.append(msg['content'])
    
    response = model.generate_content(prompt_parts)
    return response.text

def call_openrouter_api(model_path, messages):
    """OpenRouter for Claude only"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_msg = {
        "role": "system",
        "content": "You are NexaAI. Respond in Markdown with headings and code blocks when useful."
    }
    
    payload = {
        "model": model_path,
        "messages": [system_msg] + messages,
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def generate_image_with_ai(prompt, model_provider):
    """Generate images using the selected AI's native capability"""
    
    # Clean prompt
    clean_prompt = prompt.lower()
    for phrase in ['generate image of', 'create image of', 'make image of', 'draw', 'generate a picture of', 'make an image of', 'create an image of', 'picture of']:
        clean_prompt = clean_prompt.replace(phrase, '').strip()
    
    if model_provider == 'openai':
        # Use DALL-E 3
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json={
                "model": "dall-e-3",
                "prompt": clean_prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "hd"
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        image_url = data['data'][0]['url']
        
        # Download and save
        img_response = requests.get(image_url, timeout=30)
        return img_response.content, clean_prompt, 'DALL-E 3 (HD)'
        
    elif model_provider == 'google':
        # Use Google Imagen via Pollinations (Google doesn't have public Imagen API yet)
        enhanced_prompt = f"{clean_prompt}, highly detailed, 4k uhd, professional, sharp focus"
        encoded = urllib.parse.quote(enhanced_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=2048&height=2048&nologo=true&enhance=true&model=flux"
        
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        return response.content, clean_prompt, 'Flux Pro (4K)'
        
    else:
        # Fallback: Pollinations
        enhanced_prompt = f"{clean_prompt}, highly detailed, professional"
        encoded = urllib.parse.quote(enhanced_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&enhance=true"
        
        response = requests.get(image_url, timeout=90)
        response.raise_for_status()
        return response.content, clean_prompt, 'Stable Diffusion'

# ================== HELPERS ==================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def compress_image(path, max_size=(1024, 1024)):
    img = Image.open(path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img.save(path, optimize=True, quality=85)

def check_deepseek_limit(user):
    today = datetime.utcnow().strftime('%Y-%m-%d')
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    return user.deepseek_count < 50

def generate_chat_title(first_message):
    title = first_message[:50]
    if len(first_message) > 50:
        title += '...'
    return title

def get_chat_history(chat_id, limit=4):
    messages = (
        Message.query.filter_by(chat_id=chat_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    history = []
    for msg in reversed(messages):
        if msg.has_image and msg.image_path:
            try:
                image_base64 = encode_image(msg.image_path)
                history.append({
                    "role": msg.role,
                    "content": [
                        {"type": "text", "text": msg.content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                })
            except Exception:
                history.append({"role": msg.role, "content": msg.content})
        else:
            history.append({"role": msg.role, "content": msg.content})
    return history

# ================== ROUTES: PUBLIC ==================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    user_message = request.json.get('message', '')
    
    try:
        # Use Gemini for demo (completely free)
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(user_message)
        
        return jsonify({
            'response': response.text,
            'demo': True,
            'model': 'Gemini Flash (Demo)',
            'message': 'Sign up for image uploads & more models!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400

        new_user = User(
            email=email,
            name=name,
            password=generate_password_hash(password, method='pbkdf2:sha256'),
            deepseek_date=datetime.utcnow().strftime('%Y-%m-%d')
        )
        db.session.add(new_user)
        db.session.commit()

        if request.is_json:
            return jsonify({'success': True, 'redirect': 'login'})
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email')
        password = data.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            if request.is_json:
                return jsonify({'success': True, 'redirect': 'dashboard'})
            return redirect(url_for('dashboard'))

        if request.is_json:
            return jsonify({'error': 'Invalid credentials'}), 401
        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================== ROUTES: DASHBOARD ==================

@app.route('/dashboard')
@login_required
def dashboard():
    chats = (
        Chat.query.filter_by(user_id=current_user.id)
        .order_by(Chat.updated_at.desc())
        .all()
    )
    if current_user.is_premium:
        available_models = {**FREE_MODELS, **PREMIUM_MODELS}
    else:
        available_models = FREE_MODELS
    return render_template('dashboard.html', user=current_user, chats=chats, models=available_models)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{current_user.id}_{ts}_{filename}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                compress_image(path)
            except Exception:
                pass
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/generate-image', methods=['POST'])
@login_required
def generate_image_route():
    prompt = request.json.get('prompt', '').strip()
    model_key = request.json.get('model', 'gpt-4o-mini')
    
    if not prompt:
        return jsonify({'error': 'Empty prompt'}), 400
    
    # Get model info
    model_info = FREE_MODELS.get(model_key) or PREMIUM_MODELS.get(model_key)
    if not model_info:
        return jsonify({'error': 'Invalid model'}), 400
    
    # Check if model supports image generation
    if not model_info.get('image_gen'):
        return jsonify({'error': f'{model_info["name"]} cannot generate images. Switch to GPT-4o Mini or Gemini Flash!'}), 400
    
    # Check premium access
    if model_key in PREMIUM_MODELS and not current_user.is_premium:
        return jsonify({'error': 'This model requires Premium subscription'}), 403
    
    try:
        image_data, clean_prompt, gen_model = generate_image_with_ai(prompt, model_info['provider'])
        
        # Save image
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"generated_{current_user.id}_{ts}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/uploads/{filename}',
            'prompt': clean_prompt,
            'generator': gen_model
        })
        
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    new_chat_obj = Chat(user_id=current_user.id, title='New Chat')
    db.session.add(new_chat_obj)
    db.session.commit()
    return jsonify({'success': True, 'chat_id': new_chat_obj.id, 'title': new_chat_obj.title})

@app.route('/chat/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    new_title = request.json.get('title')
    chat.title = new_title
    db.session.commit()
    return jsonify({'success': True})

@app.route('/chat/<int:chat_id>/delete', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    for msg in chat.messages:
        if msg.has_image and msg.image_path:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], msg.image_path))
            except Exception:
                pass
    db.session.delete(chat)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    messages = (
        Message.query.filter_by(chat_id=chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return jsonify({
        'messages': [{
            'role': msg.role,
            'content': msg.content,
            'model': msg.model,
            'has_image': msg.has_image,
            'image_path': msg.image_path,
            'created_at': msg.created_at.isoformat()
        } for msg in messages],
        'title': chat.title
    })

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    user_message = request.json.get('message', '')
    selected_model = request.json.get('model', 'gpt-4o-mini')
    chat_id = request.json.get('chat_id')
    uploaded_file = request.json.get('uploaded_file')

    if chat_id:
        chat_obj = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat_obj:
            return jsonify({'error': 'Chat not found'}), 404
    else:
        chat_obj = Chat(user_id=current_user.id, title='New Chat')
        db.session.add(chat_obj)
        db.session.commit()
        chat_id = chat_obj.id

    # Get model info
    model_info = FREE_MODELS.get(selected_model) or PREMIUM_MODELS.get(selected_model)
    if not model_info:
        return jsonify({'error': 'Invalid model'}), 400

    # Check premium
    if selected_model in PREMIUM_MODELS and not current_user.is_premium:
        return jsonify({'error': 'This model requires Premium subscription'}), 403

    # Check DeepSeek limit
    if selected_model == 'deepseek-chat' and not current_user.is_premium:
        if not check_deepseek_limit(current_user):
            return jsonify({'error': '⚠️ Daily DeepSeek limit reached (50/day)'}), 429

    # Check vision
    if uploaded_file and not model_info.get('vision'):
        return jsonify({'error': 'Selected model does not support image input'}), 400

    # Save user message
    user_msg = Message(
        chat_id=chat_id,
        role='user',
        content=user_message,
        has_image=bool(uploaded_file),
        image_path=uploaded_file
    )
    db.session.add(user_msg)

    if chat_obj.title == 'New Chat':
        chat_obj.title = generate_chat_title(user_message)

    # Get history
    history = get_chat_history(chat_id)

    # Add current message with image if present
    if uploaded_file and model_info.get('vision'):
        try:
            image_base64 = encode_image(uploaded_file)
            history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500
    else:
        history.append({"role": "user", "content": user_message})

    try:
        # Route to correct API
        provider = model_info['provider']
        
        if provider == 'openai':
            model_name = 'gpt-4o' if selected_model == 'gpt-4o' else 'gpt-4o-mini'
            bot_response = call_openai_api(model_name, history, uploaded_file)
            
        elif provider == 'google':
            bot_response = call_google_api(selected_model, history, uploaded_file)
            
        elif provider == 'openrouter':
            bot_response = call_openrouter_api(model_info['path'], history)
        
        else:
            return jsonify({'error': 'Unknown provider'}), 500

        # Save assistant message
        assistant_msg = Message(
            chat_id=chat_id,
            role='assistant',
            content=bot_response,
            model=model_info['name']
        )
        db.session.add(assistant_msg)

        if selected_model == 'deepseek-chat' and not current_user.is_premium:
            current_user.deepseek_count += 1

        chat_obj.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'response': bot_response,
            'model': model_info['name'],
            'chat_id': chat_id,
            'chat_title': chat_obj.title
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'AI Error: {str(e)}'}), 500

# ================== STRIPE ==================

@app.route('/checkout')
@login_required
def checkout():
    return render_template('checkout.html')

@app.route('/payment-success')
@login_required
def payment_success():
    return 'Payment successful! <a href="/dashboard">Go to Dashboard</a>'

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET')
        )
    except Exception as e:
        return str(e), 400

    if event['type'] == 'checkout.session.completed':
        sess = event['data']['object']
        user = User.query.filter_by(email=sess['customer_email']).first()
        if user:
            user.is_premium = True
            user.subscription_id = sess.get('subscription')
            db.session.commit()

    return '', 200

# ================== MAIN ==================

if __name__ == '__main__':
    with app.app_context():
        migrate_database()
        db.create_all()
        print("✅ Database ready!")
        print("🚀 Starting NexaAI...")
    app.run(debug=True, port=5000)
