import os
import sqlite3
import base64
from datetime import datetime
import urllib.parse

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_from_directory
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
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
uri = os.getenv("DATABASE_URL")
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = uri if uri else 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# API Keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Configure Google Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

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

# ================== MODEL CONFIG ==================

FREE_MODELS = {
    'gemini-flash': {
        'name': 'Gemini 2.0 Flash ⚡',
        'provider': 'google',
        'model_id': 'gemini-2.0-flash-exp',
        'vision': True,
        'image_gen': True,
        'free': True,
        'note': '100% Free'
    },
    'gemini-pro': {
        'name': 'Gemini 1.5 Pro 💎',
        'provider': 'google',
        'model_id': 'gemini-1.5-pro',
        'vision': True,
        'image_gen': True,
        'free': True,
        'note': '100% Free'
    },
    'claude-haiku': {
        'name': 'Claude 3.5 Haiku 🎭',
        'provider': 'openrouter',
        'path': 'anthropic/claude-3.5-haiku:free',
        'vision': False,
        'image_gen': False,
        'free': True,
        'note': '100% Free'
    },
    'deepseek-v3': {
        'name': 'DeepSeek V3 🔍',
        'provider': 'openrouter',
        'path': 'deepseek/deepseek-chat:free',
        'vision': False,
        'image_gen': False,
        'free': True,
        'note': '100% Free'
    },
    'mistral-7b': {
        'name': 'Mistral 7B ⚡',
        'provider': 'openrouter',
        'path': 'mistralai/mistral-7b-instruct:free',
        'vision': False,
        'image_gen': False,
        'free': True,
        'note': '100% Free'
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ================== AI API FUNCTIONS ==================

def call_google_api(model_id, messages, uploaded_file=None):
    """Call Google Gemini API"""
    try:
        model = genai.GenerativeModel(model_id)
        
        # Convert messages
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
                        image_data = part['image_url']['url'].split(',')[1]
                        import io
                        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                        prompt_parts.append(img)
            else:
                prompt_parts.append(msg['content'])
        
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise Exception(f"Google API error: {str(e)}")

def call_openrouter_api(model_path, messages):
    """Call OpenRouter for free models"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexaai.app",
            "X-Title": "NexaAI"
        }
        
        # Convert messages (text only for free models)
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                text = next((p['text'] for p in msg['content'] if p['type'] == 'text'), '')
                formatted_messages.append({"role": msg['role'], "content": text})
            else:
                formatted_messages.append(msg)
        
        payload = {
            "model": model_path,
            "messages": formatted_messages,
            "max_tokens": 2048,
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
        
        if 'error' in data:
            raise Exception(data['error'].get('message', 'OpenRouter API error'))
        
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        raise Exception(f"OpenRouter error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"OpenRouter error: {str(e)}")

def generate_image_with_ai(prompt, model_provider):
    """Generate images using Pollinations (Free)"""
    clean_prompt = prompt.lower()
    for phrase in ['generate image', 'create image', 'make image', 'draw', 'picture of']:
        clean_prompt = clean_prompt.replace(phrase, '').strip()
    
    # Use Pollinations (4K, completely free)
    enhanced = f"{clean_prompt}, highly detailed, 4k uhd, professional, sharp focus, vivid colors"
    encoded = urllib.parse.quote(enhanced)
    image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=2048&height=2048&nologo=true&enhance=true&model=flux"
    
    response = requests.get(image_url, timeout=120)
    response.raise_for_status()
    return response.content, clean_prompt, 'Flux Pro (4K)'

# ================== HELPER FUNCTIONS ==================

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
            except:
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
    """Demo chat without login - Uses Mistral 7B (100% free, no limits)"""
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        # Use Mistral 7B for demo (truly unlimited free)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexaai.app",
            "X-Title": "NexaAI Demo"
        }
        
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            raise Exception(data['error'].get('message', 'API error'))
        
        bot_response = data["choices"][0]["message"]["content"]
        
        return jsonify({
            'response': bot_response,
            'demo': True,
            'model': 'Mistral 7B Demo',
            'message': '✨ Sign up for full access: Vision, Images & 5 AI models!'
        })
    except Exception as e:
        return jsonify({'error': f'Demo unavailable: {str(e)}'}), 500

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
            password=generate_password_hash(password, method='pbkdf2:sha256')
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
    return render_template('dashboard.html', user=current_user, chats=chats, models=FREE_MODELS)

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
        try:
            compress_image(path)
        except:
            pass
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/generate-image', methods=['POST'])
@login_required
def generate_image_route():
    prompt = request.json.get('prompt', '').strip()
    model_key = request.json.get('model', 'gemini-flash')
    
    if not prompt:
        return jsonify({'error': 'Empty prompt'}), 400
    
    model_info = FREE_MODELS.get(model_key)
    if not model_info:
        return jsonify({'error': 'Invalid model'}), 400
    
    if not model_info.get('image_gen'):
        return jsonify({'error': f'{model_info["name"]} cannot generate images. Switch to Gemini models!'}), 400
    
    try:
        image_data, clean_prompt, gen_model = generate_image_with_ai(prompt, model_info['provider'])
        
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
            'image_path': msg.image_path
        } for msg in messages],
        'title': chat.title
    })

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    user_message = request.json.get('message', '')
    selected_model = request.json.get('model', 'gemini-flash')
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

    model_info = FREE_MODELS.get(selected_model)
    if not model_info:
        return jsonify({'error': 'Invalid model'}), 400

    if uploaded_file and not model_info.get('vision'):
        return jsonify({'error': 'Selected model does not support image input'}), 400

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

    history = get_chat_history(chat_id)

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
            return jsonify({'error': f'Image error: {str(e)}'}), 500
    else:
        history.append({"role": "user", "content": user_message})

    try:
        provider = model_info['provider']
        
        if provider == 'google':
            model_id = model_info['model_id']
            bot_response = call_google_api(model_id, history, uploaded_file)
        elif provider == 'openrouter':
            model_path = model_info['path']
            bot_response = call_openrouter_api(model_path, history)
        else:
            return jsonify({'error': 'Unknown provider'}), 500

        assistant_msg = Message(
            chat_id=chat_id,
            role='assistant',
            content=bot_response,
            model=model_info['name']
        )
        db.session.add(assistant_msg)
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

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET')
        )
    except:
        return '', 400

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
        db.create_all()
        print("✅ Database ready!")
        print("🚀 NexaAI - 5 AI Models (100% Free):")
        print("   1. Gemini 2.0 Flash ⚡ (Vision + Images)")
        print("   2. Gemini 1.5 Pro 💎 (Vision + Images)")
        print("   3. Claude 3.5 Haiku 🎭 (Fast)")
        print("   4. DeepSeek V3 🔍 (Reasoning)")
        print("   5. Mistral 7B ⚡ (Instant)")
    app.run(debug=True, port=5000)
