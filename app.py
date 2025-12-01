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
from groq import Groq

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
# API Keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Configure APIs
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Groq client safely
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized")
    except Exception as e:
        print(f"⚠️ Groq initialization failed: {e}")
        print("   Llama 3.1 70B will be unavailable")

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
    'gpt-5': {
        'name': 'GPT-5',
        'provider': 'puter',
        'model_id': 'gpt-5.1',
        'vision': True,
        'image_gen': True,
        'free': True,
        'note': 'Unlimited & Free'
    },
    'claude-sonnet': {
        'name': 'Claude 3.5 Sonnet',
        'provider': 'puter',
        'model_id': 'claude-sonnet-4-5',
        'vision': True,
        'image_gen': False,
        'free': True,
        'note': 'Unlimited & Free'
    },
    'gemini-pro': {
        'name': 'Gemini 1.5 Pro',
        'provider': 'google',
        'model_id': 'gemini-1.5-pro',
        'vision': True,
        'image_gen': True,
        'free': True,
        'note': '1,500/day'
    },
    'gemini-flash': {
        'name': 'Gemini 2.0 Flash',
        'provider': 'google',
        'model_id': 'gemini-2.0-flash-exp',
        'vision': True,
        'image_gen': True,
        'free': True,
        'note': '1,500/day'
    },
    'llama-70b': {
        'name': 'Llama 3.1 70B',
        'provider': 'groq',
        'model_id': 'llama-3.1-70b-versatile',
        'vision': False,
        'image_gen': False,
        'free': True,
        'note': '30/min'
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ================== AI API FUNCTIONS ==================

def call_puter_api(model_id, messages):
    """Call Puter.js for GPT-5 and Claude 3.5 Sonnet (FREE)"""
    try:
        # Convert messages format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                # Handle vision (extract text only for now)
                text = next((p['text'] for p in msg['content'] if p['type'] == 'text'), '')
                formatted_messages.append({"role": msg['role'], "content": text})
            else:
                formatted_messages.append(msg)
        
        response = requests.post('https://api.puter.com/drivers/call', json={
            "interface": "puter-chat-completion",
            "driver": "openai-completion",
            "method": "complete",
            "args": {
                "messages": formatted_messages,
                "model": model_id
            }
        }, timeout=90)
        
        response.raise_for_status()
        data = response.json()
        return data['result']['message']['content']
    except Exception as e:
        raise Exception(f"Puter API error: {str(e)}")

def call_google_api(model_id, messages, uploaded_file=None):
    """Call Google Gemini (FREE forever)"""
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

def call_groq_api(model_id, messages):
    """Call Groq for Llama (FREE 30/min)"""
    try:
        # Convert messages (text only)
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                text = next((p['text'] for p in msg['content'] if p['type'] == 'text'), '')
                formatted_messages.append({"role": msg['role'], "content": text})
            else:
                formatted_messages.append(msg)
        
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")

def generate_image_with_ai(prompt, model_provider):
    """Generate images using Pollinations or DALL-E via Puter"""
    clean_prompt = prompt.lower()
    for phrase in ['generate image', 'create image', 'make image', 'draw', 'picture of']:
        clean_prompt = clean_prompt.replace(phrase, '').strip()
    
    if model_provider == 'puter':
        # Use Puter's DALL-E access
        try:
            response = requests.post('https://api.puter.com/drivers/call', json={
                "interface": "puter-image-generation",
                "driver": "openai-image-generation",
                "method": "generate",
                "args": {
                    "prompt": clean_prompt,
                    "model": "dall-e-3",
                    "size": "1024x1024",
                    "quality": "hd"
                }
            }, timeout=120)
            response.raise_for_status()
            data = response.json()
            image_url = data['result']['url']
            img_response = requests.get(image_url, timeout=30)
            return img_response.content, clean_prompt, 'DALL-E 3 (HD)'
        except:
            pass
    
    # Fallback: Pollinations (4K, free)
    enhanced = f"{clean_prompt}, highly detailed, 4k, professional"
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
        return jsonify({'error': f'{model_info["name"]} cannot generate images'}), 400
    
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
    selected_model = request.json.get('model', 'gpt-5')
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
        model_id = model_info['model_id']
        
        if provider == 'puter':
            bot_response = call_puter_api(model_id, history)
        elif provider == 'google':
            bot_response = call_google_api(model_id, history, uploaded_file)
        elif provider == 'groq':
            bot_response = call_groq_api(model_id, history)
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
        print("🚀 NexaAI with 5 FREE models:")
        print("   1. GPT-5 (Puter - Unlimited)")
        print("   2. Claude 3.5 Sonnet (Puter - Unlimited)")
        print("   3. Gemini 1.5 Pro (Google - 1,500/day)")
        print("   4. Gemini 2.0 Flash (Google - 1,500/day)")
        print("   5. Llama 3.1 70B (Groq - 30/min)")
    app.run(debug=True, port=5000)

