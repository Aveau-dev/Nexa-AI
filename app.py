from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import requests
import stripe
import os
import sqlite3
import base64
from PIL import Image
import io
import traceback

load_dotenv()

app = Flask(__name__)
CORS(app)

# ============ CONFIGURATION ============
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 10,
    'max_overflow': 20
}
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['JSON_AS_ASCII'] = False

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Configure Gemini (optional - will use OpenRouter as fallback)
GEMINI_AVAILABLE = False
try:
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key and gemini_key.strip():
        genai.configure(api_key=gemini_key)
        GEMINI_AVAILABLE = True
        print("✅ Gemini API configured")
except Exception as e:
    print(f"⚠️ Gemini API not available: {e}")
    GEMINI_AVAILABLE = False

# Configure Stripe
stripe_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_key and stripe_key.strip():
    stripe.api_key = stripe_key
    print("✅ Stripe configured")
else:
    print("⚠️ Stripe not configured (payment features disabled)")

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# ============ DATABASE MODELS ============
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    subscription_id = db.Column(db.String(100))
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
    model = db.Column(db.String(50))
    has_image = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(500))
    image_url = db.Column(db.String(1000))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        print(f"Error loading user: {e}")
        return None

# ============ DATABASE MIGRATION ============
def migrate_database():
    """Add missing columns to existing database"""
    try:
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        if not os.path.exists(db_path):
            print("📦 New database will be created")
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check Message table
        cursor.execute("PRAGMA table_info(message)")
        columns = [column[1] for column in cursor.fetchall()]
        
        changes_made = False
        
        if 'has_image' not in columns:
            cursor.execute("ALTER TABLE message ADD COLUMN has_image BOOLEAN DEFAULT 0")
            print("✅ Added has_image column")
            changes_made = True
        
        if 'image_path' not in columns:
            cursor.execute("ALTER TABLE message ADD COLUMN image_path TEXT")
            print("✅ Added image_path column")
            changes_made = True
        
        if 'image_url' not in columns:
            cursor.execute("ALTER TABLE message ADD COLUMN image_url TEXT")
            print("✅ Added image_url column")
            changes_made = True
        
        # Check User table
        cursor.execute("PRAGMA table_info(user)")
        user_columns = [column[1] for column in cursor.fetchall()]
        
        if 'subscription_id' not in user_columns:
            cursor.execute("ALTER TABLE user ADD COLUMN subscription_id TEXT")
            print("✅ Added subscription_id column")
            changes_made = True
        
        if 'is_premium' not in user_columns:
            cursor.execute("ALTER TABLE user ADD COLUMN is_premium BOOLEAN DEFAULT 0")
            print("✅ Added is_premium column")
            changes_made = True
        
        conn.commit()
        conn.close()
        
        if changes_made:
            print("✅ Database migration completed")
        else:
            print("✅ Database schema up to date")
            
    except Exception as e:
        print(f"⚠️ Migration warning: {e}")

# ============ AI MODELS CONFIGURATION ============
FREE_MODELS = {
    'gemini-flash': {
        'name': 'Gemini 2.5 Flash ⚡', 
        'model': 'google/gemini-2.0-flash-exp:free', 
        'type': 'openrouter',
        'description': 'Fast and efficient'
    },
    'chatgpt': {
        'name': 'ChatGPT 4o Mini 🤖', 
        'model': 'openai/gpt-4o-mini:free', 
        'type': 'openrouter',
        'description': 'OpenAI GPT-4o Mini'
    },
    'claude-haiku': {
        'name': 'Claude 3.5 Haiku 🎭', 
        'model': 'anthropic/claude-3.5-haiku:free', 
        'type': 'openrouter',
        'description': 'Fast Claude model'
    },
    'deepseek-v3': {
        'name': 'DeepSeek V3 🔍', 
        'model': 'deepseek/deepseek-chat:free', 
        'type': 'openrouter',
        'description': 'Code specialist'
    },
    'mistral-7b': {
        'name': 'Mistral 7B ⚡', 
        'model': 'mistralai/mistral-7b-instruct:free', 
        'type': 'openrouter',
        'description': 'Balanced performance'
    },
}

PREMIUM_MODELS = {
    'gpt-4o': {
        'name': 'GPT-4o 🚀', 
        'model': 'openai/gpt-4o', 
        'type': 'openrouter',
        'description': 'Most capable GPT-4'
    },
    'gpt-4o-mini': {
        'name': 'GPT-4o Mini 💎', 
        'model': 'openai/gpt-4o-mini', 
        'type': 'openrouter',
        'description': 'Efficient GPT-4'
    },
    'claude-sonnet': {
        'name': 'Claude 3.5 Sonnet 🎭', 
        'model': 'anthropic/claude-3.5-sonnet', 
        'type': 'openrouter',
        'description': 'Best Claude model'
    },
    'gemini-pro-vision': {
        'name': 'Gemini Pro Vision 👁️', 
        'model': 'google/gemini-pro-1.5', 
        'type': 'openrouter',
        'description': 'Multimodal vision'
    },
}


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ============ HELPER FUNCTIONS ============
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def compress_image(image_path, max_size=(1024, 1024), quality=85):
    """Compress image to reduce file size"""
    try:
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        
        # Resize if too large
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save with optimization
        img.save(image_path, optimize=True, quality=quality)
        print(f"✅ Image compressed: {image_path}")
    except Exception as e:
        print(f"⚠️ Image compression failed: {e}")

def generate_chat_title(first_message):
    """Generate chat title from first message"""
    title = first_message.strip()[:50]
    if len(first_message) > 50:
        title += '...'
    return title

def get_chat_history(chat_id, limit=10):
    """Get chat history with proper error handling"""
    try:
        messages = Message.query.filter_by(chat_id=chat_id)\
            .order_by(Message.created_at.desc())\
            .limit(limit)\
            .all()
        
        history = []
        for msg in reversed(messages):
            if msg.has_image and msg.image_path:
                try:
                    image_base64 = encode_image(msg.image_path)
                    if image_base64:
                        history.append({
                            'role': msg.role,
                            'content': [
                                {'type': 'text', 'text': msg.content},
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                            ]
                        })
                    else:
                        history.append({'role': msg.role, 'content': msg.content})
                except Exception as e:
                    print(f"Error processing image in history: {e}")
                    history.append({'role': msg.role, 'content': msg.content})
            else:
                history.append({'role': msg.role, 'content': msg.content})
        
        return history
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []

def call_openrouter(model_path, messages, timeout=60):
    """Call OpenRouter API with retry logic and better error handling"""
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or not api_key.strip():
            raise Exception("OpenRouter API key not configured")
        
        # Format messages for OpenRouter
        formatted_messages = []
        for msg in messages:
            if isinstance(msg['content'], str):
                formatted_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            else:
                # Handle complex content (images) - extract text only
                text_content = next((c['text'] for c in msg['content'] if c['type'] == 'text'), '')
                formatted_messages.append({
                    'role': msg['role'],
                    'content': text_content
                })
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nexaai.app",
                "X-Title": "NexaAI"
            },
            json={
                "model": model_path,
                "messages": formatted_messages
            },
            timeout=timeout
        )
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded. Please try again in a moment.")
        elif response.status_code == 401:
            raise Exception("API authentication failed. Please check your API key.")
        elif response.status_code == 402:
            raise Exception("Insufficient credits. Please check your OpenRouter account.")
        elif response.status_code >= 500:
            raise Exception("AI service temporarily unavailable. Please try again.")
        elif response.status_code != 200:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            raise Exception(f"API error: {error_msg}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.ConnectionError:
        raise Exception("Connection failed. Please check your internet connection.")
    except Exception as e:
        raise Exception(str(e))

def generate_image(prompt):
    """Generate image using Pollinations AI"""
    try:
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={int(datetime.now().timestamp())}"
        
        # Test if URL is accessible
        response = requests.head(image_url, timeout=5)
        if response.status_code == 200:
            return image_url
        else:
            raise Exception("Image generation service unavailable")
            
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(e):
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    if request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

# ============ ROUTES ============
@app.route('/')
def index():
    """Homepage with demo chat"""
    return render_template('index.html')

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    """Demo chat endpoint (no authentication required)"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Check for image generation keywords
        image_keywords = ['generate image', 'create image', 'draw', 'picture of', 'image of', 'make an image']
        if any(keyword in user_message.lower() for keyword in image_keywords):
            try:
                image_url = generate_image(user_message)
                return jsonify({
                    'response': 'Image generated successfully!',
                    'image_url': image_url,
                    'demo': True,
                    'model': 'Nano Banana (Pollinations AI)'
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Text chat - Use OpenRouter
        try:
            response = call_openrouter(
                'google/gemini-flash-2.5', 
                [{'role': 'user', 'content': user_message}],
                timeout=30
            )
            
            return jsonify({
                'response': response,
                'demo': True,
                'model': 'Gemini 2.5 Flash'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Demo chat error: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            email = data.get('email', '').strip().lower()
            name = data.get('name', '').strip()
            password = data.get('password', '')
            
            # Validation
            if not email or not name or not password:
                return jsonify({'error': 'All fields are required'}), 400
            
            if len(password) < 6:
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already exists'}), 400
            
            # Create new user
            new_user = User(
                email=email,
                name=name,
                password=generate_password_hash(password, method='pbkdf2:sha256')
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            print(f"✅ New user registered: {email}")
            
            if request.is_json:
                return jsonify({'success': True, 'redirect': 'login'})
            return redirect(url_for('login'))
        
        except Exception as e:
            db.session.rollback()
            print(f"Signup error: {e}")
            traceback.print_exc()
            
            if request.is_json:
                return jsonify({'error': 'An error occurred. Please try again.'}), 500
            return render_template('signup.html', error='An error occurred. Please try again.')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            
            if not email or not password:
                return jsonify({'error': 'Email and password are required'}), 400
            
            user = User.query.filter_by(email=email).first()
            
            if user and check_password_hash(user.password, password):
                login_user(user, remember=True)
                print(f"✅ User logged in: {email}")
                
                if request.is_json:
                    return jsonify({'success': True, 'redirect': 'dashboard'})
                return redirect(url_for('dashboard'))
            
            if request.is_json:
                return jsonify({'error': 'Invalid email or password'}), 401
            return render_template('login.html', error='Invalid email or password')
        
        except Exception as e:
            print(f"Login error: {e}")
            traceback.print_exc()
            
            if request.is_json:
                return jsonify({'error': 'An error occurred. Please try again.'}), 500
            return render_template('login.html', error='An error occurred. Please try again.')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with chat history"""
    try:
        chats = Chat.query.filter_by(user_id=current_user.id)\
            .order_by(Chat.updated_at.desc()).all()
        
        return render_template('dashboard.html', user=current_user, chats=chats)
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template('dashboard.html', user=current_user, chats=[])

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{current_user.id}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            compress_image(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed. Please try again.'}), 500

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    """Create new chat"""
    try:
        new_chat_obj = Chat(user_id=current_user.id, title='New Chat')
        db.session.add(new_chat_obj)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'chat_id': new_chat_obj.id,
            'title': new_chat_obj.title
        })
    except Exception as e:
        db.session.rollback()
        print(f"New chat error: {e}")
        return jsonify({'error': 'Failed to create chat'}), 500

@app.route('/chat/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    """Rename chat"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        new_title = request.json.get('title', '').strip()
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400
        
        chat.title = new_title
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Rename error: {e}")
        return jsonify({'error': 'Failed to rename chat'}), 500

@app.route('/chat/<int:chat_id>/delete', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    """Delete chat"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Delete uploaded files
        for message in chat.messages:
            if message.has_image and message.image_path:
                try:
                    if os.path.exists(message.image_path):
                        os.remove(message.image_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")
        
        db.session.delete(chat)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Delete error: {e}")
        return jsonify({'error': 'Failed to delete chat'}), 500

@app.route('/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    """Get chat messages"""
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        messages = Message.query.filter_by(chat_id=chat_id)\
            .order_by(Message.created_at.asc()).all()
        
        return jsonify({
            'messages': [{
                'role': msg.role,
                'content': msg.content,
                'model': msg.model,
                'image_url': msg.image_url,
                'created_at': msg.created_at.isoformat()
            } for msg in messages],
            'title': chat.title
        })
    except Exception as e:
        print(f"Get messages error: {e}")
        return jsonify({'error': 'Failed to load messages'}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        selected_model = data.get('model', 'gemini-flash')
        chat_id = data.get('chat_id')
        uploaded_file = data.get('uploaded_file')
        
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
        
        # Check for premium models
        if selected_model in PREMIUM_MODELS and not current_user.is_premium:
            return jsonify({
                'error': 'This model requires Premium subscription',
                'upgrade_required': True,
                'upgrade_url': '/checkout'
            }), 403
        
        # Check for image generation
        image_keywords = ['generate image', 'create image', 'draw', 'picture of', 'image of', 'make an image']
        if any(keyword in user_message.lower() for keyword in image_keywords):
            try:
                image_url = generate_image(user_message)
                
                # Save user message
                user_msg = Message(
                    chat_id=chat_id,
                    role='user',
                    content=user_message
                )
                db.session.add(user_msg)
                
                # Save AI response
                bot_response = "I've generated the image for you!"
                assistant_msg = Message(
                    chat_id=chat_id,
                    role='assistant',
                    content=bot_response,
                    model='Nano Banana',
                    image_url=image_url
                )
                db.session.add(assistant_msg)
                
                if chat_obj.title == 'New Chat':
                    chat_obj.title = generate_chat_title(user_message)
                
                chat_obj.updated_at = datetime.utcnow()
                db.session.commit()
                
                return jsonify({
                    'response': bot_response,
                    'image_url': image_url,
                    'model': 'Nano Banana',
                    'chat_id': chat_id,
                    'chat_title': chat_obj.title
                })
            except Exception as e:
                db.session.rollback()
                print(f"Image generation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Get model info
        if selected_model in FREE_MODELS:
            model_info = FREE_MODELS[selected_model]
        elif selected_model in PREMIUM_MODELS:
            model_info = PREMIUM_MODELS[selected_model]
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
        
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
        
        # Get chat history
        history = get_chat_history(chat_id, limit=10)
        history.append({'role': 'user', 'content': user_message})
        
        # Call AI API
        try:
            bot_response = call_openrouter(model_info['model'], history, timeout=60)
        except Exception as e:
            db.session.rollback()
            print(f"AI API error: {e}")
            return jsonify({'error': str(e)}), 500
        
        # Save assistant message
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
            'chat_title': chat_obj.title,
            'premium': current_user.is_premium
        })
    
    except Exception as e:
        db.session.rollback()
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

# ============ STRIPE CHECKOUT ============
@app.route('/checkout')
@login_required
def checkout():
    """Create Stripe checkout session"""
    try:
        if not stripe.api_key:
            return "Payment system not configured", 503
        
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=current_user.email,
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'NexaAI Premium',
                        'description': 'Unlimited access to GPT-4o, Claude 3.5 Sonnet & all premium models',
                    },
                    'unit_amount': 1999,  # $19.99
                    'recurring': {
                        'interval': 'month',
                    },
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=url_for('payment_success', _external=True),
            cancel_url=url_for('dashboard', _external=True),
        )
        return redirect(checkout_session.url)
    except Exception as e:
        print(f"Checkout error: {e}")
        return str(e), 500

@app.route('/payment-success')
@login_required
def payment_success():
    """Payment success page"""
    return '<h1>Payment Successful!</h1><p>Your premium features are now active.</p><a href="/dashboard">Go to Dashboard</a>'

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Stripe webhook handler"""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        if not webhook_secret:
            return '', 200
        
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except Exception as e:
        print(f"Webhook error: {e}")
        return str(e), 400
    
    # Handle successful payment
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user = User.query.filter_by(email=session['customer_email']).first()
        if user:
            user.is_premium = True
            user.subscription_id = session.get('subscription')
            db.session.commit()
            print(f"✅ User upgraded to premium: {user.email}")
    
    # Handle subscription cancellation
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        user = User.query.filter_by(subscription_id=subscription['id']).first()
        if user:
            user.is_premium = False
            user.subscription_id = None
            db.session.commit()
            print(f"⚠️ User subscription cancelled: {user.email}")
    
    return '', 200

# ============ API ROUTES ============
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available AI models"""
    return jsonify({
        'free': FREE_MODELS,
        'premium': PREMIUM_MODELS
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gemini_available': GEMINI_AVAILABLE,
        'stripe_configured': bool(stripe.api_key)
    })

# ============ MAIN ============
if __name__ == '__main__':
    with app.app_context():
        try:
            # Run migration
            migrate_database()
            
            # Create all tables
            db.create_all()
            
            print("\n" + "="*50)
            print("✅ Database ready!")
            print("✅ All tables created")
            print(f"✅ Gemini: {'Available' if GEMINI_AVAILABLE else 'Using OpenRouter only'}")
            print(f"✅ Stripe: {'Configured' if stripe.api_key else 'Not configured'}")
            print("="*50)
            print("🚀 Starting NexaAI on http://localhost:5000")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            traceback.print_exc()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

