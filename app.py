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

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-this-in-production-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Configure Gemini (optional - will use OpenRouter as fallback)
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini API not configured, using OpenRouter only")

stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============ DATABASE MODELS ============
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
    image_url = db.Column(db.String(1000))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ============ DATABASE MIGRATION ============
def migrate_database():
    """Add missing columns to existing database"""
    try:
        db_path = 'database.db'
        if not os.path.exists(db_path):
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check Message table
        cursor.execute("PRAGMA table_info(message)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'has_image' not in columns:
            cursor.execute("ALTER TABLE message ADD COLUMN has_image BOOLEAN DEFAULT 0")
            print("✅ Added has_image column")
        
        if 'image_path' not in columns:
            cursor.execute("ALTER TABLE message ADD COLUMN image_path TEXT")
            print("✅ Added image_path column")
        
        if 'image_url' not in columns:
            cursor.execute("ALTER TABLE message ADD COLUMN image_url TEXT")
            print("✅ Added image_url column")
        
        # Check User table
        cursor.execute("PRAGMA table_info(user)")
        user_columns = [column[1] for column in cursor.fetchall()]
        
        if 'subscription_id' not in user_columns:
            cursor.execute("ALTER TABLE user ADD COLUMN subscription_id TEXT")
            print("✅ Added subscription_id column")
        
        conn.commit()
        conn.close()
        print("✅ Database migration completed")
    except Exception as e:
        print(f"⚠️ Migration warning: {e}")

# ============ AI MODELS ============
FREE_MODELS = {
    'gemini-flash': {
        'name': 'Gemini 2.5 Flash ⚡', 
        'model': 'google/gemini-flash-2.5', 
        'type': 'openrouter'
    },
    'gemini-pro': {
        'name': 'Gemini 2.5 Pro 🧠', 
        'model': 'google/gemini-pro-2.5', 
        'type': 'openrouter'
    },
    'claude-haiku': {
        'name': 'Claude 3.5 Haiku 🎭', 
        'model': 'anthropic/claude-3.5-haiku', 
        'type': 'openrouter'
    },
    'deepseek-v3': {
        'name': 'DeepSeek V3 🔍', 
        'model': 'deepseek/deepseek-chat', 
        'type': 'openrouter'
    },
    'mistral-7b': {
        'name': 'Mistral 7B ⚡', 
        'model': 'mistralai/mistral-7b-instruct', 
        'type': 'openrouter'
    },
}

PREMIUM_MODELS = {
    'gpt-4o': {
        'name': 'GPT-4o 🚀', 
        'model': 'openai/gpt-4o', 
        'type': 'openrouter'
    },
    'gpt-4o-mini': {
        'name': 'GPT-4o Mini 💎', 
        'model': 'openai/gpt-4o-mini', 
        'type': 'openrouter'
    },
    'claude-sonnet': {
        'name': 'Claude 3.5 Sonnet 🎭', 
        'model': 'anthropic/claude-3.5-sonnet', 
        'type': 'openrouter'
    },
    'gemini-pro-vision': {
        'name': 'Gemini Pro Vision 👁️', 
        'model': 'google/gemini-pro-1.5', 
        'type': 'openrouter'
    },
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ============ HELPER FUNCTIONS ============
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        return None

def compress_image(image_path, max_size=(1024, 1024)):
    """Compress image to reduce file size"""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
    except:
        pass

def generate_chat_title(first_message):
    title = first_message[:50]
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
                except:
                    history.append({'role': msg.role, 'content': msg.content})
            else:
                history.append({'role': msg.role, 'content': msg.content})
        
        return history
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []

def call_openrouter(model_path, messages):
    """Call OpenRouter API with retry logic"""
    try:
        # Format messages for OpenRouter
        formatted_messages = []
        for msg in messages:
            if isinstance(msg['content'], str):
                formatted_messages.append({'role': msg['role'], 'content': msg['content']})
            else:
                # Handle complex content (images)
                formatted_messages.append({'role': msg['role'], 'content': msg['content'][0]['text']})
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nexaai.app",
                "X-Title": "NexaAI"
            },
            json={
                "model": model_path,
                "messages": formatted_messages
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API returned status {response.status_code}: {response.text}")
        
        return response.json()['choices'][0]['message']['content']
    except requests.Timeout:
        raise Exception("Request timed out. Please try again.")
    except Exception as e:
        raise Exception(f"API error: {str(e)}")

def generate_image(prompt):
    """Generate image using Pollinations AI"""
    try:
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
        return image_url
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

# ============ ROUTES ============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    user_message = request.json.get('message')
    
    # Check for image generation
    if any(keyword in user_message.lower() for keyword in ['generate image', 'create image', 'draw', 'picture of', 'image of']):
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
    
    # Text chat - Use OpenRouter as primary
    try:
        response = call_openrouter('google/gemini-flash-1.5', [{'role': 'user', 'content': user_message}])
        
        return jsonify({
            'response': response,
            'demo': True,
            'model': 'Gemini 2.5 Flash'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            email = data.get('email')
            name = data.get('name')
            password = data.get('password')
            
            if not email or not name or not password:
                return jsonify({'error': 'All fields are required'}), 400
            
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
        
        except Exception as e:
            db.session.rollback()
            print(f"Signup error: {e}")
            if request.is_json:
                return jsonify({'error': 'An error occurred. Please try again.'}), 500
            return render_template('signup.html', error='An error occurred. Please try again.')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            email = data.get('email')
            password = data.get('password')
            
            if not email or not password:
                return jsonify({'error': 'Email and password are required'}), 400
            
            user = User.query.filter_by(email=email).first()
            
            if user and check_password_hash(user.password, password):
                login_user(user)
                if request.is_json:
                    return jsonify({'success': True, 'redirect': 'dashboard'})
                return redirect(url_for('dashboard'))
            
            if request.is_json:
                return jsonify({'error': 'Invalid credentials'}), 401
            return render_template('login.html', error='Invalid credentials')
        
        except Exception as e:
            print(f"Login error: {e}")
            if request.is_json:
                return jsonify({'error': 'An error occurred. Please try again.'}), 500
            return render_template('login.html', error='An error occurred. Please try again.')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    chats = Chat.query.filter_by(user_id=current_user.id)\
        .order_by(Chat.updated_at.desc()).all()
    
    return render_template('dashboard.html', user=current_user, chats=chats)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file uploads"""
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

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
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
        return jsonify({'error': str(e)}), 500

@app.route('/chat/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        new_title = request.json.get('title')
        chat.title = new_title
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/chat/<int:chat_id>/delete', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Delete uploaded files
        for message in chat.messages:
            if message.has_image and message.image_path:
                try:
                    os.remove(message.image_path)
                except:
                    pass
        
        db.session.delete(chat)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/chat/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
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
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    try:
        user_message = request.json.get('message')
        selected_model = request.json.get('model', 'gemini-flash')
        chat_id = request.json.get('chat_id')
        uploaded_file = request.json.get('uploaded_file')
        
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
                'upgrade_url': '/checkout'
            }), 403
        
        # Check for image generation
        if any(keyword in user_message.lower() for keyword in ['generate image', 'create image', 'draw', 'picture of', 'image of']):
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
        
        # Get model info
        if selected_model in FREE_MODELS:
            model_info = FREE_MODELS[selected_model]
        elif selected_model in PREMIUM_MODELS:
            model_info = PREMIUM_MODELS[selected_model]
        else:
            return jsonify({'error': 'Invalid model'}), 400
        
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
        history = get_chat_history(chat_id)
        history.append({'role': 'user', 'content': user_message})
        
        # Call API
        bot_response = call_openrouter(model_info['model'], history)
        
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
        return jsonify({'error': str(e)}), 500

# ============ STRIPE CHECKOUT ============
@app.route('/checkout')
@login_required
def checkout():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=current_user.email,
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'NexaAI Premium',
                        'description': 'Unlimited access to GPT-4, Claude 3.5 Sonnet & all premium models',
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
        return redirect(session.url)
    except Exception as e:
        return str(e), 500

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
        session = event['data']['object']
        user = User.query.filter_by(email=session['customer_email']).first()
        if user:
            user.is_premium = True
            user.subscription_id = session.get('subscription')
            db.session.commit()
    
    return '', 200

if __name__ == '__main__':
    with app.app_context():
        migrate_database()
        db.create_all()
        print("✅ Database ready!")
        print("🚀 Starting NexaAI...")
    
    app.run(debug=True, port=5000)

