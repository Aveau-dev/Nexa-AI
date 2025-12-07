from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
import openai
import stripe
import os
import base64
from PIL import Image
import io
import google.generativeai as genai
import anthropic
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-this')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Database configuration with PostgreSQL support
database_url = os.getenv('DATABASE_URL')
if database_url:
    # PostgreSQL
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    logger.info("🐘 Using PostgreSQL database")
else:
    # SQLite fallback
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    logger.info("📁 Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# Configure AI APIs
logger.info("🤖 Configuring AI APIs...")

# Google Generative AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("✅ Google Generative AI configured")
else:
    logger.warning("⚠️ Google API key not found")

# OpenRouter client
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
)
logger.info("✅ OpenRouter configured")

# Anthropic client
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("✅ Anthropic configured")
else:
    anthropic_client = None
    logger.warning("⚠️ Anthropic API key not found")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ==================== DATABASE MODELS ====================

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

# ==================== DATABASE INITIALIZATION ====================

def init_database():
    """Initialize database and create tables"""
    try:
        with app.app_context():
            db.create_all()
            logger.info("✅ Database tables created/verified")
            
            # Verify tables
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            logger.info(f"✅ Tables in database: {tables}")
            logger.info("🎉 Database initialization successful!")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        logger.error("Database error details:")
        import traceback
        traceback.print_exc()

# ==================== AI MODEL CONFIGURATION ====================

FREE_MODELS = {
    'gemini-2.0-flash': {
        'path': 'google/gemini-2.0-flash-exp:free',
        'name': 'Gemini 2.0 Flash',
        'limit': None,
        'vision': True,
        'provider': 'openrouter'
    },
    'gemini-flash': {
        'path': 'google/gemini-flash-1.5',
        'name': 'Gemini Flash 1.5',
        'limit': None,
        'vision': True,
        'provider': 'openrouter'
    },
    'gpt-3.5-turbo': {
        'path': 'openai/gpt-3.5-turbo',
        'name': 'GPT-3.5 Turbo',
        'limit': None,
        'vision': False,
        'provider': 'openrouter'
    },
    'deepseek-chat': {
        'path': 'deepseek/deepseek-chat',
        'name': 'DeepSeek Chat',
        'limit': 50,
        'vision': False,
        'provider': 'openrouter'
    },
}

PREMIUM_MODELS = {
    'gemini-pro': {
        'path': 'gemini-1.5-pro',
        'name': 'Gemini Pro 1.5',
        'vision': True,
        'provider': 'google'
    },
    'gemini-flash-premium': {
        'path': 'gemini-1.5-flash',
        'name': 'Gemini Flash (Premium)',
        'vision': True,
        'provider': 'google'
    },
    'claude-3.5-sonnet': {
        'path': 'claude-3-5-sonnet-20241022',
        'name': 'Claude 3.5 Sonnet',
        'vision': True,
        'provider': 'anthropic'
    },
    'claude-3-opus': {
        'path': 'claude-3-opus-20240229',
        'name': 'Claude 3 Opus',
        'vision': True,
        'provider': 'anthropic'
    },
    'gpt-4o': {
        'path': 'openai/gpt-4o',
        'name': 'GPT-4o',
        'vision': True,
        'provider': 'openrouter'
    },
    'gpt-4o-mini': {
        'path': 'openai/gpt-4o-mini',
        'name': 'GPT-4o Mini',
        'vision': True,
        'provider': 'openrouter'
    },
    'deepseek-r1': {
        'path': 'deepseek/deepseek-r1',
        'name': 'DeepSeek R1',
        'vision': False,
        'provider': 'openrouter'
    },
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf', 'txt', 'doc', 'docx'}

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    """Encode image to base64 for API"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def compress_image(image_path, max_size=(1024, 1024)):
    """Compress image to reduce file size"""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
    except Exception as e:
        logger.error(f"Image compression failed: {e}")

def check_deepseek_limit(user):
    """Check DeepSeek daily limit"""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    if user.deepseek_date != today:
        user.deepseek_count = 0
        user.deepseek_date = today
        db.session.commit()
    return user.deepseek_count < 50

def generate_chat_title(first_message):
    """Generate chat title from first message"""
    title = first_message[:50]
    if len(first_message) > 50:
        title += "..."
    return title

def get_chat_history(chat_id, limit=10):
    """Get chat history for context"""
    messages = Message.query.filter_by(chat_id=chat_id)\
        .order_by(Message.created_at.desc())\
        .limit(limit)\
        .all()
    
    history = []
    for msg in reversed(messages):
        if msg.has_image and msg.image_path:
            # For vision models, include image
            try:
                image_base64 = encode_image(msg.image_path)
                history.append({
                    'role': msg.role,
                    'content': [
                        {'type': 'text', 'text': msg.content},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                    ]
                })
            except:
                history.append({'role': msg.role, 'content': msg.content})
        else:
            history.append({'role': msg.role, 'content': msg.content})
    
    return history

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo-chat', methods=['POST'])
def demo_chat():
    """Demo chat for non-logged-in users"""
    user_message = request.json.get('message')
    
    try:
        response = openrouter_client.chat.completions.create(
            model=FREE_MODELS['gemini-2.0-flash']['path'],
            messages=[{'role': 'user', 'content': user_message}],
        )
        
        return jsonify({
            'response': response.choices[0].message.content,
            'demo': True,
            'model': 'Gemini 2.0 Flash (Demo)',
            'message': 'Sign up for image uploads, document analysis & more!'
        })
    except Exception as e:
        logger.error(f"Demo chat error: {e}")
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
        
        logger.info(f"✅ New user registered: {email}")
        
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
            logger.info(f"✅ User logged in: {email}")
            if request.is_json:
                return jsonify({'success': True, 'redirect': 'dashboard'})
            return redirect(url_for('dashboard'))
        
        logger.warning(f"Failed login attempt for: {email}")
        if request.is_json:
            return jsonify({'error': 'Invalid credentials'}), 401
        return render_template('login.html', error='Invalid credentials')
    
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
    
    if current_user.is_premium:
        available_models = {**FREE_MODELS, **PREMIUM_MODELS}
    else:
        available_models = FREE_MODELS
    
    return render_template('dashboard.html', user=current_user, chats=chats, models=available_models)

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
        
        # Compress images
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                compress_image(filepath)
            except:
                pass
        
        logger.info(f"✅ File uploaded: {filename}")
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded/generated images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    new_chat_obj = Chat(user_id=current_user.id, title='New Chat')
    db.session.add(new_chat_obj)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'chat_id': new_chat_obj.id,
        'title': new_chat_obj.title
    })

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
    
    # Delete associated uploaded files
    for message in chat.messages:
        if message.has_image and message.image_path:
            try:
                os.remove(message.image_path)
            except:
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
    
    messages = Message.query.filter_by(chat_id=chat_id)\
        .order_by(Message.created_at.asc()).all()
    
    return jsonify({
        'messages': [{
            'role': msg.role,
            'content': msg.content,
            'model': msg.model,
            'has_image': msg.has_image,
            'image_path': f'/uploads/{os.path.basename(msg.image_path)}' if msg.has_image and msg.image_path else None,
            'created_at': msg.created_at.isoformat()
        } for msg in messages],
        'title': chat.title
    })

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Main chat endpoint with multi-provider support"""
    user_message = request.json.get('message')
    selected_model = request.json.get('model', 'gemini-2.0-flash')
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
    
    # Check premium access
    if selected_model in PREMIUM_MODELS and not current_user.is_premium:
        return jsonify({
            'error': 'This model requires Premium subscription',
            'upgrade_url': '/checkout'
        }), 403
    
    # Check DeepSeek limit
    if selected_model == 'deepseek-chat' and not current_user.is_premium:
        if not check_deepseek_limit(current_user):
            return jsonify({
                'error': 'Daily DeepSeek limit reached (50/day). Try tomorrow or upgrade to Premium!'
            }), 429
    
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
    
    # Generate chat title
    if chat_obj.title == 'New Chat':
        chat_obj.title = generate_chat_title(user_message)
    
    try:
        # Prepare messages
        history = get_chat_history(chat_id)
        
        # Handle vision models with images
        if uploaded_file and model_info.get('vision'):
            image_base64 = encode_image(uploaded_file)
            
            if model_info['provider'] == 'google':
                # Google Gemini
                model = genai.GenerativeModel(model_info['path'])
                img = Image.open(uploaded_file)
                response = model.generate_content([user_message, img])
                bot_response = response.text
                
            elif model_info['provider'] == 'anthropic':
                # Claude
                with open(uploaded_file, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                response = anthropic_client.messages.create(
                    model=model_info['path'],
                    max_tokens=4096,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': 'image/jpeg',
                                    'data': image_data
                                }
                            },
                            {'type': 'text', 'text': user_message}
                        ]
                    }]
                )
                bot_response = response.content[0].text
                
            else:
                # OpenRouter
                history.append({
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_message},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                    ]
                })
                response = openrouter_client.chat.completions.create(
                    model=model_info['path'],
                    messages=history
                )
                bot_response = response.choices[0].message.content
        
        else:
            # Text-only models
            history.append({'role': 'user', 'content': user_message})
            
            if model_info['provider'] == 'google':
                # Google Gemini
                model = genai.GenerativeModel(model_info['path'])
                response = model.generate_content(user_message)
                bot_response = response.text
                
            elif model_info['provider'] == 'anthropic':
                # Claude
                response = anthropic_client.messages.create(
                    model=model_info['path'],
                    max_tokens=4096,
                    messages=[{'role': 'user', 'content': user_message}]
                )
                bot_response = response.content[0].text
                
            else:
                # OpenRouter
                response = openrouter_client.chat.completions.create(
                    model=model_info['path'],
                    messages=history
                )
                bot_response = response.choices[0].message.content
        
        # Save assistant message
        assistant_msg = Message(
            chat_id=chat_id,
            role='assistant',
            content=bot_response,
            model=model_info['name']
        )
        db.session.add(assistant_msg)
        
        # Update DeepSeek count
        if selected_model == 'deepseek-chat' and not current_user.is_premium:
            current_user.deepseek_count += 1
        
        chat_obj.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'response': bot_response,
            'model': model_info['name'],
            'chat_id': chat_id,
            'chat_title': chat_obj.title,
            'premium': current_user.is_premium,
            'deepseek_remaining': 50 - current_user.deepseek_count if selected_model == 'deepseek-chat' else None
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'AI Error: {str(e)}'}), 500

@app.route('/generate-image', methods=['POST'])
@login_required
def generate_image():
    """Generate AI images using Google Imagen"""
    try:
        prompt = request.json.get('prompt')
        chat_id = request.json.get('chat_id')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Check premium
        if not current_user.is_premium:
            return jsonify({
                'error': 'Image generation requires Premium subscription',
                'upgrade_url': '/checkout'
            }), 403
        
        # Generate using Google Imagen
        model = genai.ImageGenerationModel("imagen-3.0-generate-001")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{current_user.id}_{timestamp}_generated.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        response.images[0]._pil_image.save(filepath)
        
        # Save to chat if provided
        if chat_id:
            chat_obj = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
            if chat_obj:
                user_msg = Message(
                    chat_id=chat_id,
                    role='user',
                    content=f"🎨 Generate image: {prompt}"
                )
                db.session.add(user_msg)
                
                ai_msg = Message(
                    chat_id=chat_id,
                    role='assistant',
                    content=f"Generated image: {prompt}",
                    has_image=True,
                    image_path=filepath,
                    model="Imagen 3.0"
                )
                db.session.add(ai_msg)
                chat_obj.updated_at = datetime.utcnow()
                db.session.commit()
        
        logger.info(f"✅ Image generated for user {current_user.id}")
        return jsonify({
            'success': True,
            'image_url': f'/uploads/{filename}',
            'prompt': prompt
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Image generation error: {e}")
        return jsonify({'error': f'Failed: {str(e)}'}), 500

@app.route('/text-to-speech', methods=['POST'])
@login_required
def text_to_speech():
    """Convert text to speech"""
    text = request.json.get('text')
    
    # Use browser's built-in speech synthesis via JavaScript
    # This is just a placeholder endpoint
    return jsonify({'success': True, 'text': text})

# ==================== STRIPE/PAYMENT ====================

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
                        'description': 'Unlimited access to GPT-4, Claude, Gemini & image generation',
                    },
                    'unit_amount': 1999,
                    'recurring': {'interval': 'month'},
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=url_for('payment_success', _external=True),
            cancel_url=url_for('dashboard', _external=True),
        )
        return redirect(session.url)
    except Exception as e:
        logger.error(f"Checkout error: {e}")
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

# ==================== MAIN ====================

logger.info("✅ Stripe configured")

if __name__ == '__main__':
    init_database()
    print("🚀 Starting NexaAI with all features...")
    app.run(debug=True, port=5000)
