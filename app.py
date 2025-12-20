"""
Nexa-AI - Complete Fixed Version
Production-ready Flask app with proper database schema,
Stripe integration, and AI model support
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy import text as sqltext, desc
import requests
import stripe
import os
import base64
from PIL import Image
import logging
import io
import google.generativeai as genai

# ========== Load Environment ==========
load_dotenv()

# ========== App Setup ==========
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ========== Configuration ==========
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-2025')

# Database Configuration
database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')
if database_url:
    if database_url.startswith('postgres'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    log.info('Using PostgreSQL database')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nexa-ai.db'
    log.info('Using SQLite database')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['JSON_AS_ASCII'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== Database Setup ==========
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# ========== Configure AI APIs ==========
log.info('Configuring AI APIs...')

# Google Generative AI
google_api_key = os.getenv('GOOGLE_API_KEY')
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        log.info('Google Generative AI configured')
    except Exception as e:
        log.error(f'Google AI config error: {e}')
else:
    log.warning('GOOGLE_API_KEY not set')

# OpenRouter API Key
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if openrouter_api_key:
    log.info('OpenRouter configured')
else:
    log.warning('OPENROUTER_API_KEY not set')

# Stripe Configuration (Fixed)
stripe_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_key:
    stripe_key = stripe_key.strip()
    if stripe_key.startswith('sk_'):
        stripe.api_key = stripe_key
        log.info('Stripe configured: %s', stripe_key[:20] + '...')
    else:
        log.error('INVALID Stripe key format (must start with sk_test_ or sk_live_)')
        stripe.api_key = None
else:
    stripe.api_key = None
    log.warning('STRIPE_SECRET_KEY not set in environment')

# ========== Database Models ==========

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    subscription_id = db.Column(db.String(100), nullable=True)
    deepseek_count = db.Column(db.Integer, default=0)
    deepseek_date = db.Column(db.String(10), default=str(datetime.utcnow().date()))
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
    role = db.Column(db.String(20), nullable=False)  # user, assistant, system
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(200), nullable=True)
    has_image = db.Column(db.Boolean, default=False)
    image_data = db.Column(db.Text, nullable=True)  # Base64 for vision models
    image_path = db.Column(db.String(1000), nullable=True)  # Uploaded file path
    image_url = db.Column(db.String(1000), nullable=True)  # Generated image URL
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<Message {self.id} in Chat {self.chat_id}>'


# ========== Database Initialization & Migration ==========

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        log.exception('Error loading user')
        return None


def migrate_message_table_schema():
    """Add missing columns to message table if they don't exist (for existing databases)"""
    try:
        with db.engine.connect() as conn:
            conn.begin()
            
            if 'sqlite' in str(db.engine.url):
                # SQLite: Check and add columns
                result = conn.execute(sqltext('PRAGMA table_info(message)'))
                columns = [row[1] for row in result.fetchall()]
                
                if 'image_url' not in columns:
                    conn.execute(sqltext('ALTER TABLE message ADD COLUMN image_url VARCHAR(1000);'))
                    log.info('✓ Added image_url column to SQLite')
                
                if 'image_path' not in columns:
                    conn.execute(sqltext('ALTER TABLE message ADD COLUMN image_path VARCHAR(1000);'))
                    log.info('✓ Added image_path column to SQLite')
                
                if 'image_data' not in columns:
                    conn.execute(sqltext('ALTER TABLE message ADD COLUMN image_data TEXT;'))
                    log.info('✓ Added image_data column to SQLite')
            
            else:
                # PostgreSQL: Try to add columns
                for col_name, col_type in [('image_url', 'VARCHAR(1000)'), 
                                           ('image_path', 'VARCHAR(1000)'),
                                           ('image_data', 'TEXT')]:
                    try:
                        conn.execute(sqltext(f'ALTER TABLE message ADD COLUMN {col_name} {col_type};'))
                        log.info(f'✓ Added {col_name} column to PostgreSQL')
                    except Exception as e:
                        if 'already exists' not in str(e):
                            log.warning(f'Column {col_name} migration note: {e}')
            
            conn.commit()
            log.info('✓ Message table schema verified')
    except Exception as e:
        log.warning(f'Schema migration note: {e}')


def init_database():
    """Initialize database with proper schema"""
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            log.info('✓ Database tables created/verified')
            
            # Run migration for existing databases
            migrate_message_table_schema()
            
            return True
        except Exception as e:
            log.error(f'Database initialization failed: {e}')
            log.exception('Database error details')
            return False


init_database()

# ========== AI Models Configuration ==========

FREE_MODELS = [
    {
        'name': 'Gemini 2.5 Flash',
        'model': 'gemini-2.5-flash-lite',
        'provider': 'google',
        'vision': True,
        'limit': None,
        'description': 'Fast and efficient Gemini Flash model'
    },
    {
        'name': 'ChatGPT 3.5 Turbo',
        'model': 'openai/gpt-3.5-turbo',
        'provider': 'openrouter',
        'vision': False,
        'limit': None,
        'description': 'OpenAI GPT-3.5 Turbo'
    },
    {
        'name': 'Claude 3 Haiku',
        'model': 'anthropic/claude-3-haiku',
        'provider': 'openrouter',
        'vision': True,
        'limit': None,
        'description': 'Fast Claude model'
    },
    {
        'name': 'DeepSeek Chat',
        'model': 'deepseek/deepseek-chat',
        'provider': 'openrouter',
        'vision': False,
        'limit': 50,
        'description': 'Powerful for code & logic (50/day for free users)'
    }
]

PREMIUM_MODELS = [
    {
        'name': 'GPT-4o',
        'model': 'openai/gpt-4o',
        'provider': 'openrouter',
        'vision': True,
        'description': 'Most capable GPT-4 family model'
    },
    {
        'name': 'GPT-4o Mini',
        'model': 'openai/gpt-4o-mini',
        'provider': 'openrouter',
        'vision': True,
        'description': 'Efficient GPT-4 style model'
    },
    {
        'name': 'Claude 3.5 Sonnet',
        'model': 'anthropic/claude-3.5-sonnet',
        'provider': 'openrouter',
        'vision': True,
        'description': 'Best for coding & analysis'
    },
    {
        'name': 'Claude 3 Opus',
        'model': 'anthropic/claude-3-opus',
        'provider': 'openrouter',
        'vision': True,
        'description': 'Most capable Claude model'
    },
    {
        'name': 'Gemini 1.5 Pro',
        'model': 'gemini-1.5-pro',
        'provider': 'google',
        'vision': True,
        'description': 'Multimodal vision-capable'
    },
    {
        'name': 'DeepSeek R1',
        'model': 'deepseek/deepseek-r1',
        'provider': 'openrouter',
        'vision': False,
        'description': 'Advanced reasoning model'
    }
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

# ========== Utility Functions ==========

def extract_text_content(content):
    """Extract text from various content formats"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
    return ''


def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        log.exception(f'Failed to encode image: {image_path}')
        return None


def call_google_gemini(model_path, messages, image_data=None, image_path=None, timeout=60):
    """Call Google Generative AI (Gemini)"""
    try:
        if not google_api_key:
            raise Exception('Google API key not configured')
        
        model = genai.GenerativeModel(model_path)
        
        # Prepare content
        content = []
        
        # Add text message
        if messages:
            last_msg = messages[-1]
            text = extract_text_content(last_msg.get('content', ''))
            if text:
                content.append(text)
        
        # Add image if provided
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                content.append(image)
            except Exception as e:
                log.warning(f'Failed to process image data: {e}')
        
        if not content:
            content = ['Please respond.']
        
        response = model.generate_content(content, request_options={'timeout': timeout})
        return response.text if response else 'No response'
        
    except Exception as e:
        log.error(f'Gemini API error: {e}')
        return f'Error: {str(e)[:100]}'


def call_openrouter(model_path, messages, image_data=None, image_path=None):
    """Call OpenRouter API"""
    try:
        if not openrouter_api_key:
            raise Exception('OpenRouter API key not configured')
        
        headers = {
            'Authorization': f'Bearer {openrouter_api_key}',
            'Content-Type': 'application/json',
        }
        
        # Prepare messages for OpenRouter
        formatted_messages = []
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                formatted_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': content
                })
            else:
                formatted_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': extract_text_content(content)
                })
        
        # Add image if available (for vision models)
        if image_data and '-vision' in model_path.lower() or 'claude' in model_path.lower():
            formatted_messages[-1]['content'] = [
                {'type': 'text', 'text': formatted_messages[-1]['content']},
                {'type': 'image', 'image': image_data}
            ]
        
        payload = {
            'model': model_path,
            'messages': formatted_messages,
            'temperature': 0.7,
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            error_msg = response.json().get('error', {}).get('message', response.text)
            raise Exception(f'OpenRouter error: {error_msg}')
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        log.error(f'OpenRouter API error: {e}')
        return f'Error: {str(e)[:100]}'


def is_stripe_configured():
    """Check if Stripe is properly configured"""
    if not stripe.api_key:
        return False
    try:
        stripe.Account.retrieve()
        return True
    except Exception as e:
        log.error(f'Stripe API test failed: {e}')
        return False


# ========== Routes ==========

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
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')


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
        
        user = User(
            email=email,
            password=generate_password_hash(password),
            name=name
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            db.session.rollback()
            log.error(f'Signup error: {e}')
            return jsonify({'success': False, 'error': 'Registration failed'}), 500
    
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(desc(Chat.updated_at)).all()
    
    available_models = FREE_MODELS
    if current_user.is_premium:
        available_models = FREE_MODELS + PREMIUM_MODELS
    
    return render_template('dashboard.html', 
                         chats=chats, 
                         models=available_models,
                         user=current_user)


@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    """Create new chat"""
    try:
        chat = Chat(user_id=current_user.id, title='New Chat')
        db.session.add(chat)
        db.session.commit()
        return jsonify({'chatid': chat.id, 'title': chat.title})
    except Exception as e:
        db.session.rollback()
        log.error(f'Create chat error: {e}')
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
        'created_at': m.created_at.isoformat() if m.created_at else None
    } for m in chat.messages]
    
    return jsonify({'messages': messages})


@app.route('/chat', methods=['POST'])
@login_required
def chat_route():
    """Send message and get AI response"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        model_key = data.get('model', 'gemini-2.5-flash-lite')
        chat_id = data.get('chatid')
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create chat
        if chat_id:
            chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
        else:
            chat = Chat(user_id=current_user.id, title=message[:50])
            db.session.add(chat)
            db.session.flush()
        
        # Save user message
        user_msg = Message(
            chat_id=chat.id,
            role='user',
            content=message,
            model=model_key
        )
        db.session.add(user_msg)
        db.session.commit()
        
        # Find model config
        all_models = FREE_MODELS + (PREMIUM_MODELS if current_user.is_premium else [])
        model_config = next((m for m in all_models if m['model'] == model_key), None)
        
        if not model_config:
            return jsonify({'error': 'Model not found'}), 400
        
        # Prepare history for AI
        history = Message.query.filter_by(chat_id=chat.id).order_by(Message.created_at).all()
        messages = [{'role': m.role, 'content': m.content} for m in history]
        
        # Call AI API
        if model_config['provider'] == 'google':
            response = call_google_gemini(model_config['model'], messages)
        else:
            response = call_openrouter(model_config['model'], messages)
        
        # Save AI response
        ai_msg = Message(
            chat_id=chat.id,
            role='assistant',
            content=response,
            model=model_key
        )
        db.session.add(ai_msg)
        
        # Update chat title if new
        if not chat_id and message:
            chat.title = message[:50] + ('...' if len(message) > 50 else '')
        
        db.session.commit()
        
        return jsonify({
            'chatid': chat.id,
            'response': response,
            'model': model_key,
            'chattitle': chat.title
        })
        
    except Exception as e:
        db.session.rollback()
        log.exception('Chat error')
        return jsonify({'error': str(e)[:100]}), 500


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
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        log.error(f'Rename chat error: {e}')
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
        log.error(f'Delete chat error: {e}')
        return jsonify({'error': 'Failed to delete chat'}), 500


@app.route('/set-model', methods=['POST'])
@login_required
def set_model():
    """Set user's default model"""
    data = request.get_json()
    session['selected_model'] = data.get('model', 'gemini-2.5-flash-lite')
    return jsonify({'success': True})


# ========== Stripe Routes ==========

@app.route('/checkout')
@login_required
def checkout():
    """Stripe checkout"""
    if not stripe.api_key:
        return render_template('error.html', 
            error='Payment system is not configured. Please contact support.'), 503
    
    if current_user.is_premium:
        return redirect(url_for('dashboard'))
    
    try:
        if not stripe.api_key.startswith('sk_'):
            log.error('Invalid Stripe key format in checkout')
            return render_template('error.html', 
                error='Payment system configuration error.'), 500
        
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
    
    except stripe.error.AuthenticationError as e:
        log.error(f'Stripe Authentication Error: {e}')
        return render_template('error.html', 
            error='Payment authentication failed. Please check Stripe configuration.'), 500
    
    except stripe.error.RateLimitError as e:
        log.error(f'Stripe Rate Limit: {e}')
        return render_template('error.html', 
            error='Too many requests to payment service. Try again later.'), 429
    
    except stripe.error.StripeError as e:
        log.error(f'Stripe Error: {e}')
        return render_template('error.html', 
            error=f'Payment error: {str(e)[:100]}'), 500
    
    except Exception as e:
        log.exception('Checkout failed')
        return render_template('error.html', 
            error='Failed to create checkout session'), 500


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
            log.info(f'User {current_user.email} became premium')
    
    except Exception as e:
        log.error(f'Payment verification error: {e}')
    
    return redirect(url_for('dashboard'))


# ========== Debug Endpoints ==========

@app.route('/api/stripe-status', methods=['GET'])
def stripe_status():
    """Debug endpoint to check Stripe configuration"""
    return jsonify({
        'stripe_configured': bool(stripe.api_key),
        'has_api_key': bool(os.getenv('STRIPE_SECRET_KEY')),
        'key_format_valid': stripe.api_key.startswith('sk_') if stripe.api_key else False,
        'webhook_secret_set': bool(os.getenv('STRIPE_WEBHOOK_SECRET')),
        'error': None if stripe.api_key else 'STRIPE_SECRET_KEY not set in environment'
    })


@app.route('/api/models', methods=['GET'])
@login_required
def get_available_models():
    """Get available AI models for user"""
    models = FREE_MODELS
    if current_user.is_premium:
        models = FREE_MODELS + PREMIUM_MODELS
    return jsonify({'models': models})


@app.route('/api/user', methods=['GET'])
@login_required
def get_user_info():
    """Get current user info"""
    return jsonify({
        'email': current_user.email,
        'name': current_user.name,
        'is_premium': current_user.is_premium
    })


# ========== Error Handlers ==========

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    log.error(f'Server error: {e}')
    return jsonify({'error': 'Server error'}), 500


# ========== Main ==========

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
