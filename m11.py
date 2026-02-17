# traditional_medicine_ultimate.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_caching import Cache
import json
import os
import hashlib
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import re
import threading
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
import gc
import psutil
import secrets

# ================ تنظیمات پیشرفته ================
class Config:
    SECRET_KEY = secrets.token_urlsafe(64)
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB
    
    # Redis Cache
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = 'redis://localhost:6379/0'
    CACHE_DEFAULT_TIMEOUT = 3600
    CACHE_KEY_PREFIX = 'tm_'
    
    # Performance
    MAX_WORKERS = 8
    REQUEST_TIMEOUT = 30
    BATCH_SIZE = 100
    
    # Paths
    DATA_DIR = 'data'
    LOG_DIR = 'logs'
    TEMP_DIR = 'temp'
    
    # Security
    PASSWORD_SALT = secrets.token_urlsafe(16)
    SESSION_TIMEOUT = timedelta(hours=24)
    
    @staticmethod
    def init_dirs():
        for d in [Config.UPLOAD_FOLDER, Config.DATA_DIR, Config.LOG_DIR, Config.TEMP_DIR]:
            os.makedirs(d, exist_ok=True)

Config.init_dirs()

# ================ راه‌اندازی Flask ================
app = Flask(__name__)
app.config.from_object(Config)
app.permanent_session_lifetime = Config.SESSION_TIMEOUT

CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ================ لاگینگ حرفه‌ای ================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = RotatingFileHandler(
    os.path.join(Config.LOG_DIR, 'app.log'),
    maxBytes=10*1024*1024,
    backupCount=10
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# ================ کش ================
cache = Cache(app)

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password_hash, role='admin'):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

def hash_password(password):
    return hashlib.sha256(f"{password}{Config.PASSWORD_SALT}".encode()).hexdigest()

users = {
    '1': User('1', 'admin', hash_password('admin123'), 'admin'),
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ موتور جستجوی پیشرفته ================
class AdvancedSearchEngine:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 4),
            analyzer='char_wb',
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        self.count_vec = CountVectorizer(
            max_features=10000,
            ngram_range=(1, 3)
        )
        self.vectors = None
        self.keywords = None
        self.lock = threading.RLock()
        
    def fit(self, texts):
        with self.lock:
            if len(texts) > 1:
                self.vectors = self.tfidf.fit_transform(texts)
                self.keywords = self.count_vec.fit_transform(texts)
                return True
        return False
    
    def search(self, query, texts, top_k=5):
        if self.vectors is None or len(texts) == 0:
            return []
        
        with self.lock:
            try:
                q_vec = self.tfidf.transform([query])
                scores = cosine_similarity(q_vec, self.vectors)[0]
                
                results = []
                for i, score in enumerate(scores):
                    if score > 0.15:
                        results.append((i, float(score)))
                
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

# ================ مغز هوش مصنوعی طب سنتی ================
class TraditionalMedicineAI:
    def __init__(self):
        self.data_file = os.path.join(Config.DATA_DIR, 'knowledge.json')
        self.unanswered_file = os.path.join(Config.DATA_DIR, 'unanswered.json')
        self.stats_file = os.path.join(Config.DATA_DIR, 'stats.json')
        
        self.knowledge = []
        self.search_engine = AdvancedSearchEngine()
        self.unanswered = []
        self.stats = {
            'total_queries': 0,
            'answered': 0,
            'unanswered': 0,
            'avg_response_time': 0,
            'popular_categories': Counter(),
            'daily_queries': defaultdict(int),
            'learning_rate': 0
        }
        
        self.lock = threading.RLock()
        self.load_data()
        self.start_background_tasks()
        
    def load_data(self):
        """بارگذاری همه داده‌ها"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.knowledge = json.load(f)
                logger.info(f"✅ {len(self.knowledge)} دانش بارگذاری شد")
            
            if os.path.exists(self.unanswered_file):
                with open(self.unanswered_file, 'r', encoding='utf-8') as f:
                    self.unanswered = json.load(f)
            
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.stats.update(json.load(f))
            
            self.update_search_index()
            
        except Exception as e:
            logger.error(f"خطا در بارگذاری: {e}")
            self.knowledge = self.get_default_knowledge()
    
    def get_default_knowledge(self):
        """دانش پیش‌فرض"""
        return [
            {
                "id": 1,
                "keywords": ["سردرد", "میگرن", "درد سر", "پیشانی"],
                "response": "برای سردرد، دمنوش بابونه با عسل مفید است. روغن بنفشه را به پیشانی بمالید.",
                "category": "اعصاب",
                "usage": 0,
                "effectiveness": 0,
                "created": datetime.now().isoformat()
            },
            {
                "id": 2,
                "keywords": ["دل درد", "دل پیچه", "نفخ", "سوء هاضمه"],
                "response": "عرق نعناع با عسل گرم بنوشید. دمنوش زیره و رازیانه مفید است.",
                "category": "گوارش",
                "usage": 0,
                "effectiveness": 0,
                "created": datetime.now().isoformat()
            },
            {
                "id": 3,
                "keywords": ["بی خوابی", "کم خوابی", "بدخوابی"],
                "response": "شیر گرم با عسل و گلاب قبل از خواب. دمنوش اسطوخدوس و بادرنجبویه.",
                "category": "اعصاب",
                "usage": 0,
                "effectiveness": 0,
                "created": datetime.now().isoformat()
            }
        ]
    
    def update_search_index(self):
        """به‌روزرسانی ایندکس جستجو"""
        texts = [' '.join(item['keywords']) for item in self.knowledge]
        self.search_engine.fit(texts)
    
    def understand(self, text):
        """درک و پاسخ به کاربر"""
        start_time = time.time()
        
        with self.lock:
            self.stats['total_queries'] += 1
            today = datetime.now().strftime('%Y-%m-%d')
            self.stats['daily_queries'][today] += 1
        
        # پیش‌پردازش
        text = self.preprocess(text)
        
        # جستجو
        texts = [' '.join(item['keywords']) for item in self.knowledge]
        results = self.search_engine.search(text, texts)
        
        response_time = time.time() - start_time
        
        with self.lock:
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + response_time) 
                / self.stats['total_queries']
            )
        
        if results:
            idx, score = results[0]
            item = self.knowledge[idx]
            
            with self.lock:
                item['usage'] = item.get('usage', 0) + 1
                self.stats['answered'] += 1
                self.stats['popular_categories'][item['category']] += 1
                self.save_data()
            
            return {
                'answer': item['response'],
                'category': item['category'],
                'confidence': score,
                'id': item['id']
            }
        
        # ثبت سوال بی‌پاسخ
        self.record_unanswered(text)
        
        with self.lock:
            self.stats['unanswered'] += 1
            self.save_data()
        
        return None
    
    def preprocess(self, text):
        """پیش‌پردازش پیشرفته متن"""
        if not text:
            return ""
        
        # حذف کاراکترهای خاص
        text = re.sub(r'[^\w\sآ-یa-zA-Z0-9]', ' ', text)
        
        # نرمال‌سازی
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # حذف کلمات اضافی
        stop_words = {'یک', 'یه', 'چند', 'چطور', 'چجوری', 'لطفا', 'میشه', 'ممنون'}
        words = [w for w in text.split() if w not in stop_words]
        
        return ' '.join(words)
    
    def record_unanswered(self, text):
        """ثبت سوال بی‌پاسخ"""
        with self.lock:
            for item in self.unanswered:
                if item['text'] == text:
                    item['count'] += 1
                    item['last_seen'] = datetime.now().isoformat()
                    break
            else:
                self.unanswered.append({
                    'id': str(uuid.uuid4()),
                    'text': text,
                    'count': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'status': 'pending'
                })
            
            # محدودیت حجم
            if len(self.unanswered) > 1000:
                self.unanswered = self.unanswered[-1000:]
            
            self.save_unanswered()
    
    def learn(self, keywords, response, category):
        """یادگیری دانش جدید"""
        with self.lock:
            keywords_list = [k.strip() for k in keywords.split(',')]
            
            # بررسی تکراری
            for item in self.knowledge:
                if any(k in ' '.join(item['keywords']) for k in keywords_list):
                    return False, "بخشی از این کلمات قبلاً ثبت شده"
            
            new_id = max([i.get('id', 0) for i in self.knowledge] or [0]) + 1
            
            self.knowledge.append({
                'id': new_id,
                'keywords': keywords_list,
                'response': response,
                'category': category,
                'usage': 0,
                'effectiveness': 0,
                'created': datetime.now().isoformat(),
                'last_used': None
            })
            
            self.update_search_index()
            self.save_data()
            
            # پاک کردن سوالات مرتبط از بی‌پاسخ
            self.unanswered = [
                u for u in self.unanswered 
                if not any(k in u['text'] for k in keywords_list)
            ]
            self.save_unanswered()
            
            return True, "✅ یاد گرفتم"
    
    def learn_from_file(self, filepath):
        """یادگیری از فایل"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            count = 0
            errors = []
            
            for line in lines:
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        keywords = parts[0]
                        response = parts[1]
                        category = parts[2] if len(parts) > 2 else 'عمومی'
                        
                        success, msg = self.learn(keywords, response, category)
                        if success:
                            count += 1
                        else:
                            errors.append(msg)
            
            return count, errors
            
        except Exception as e:
            logger.error(f"File learning error: {e}")
            return 0, [str(e)]
    
    def save_data(self):
        """ذخیره دانش"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def save_unanswered(self):
        """ذخیره سوالات بی‌پاسخ"""
        try:
            with open(self.unanswered_file, 'w', encoding='utf-8') as f:
                json.dump(self.unanswered, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Save unanswered error: {e}")
    
    def save_stats(self):
        """ذخیره آمار"""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Save stats error: {e}")
    
    def get_stats(self):
        """گرفتن آمار"""
        with self.lock:
            return {
                'total_knowledge': len(self.knowledge),
                'total_queries': self.stats['total_queries'],
                'answered': self.stats['answered'],
                'unanswered': self.stats['unanswered'],
                'avg_response_time': round(self.stats['avg_response_time'], 3),
                'categories': dict(self.stats['popular_categories'].most_common(10)),
                'daily_queries': dict(list(self.stats['daily_queries'].items())[-7:]),
                'unanswered_count': len(self.unanswered),
                'learning_rate': round(self.stats['answered'] / max(self.stats['total_queries'], 1) * 100, 2)
            }
    
    def start_background_tasks(self):
        """شروع تسک‌های پس‌زمینه"""
        def save_periodically():
            while True:
                time.sleep(300)  # هر ۵ دقیقه
                self.save_stats()
                logger.info("📊 آمار ذخیره شد")
        
        def cleanup_old_data():
            while True:
                time.sleep(3600)  # هر ساعت
                with self.lock:
                    # حذف سوالات قدیمی
                    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                    self.unanswered = [
                        u for u in self.unanswered 
                        if u.get('last_seen', '') > cutoff
                    ]
                    self.save_unanswered()
                logger.info("🧹 پاکسازی انجام شد")
        
        threading.Thread(target=save_periodically, daemon=True).start()
        threading.Thread(target=cleanup_old_data, daemon=True).start()

# ================ نمونه اصلی ================
ai = TraditionalMedicineAI()

# ================ صفحه اصلی - طراحی فوق‌پrofessional ================
@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>پزشک یار | طب سنتی</title>
        
        <!-- فونت‌های مدرن -->
        <link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css" rel="stylesheet">
        
        <!-- اموجی‌ها -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        
        <style>
            /* ========== RESET & VARIABLES ========== */
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
                -webkit-touch-callout: none;
                -webkit-user-select: none;
                -khtml-user-select: none;
                -moz-user-select: none;
                -ms-user-select: none;
                user-select: none;
                -webkit-text-size-adjust: 100%;
                -ms-text-size-adjust: 100%;
                text-size-adjust: 100%;
            }
            
            /* جلوگیری از زوم دو انگشتی */
            body {
                touch-action: pan-y pinch-zoom;
                overflow: hidden;
                position: fixed;
                width: 100%;
                height: 100%;
            }
            
            :root {
                --primary: #2e7d32;
                --primary-light: #4caf50;
                --primary-dark: #1b5e20;
                --secondary: #8bc34a;
                --accent: #ff9800;
                --text: #263238;
                --text-light: #607d8b;
                --background: #f5f5f5;
                --white: #ffffff;
                --shadow: 0 10px 30px rgba(0,0,0,0.1);
                --shadow-hover: 0 20px 40px rgba(0,100,0,0.2);
                --gradient: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                --gradient-light: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
                --border-radius: 24px;
                --border-radius-sm: 16px;
                --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            body {
                font-family: 'Vazirmatn', 'Tahoma', sans-serif;
                background: var(--gradient);
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                color: var(--text);
                line-height: 1.6;
            }
            
            /* ========== CONTAINER ========== */
            .chat-container {
                width: 100%;
                max-width: 450px;
                height: 90vh;
                background: var(--white);
                border-radius: var(--border-radius);
                box-shadow: var(--shadow);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
                margin: 0 auto;
                animation: slideUp 0.5s ease;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* ========== HEADER ========== */
            .chat-header {
                background: var(--gradient);
                padding: 25px 20px;
                color: var(--white);
                position: relative;
                overflow: hidden;
            }
            
            .chat-header::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: rotate 20s linear infinite;
            }
            
            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            
            .header-content {
                position: relative;
                z-index: 1;
                text-align: center;
            }
            
            .header-icon {
                font-size: 3em;
                margin-bottom: 10px;
                display: inline-block;
                filter: drop-shadow(0 5px 10px rgba(0,0,0,0.2));
                animation: float 3s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
            
            .chat-header h1 {
                font-size: 1.8em;
                font-weight: 700;
                margin-bottom: 5px;
                letter-spacing: 1px;
                text-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            
            .chat-header p {
                font-size: 0.9em;
                opacity: 0.9;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 5px;
            }
            
            .header-stats {
                position: absolute;
                top: 15px;
                left: 15px;
                background: rgba(255,255,255,0.2);
                padding: 5px 12px;
                border-radius: 50px;
                font-size: 0.8em;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255,255,255,0.3);
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            /* ========== MESSAGES ========== */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: var(--background);
                scroll-behavior: smooth;
                -webkit-overflow-scrolling: touch;
            }
            
            .message {
                display: flex;
                margin-bottom: 20px;
                animation: messageIn 0.4s ease;
            }
            
            @keyframes messageIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message.bot {
                justify-content: flex-start;
            }
            
            .message-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: var(--gradient-light);
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--white);
                font-size: 1.2em;
                margin-left: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                flex-shrink: 0;
            }
            
            .user .message-avatar {
                background: var(--text-light);
                order: 1;
                margin-left: 0;
                margin-right: 10px;
            }
            
            .message-content {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 20px;
                position: relative;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                line-height: 1.6;
                font-size: 0.95em;
            }
            
            .user .message-content {
                background: var(--gradient);
                color: var(--white);
                border-bottom-right-radius: 5px;
            }
            
            .bot .message-content {
                background: var(--white);
                border-bottom-left-radius: 5px;
            }
            
            .message-content::before {
                content: '';
                position: absolute;
                bottom: 0;
                width: 10px;
                height: 10px;
            }
            
            .user .message-content::before {
                right: -5px;
                background: var(--primary);
                border-radius: 0 0 0 10px;
            }
            
            .bot .message-content::before {
                left: -5px;
                background: var(--white);
                border-radius: 0 0 10px 0;
            }
            
            .message-time {
                font-size: 0.7em;
                opacity: 0.7;
                margin-top: 5px;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .user .message-time {
                justify-content: flex-end;
            }
            
            /* ========== INPUT ========== */
            .chat-input-container {
                padding: 20px;
                background: var(--white);
                border-top: 1px solid rgba(0,0,0,0.05);
            }
            
            .input-wrapper {
                display: flex;
                align-items: center;
                background: var(--background);
                border-radius: 30px;
                padding: 5px 5px 5px 20px;
                border: 2px solid transparent;
                transition: var(--transition);
            }
            
            .input-wrapper:focus-within {
                border-color: var(--primary-light);
                background: var(--white);
                box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.1);
            }
            
            .chat-input {
                flex: 1;
                border: none;
                background: transparent;
                padding: 12px 0;
                font-size: 1em;
                font-family: inherit;
                outline: none;
                color: var(--text);
            }
            
            .chat-input::placeholder {
                color: var(--text-light);
                font-size: 0.9em;
            }
            
            .send-btn {
                width: 45px;
                height: 45px;
                border-radius: 50%;
                background: var(--gradient);
                color: var(--white);
                border: none;
                cursor: pointer;
                font-size: 1.2em;
                transition: var(--transition);
                display: flex;
                align-items: center;
                justify-content: center;
                margin-left: 5px;
            }
            
            .send-btn:hover {
                transform: scale(1.1) rotate(10deg);
                box-shadow: 0 5px 15px rgba(46, 125, 50, 0.4);
            }
            
            .send-btn:active {
                transform: scale(0.9);
            }
            
            /* ========== TYPING ========== */
            .typing-indicator {
                padding: 15px 20px;
                background: var(--white);
                border-radius: 20px;
                display: inline-block;
            }
            
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--primary);
                margin: 0 3px;
                animation: typing 1.4s infinite;
            }
            
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
            
            /* ========== QUICK SUGGESTIONS ========== */
            .quick-suggestions {
                display: flex;
                gap: 10px;
                overflow-x: auto;
                padding: 15px 0;
                scrollbar-width: none;
                -webkit-overflow-scrolling: touch;
            }
            
            .quick-suggestions::-webkit-scrollbar {
                display: none;
            }
            
            .suggestion-chip {
                background: var(--background);
                padding: 8px 16px;
                border-radius: 50px;
                font-size: 0.9em;
                white-space: nowrap;
                cursor: pointer;
                transition: var(--transition);
                border: 1px solid rgba(0,0,0,0.05);
                color: var(--text);
            }
            
            .suggestion-chip:hover {
                background: var(--primary-light);
                color: var(--white);
                border-color: transparent;
            }
            
            /* ========== ADMIN LINK ========== */
            .admin-link {
                position: fixed;
                bottom: 15px;
                right: 15px;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: rgba(0,0,0,0.2);
                color: var(--white);
                display: flex;
                align-items: center;
                justify-content: center;
                text-decoration: none;
                font-size: 1em;
                backdrop-filter: blur(5px);
                transition: var(--transition);
                z-index: 1000;
            }
            
            .admin-link:hover {
                background: var(--primary);
                transform: rotate(90deg);
            }
            
            /* ========== ANIMATIONS ========== */
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            /* ========== RESPONSIVE ========== */
            @media (max-width: 480px) {
                .chat-container {
                    height: 100vh;
                    max-width: 100%;
                    border-radius: 0;
                }
                
                .message-content {
                    max-width: 85%;
                }
            }
            
            /* ========== RTL SUPPORT ========== */
            [dir="rtl"] .send-btn {
                margin-left: 0;
                margin-right: 5px;
            }
            
            [dir="rtl"] .user .message-avatar {
                margin-left: 10px;
                margin-right: 0;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <!-- HEADER -->
            <div class="chat-header">
                <div class="header-stats">
                    <i class="fas fa-leaf"></i>
                    <span id="knowledgeCount">...</span>
                </div>
                
                <div class="header-content">
                    <div class="header-icon">🌿</div>
                    <h1>پزشک یار</h1>
                    <p>
                        <i class="fas fa-mortar-pestle"></i>
                        طب سنتی ایران
                    </p>
                </div>
            </div>
            
            <!-- MESSAGES -->
            <div class="chat-messages" id="messages">
                <div class="message bot">
                    <div class="message-avatar">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <div class="message-content">
                        سلام! نشانه‌های خود را بگویید تا درمان مناسب را پیشنهاد کنم.
                        <div class="message-time">
                            <i class="far fa-clock"></i>
                            همین الان
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- QUICK SUGGESTIONS -->
            <div class="quick-suggestions" id="suggestions">
                <div class="suggestion-chip" onclick="useSuggestion('سردرد دارم')">🤕 سردرد</div>
                <div class="suggestion-chip" onclick="useSuggestion('دل درد دارم')">😖 دل درد</div>
                <div class="suggestion-chip" onclick="useSuggestion('بی خوابی دارم')">😴 بی‌خوابی</div>
                <div class="suggestion-chip" onclick="useSuggestion('تب دارم')">🌡️ تب</div>
                <div class="suggestion-chip" onclick="useSuggestion('سرفه می‌کنم')">🤧 سرفه</div>
            </div>
            
            <!-- INPUT -->
            <div class="chat-input-container">
                <div class="input-wrapper">
                    <input type="text" class="chat-input" id="input" 
                           placeholder="نشانه‌ها را بگویید..."
                           autocomplete="off"
                           onkeypress="if(event.key === 'Enter') sendMessage()">
                    <button class="send-btn" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
        
        <!-- ADMIN LINK (مخفی) -->
        <a href="/admin-login" class="admin-link" title="پنل مدیریت">
            <i class="fas fa-cog"></i>
        </a>
        
        <script>
            // ========== CONFIG ==========
            const CONFIG = {
                typingDelay: 300,
                maxMessageLength: 500,
                animationDuration: 300
            };
            
            // ========== DOM ELEMENTS ==========
            const messages = document.getElementById('messages');
            const input = document.getElementById('input');
            const knowledgeCount = document.getElementById('knowledgeCount');
            
            // ========== STATE ==========
            let isTyping = false;
            let messageQueue = [];
            let sessionId = localStorage.getItem('sessionId') || generateId();
            localStorage.setItem('sessionId', sessionId);
            
            // ========== UTILS ==========
            function generateId() {
                return 'session_' + Math.random().toString(36).substr(2, 9);
            }
            
            function formatTime() {
                return new Date().toLocaleTimeString('fa-IR', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                });
            }
            
            // ========== MESSAGES ==========
            function addMessage(text, isUser = false, isError = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const avatar = isUser ? '👤' : '<i class="fas fa-leaf"></i>';
                const time = formatTime();
                
                let errorClass = isError ? 'error-message' : '';
                
                messageDiv.innerHTML = `
                    <div class="message-avatar">${avatar}</div>
                    <div class="message-content ${errorClass}">
                        ${text}
                        <div class="message-time">
                            <i class="far fa-clock"></i>
                            ${time}
                        </div>
                    </div>
                `;
                
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
                
                return messageDiv;
            }
            
            function showTyping() {
                if (isTyping) return;
                
                isTyping = true;
                
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message bot';
                typingDiv.id = 'typing';
                typingDiv.innerHTML = `
                    <div class="message-avatar">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                `;
                
                messages.appendChild(typingDiv);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function hideTyping() {
                const typing = document.getElementById('typing');
                if (typing) {
                    typing.remove();
                }
                isTyping = false;
            }
            
            // ========== SEND MESSAGE ==========
            async function sendMessage() {
                const text = input.value.trim();
                if (!text || text.length > CONFIG.maxMessageLength) {
                    if (text.length > CONFIG.maxMessageLength) {
                        addMessage('❌ متن خیلی طولانی است', false, true);
                    }
                    return;
                }
                
                // نمایش پیام کاربر
                addMessage(text, true);
                input.value = '';
                
                // نمایش تایپینگ
                showTyping();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Session-ID': sessionId
                        },
                        body: JSON.stringify({ 
                            message: text,
                            session_id: sessionId
                        })
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer, false);
                    } else {
                        addMessage('من هنوز در این مورد یاد نگرفته‌ام. نشانه‌ها را دقیق‌تر بگویید.', false);
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا در ارتباط. دوباره تلاش کنید.', false, true);
                    console.error('Error:', error);
                }
            }
            
            // ========== USE SUGGESTION ==========
            function useSuggestion(text) {
                input.value = text;
                sendMessage();
            }
            
            // ========== LOAD STATS ==========
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const data = await response.json();
                    knowledgeCount.textContent = data.total_knowledge || '...';
                } catch (error) {
                    console.error('Stats error:', error);
                    knowledgeCount.textContent = '🌿';
                }
            }
            
            // ========== AUTO RESIZE ==========
            input.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            });
            
            // ========== PREVENT ZOOM ==========
            document.addEventListener('gesturestart', function(e) {
                e.preventDefault();
            });
            
            document.addEventListener('touchmove', function(e) {
                if (e.touches.length > 1) {
                    e.preventDefault();
                }
            }, { passive: false });
            
            document.addEventListener('touchstart', function(e) {
                if (e.touches.length > 1) {
                    e.preventDefault();
                }
            }, { passive: false });
            
            // ========== INIT ==========
            loadStats();
            
            // فوکوس خودکار
            setTimeout(() => input.focus(), 500);
            
            // بارگذاری سریع
            window.addEventListener('load', function() {
                document.body.classList.add('loaded');
            });
        </script>
    </body>
    </html>
    ''', now=datetime.now())

# ================ API ================
@app.route('/api/chat', methods=['POST'])
@cache.cached(timeout=300, unless=lambda: request.headers.get('X-No-Cache'))
def api_chat():
    """API چت با کش و محدودیت نرخ"""
    client_ip = request.remote_addr
    session_id = request.headers.get('X-Session-ID', 'unknown')
    
    # محدودیت نرخ
    rate_key = f"rate:{client_ip}"
    rate = cache.get(rate_key) or 0
    if rate > 50:  # حداکثر ۵۰ درخواست در ۵ دقیقه
        return jsonify({'error': 'محدودیت درخواست'}), 429
    
    cache.set(rate_key, rate + 1, timeout=300)
    
    # پردازش
    data = request.json
    text = data.get('message', '').strip()
    
    if not text:
        return jsonify({'error': 'پیام خالی'}), 400
    
    result = ai.understand(text)
    
    if result:
        return jsonify({
            'answer': result['answer'],
            'category': result['category'],
            'confidence': result['confidence'],
            'found': True
        })
    else:
        return jsonify({
            'answer': None,
            'found': False
        })

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """آمار عمومی (بدون نیاز به لاگین)"""
    return jsonify(ai.get_stats())

# ================ پنل مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = hash_password(request.form.get('password', ''))
        
        for user in users.values():
            if user.username == username and user.password_hash == password:
                login_user(user, remember=True)
                session.permanent = True
                return redirect(url_for('admin_panel'))
        
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><title>خطا</title></head>
        <body style="font-family:Tahoma;background:#f5f5f5;">
            <h3>❌ خطا در ورود</h3>
            <a href="/admin-login">بازگشت</a>
        </body>
        </html>
        ''')
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ورود مدیریت</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #2e7d32, #1b5e20);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 10px;
                width: 350px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h2 { text-align: center; color: #2e7d32; margin-bottom: 30px; }
            input {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-family: inherit;
            }
            button {
                width: 100%;
                padding: 12px;
                background: #2e7d32;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 ورود</h2>
            <form method="POST">
                <input type="text" name="username" placeholder="نام کاربری" value="admin">
                <input type="password" name="password" placeholder="رمز عبور" value="admin123">
                <button type="submit">ورود</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/admin')
@login_required
def admin_panel():
    stats = ai.get_stats()
    unanswered = ai.unanswered[-20:]
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>پنل مدیریت - پزشک یار</title>
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; }}
            body {{
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }}
            .header {{
                background: #2e7d32;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 2em;
                color: #2e7d32;
                font-weight: bold;
            }}
            .section {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            input, textarea, select {{
                width: 100%;
                padding: 10px;
                margin: 8px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            button {{
                background: #2e7d32;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            .file-upload {{
                border: 2px dashed #2e7d32;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: right;
                border-bottom: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚙️ پنل مدیریت</h1>
            <div>
                <a href="/" target="_blank" style="color:white; margin-right:15px;">🌐 صفحه اصلی</a>
                <a href="/logout" style="color:white;">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{stats['total_knowledge']}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['total_queries']}</div>
                <div>کل سوالات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['answered']}</div>
                <div>پاسخ داده شده</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['unanswered']}</div>
                <div>بی‌پاسخ</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['avg_response_time']}s</div>
                <div>زمان پاسخ</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['learning_rate']}%</div>
                <div>نرخ یادگیری</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="section">
                <h3>➕ یادگیری تکی</h3>
                <form action="/admin/learn" method="POST">
                    <input type="text" name="keywords" placeholder="کلمات کلیدی (با ویرگول)" required>
                    <textarea name="response" rows="4" placeholder="پاسخ درمانی" required></textarea>
                    <select name="category">
                        <option value="اعصاب">اعصاب</option>
                        <option value="گوارش">گوارش</option>
                        <option value="تنفسی">تنفسی</option>
                        <option value="عمومی">عمومی</option>
                    </select>
                    <button type="submit">➕ یاد بده</button>
                </form>
            </div>
            
            <div class="section">
                <h3>📁 یادگیری گروهی</h3>
                <form action="/admin/upload" method="POST" enctype="multipart/form-data">
                    <div class="file-upload" onclick="document.getElementById('file').click()">
                        <p>📤 برای آپلود کلیک کنید</p>
                        <p style="font-size:0.9em;">فرمت: کلمات کلیدی | پاسخ | دسته</p>
                    </div>
                    <input type="file" id="file" name="file" style="display:none;" multiple accept=".txt">
                    <button type="submit">📥 آپلود و یادگیری</button>
                </form>
            </div>
        </div>
        
        <div class="section">
            <h3>❌ سوالات بی‌پاسخ اخیر</h3>
            <table>
                <tr>
                    <th>سوال</th>
                    <th>تعداد</th>
                    <th>آخرین بار</th>
                </tr>
                {"".join([f"<tr><td>{u['text']}</td><td>{u['count']}</td><td>{u['last_seen'][:10]}</td></tr>" for u in unanswered])}
            </table>
        </div>
        
        <div class="section">
            <h3>📊 دسته‌بندی‌های پرطرفدار</h3>
            <table>
                <tr>
                    <th>دسته</th>
                    <th>تعداد</th>
                </tr>
                {"".join([f"<tr><td>{cat}</td><td>{count}</td></tr>" for cat, count in stats['categories'].items()])}
            </table>
        </div>
        
        <script>
            document.querySelector('.file-upload').addEventListener('click', function() {{
                document.getElementById('file').click();
            }});
        </script>
    </body>
    </html>
    '''

@app.route('/admin/learn', methods=['POST'])
@login_required
def admin_learn():
    keywords = request.form['keywords']
    response = request.form['response']
    category = request.form.get('category', 'عمومی')
    
    success, msg = ai.learn(keywords, response, category)
    cache.clear()
    
    if success:
        return redirect(url_for('admin_panel'))
    else:
        return f"❌ {msg} <a href='/admin'>بازگشت</a>"

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    if 'file' not in request.files:
        return "❌ فایلی انتخاب نشده"
    
    files = request.files.getlist('file')
    total = 0
    all_errors = []
    
    for file in files:
        if file and file.filename:
            path = os.path.join(Config.UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(path)
            
            count, errors = ai.learn_from_file(path)
            total += count
            all_errors.extend(errors)
            
            try:
                os.remove(path)
            except:
                pass
    
    cache.clear()
    
    if all_errors:
        return f"✅ {total} مورد یاد گرفتم<br>❌ خطاها: {all_errors[:5]} <a href='/admin'>بازگشت</a>"
    else:
        return f"✅ {total} مورد با موفقیت یاد گرفتم <a href='/admin'>بازگشت</a>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================ HEALTH CHECK ================
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'time': datetime.now().isoformat(),
        'knowledge': len(ai.knowledge),
        'memory': psutil.Process().memory_info().rss / 1024 / 1024
    })

# ================ ERROR HANDLERS ================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'صفحه پیدا نشد'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'خطای سرور'}), 500

# ================ اجرا ================
if __name__ == '__main__':
    stats = ai.get_stats()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     🌿 پزشک یار طب سنتی - نسخه نهایی                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {} مورد                                                 ║
    ║  💬 کل سوالات: {}                                                ║
    ║  ✅ پاسخ داده: {}                                                ║
    ║  ❌ بی‌پاسخ: {}                                                  ║
    ║  ⚡ زمان پاسخ: {}s                                                ║
    ║  🌐 صفحه اصلی: http://localhost:5000                            ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login               ║
    ║  👤 کاربر: admin / رمز: admin123                                ║
    ║  🎯 یادگیری: {}%                                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """.format(
        stats['total_knowledge'],
        stats['total_queries'],
        stats['answered'],
        stats['unanswered'],
        stats['avg_response_time'],
        stats['learning_rate']
    ))
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
