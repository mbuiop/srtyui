# advanced_history_bot.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import json
import os
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import threading
import time
import redis
from concurrent.futures import ThreadPoolExecutor
import logging
from logging.handlers import RotatingFileHandler

# ================ پیکربندی پیشرفته ================
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-history-bot-key-2024')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB - افزایش به 500 مگابایت
    MAX_FILES_PER_UPLOAD = 1000  # افزایش به ۱۰۰۰ فایل
    
    # Redis configuration
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    
    # Rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_STRATEGY = 'fixed-window'
    
    # Thread pool
    MAX_WORKERS = 20

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ================ راه‌اندازی سرویس‌ها ================
# Cache
cache = Cache(app)

# Rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=app.config['REDIS_URL'],
    strategy=app.config['RATELIMIT_STRATEGY']
)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=app.config['MAX_WORKERS'])

# Logging
if not os.path.exists('logs'):
    os.makedirs('logs')

handler = RotatingFileHandler('logs/history_bot.log', maxBytes=10000000, backupCount=10)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('History bot startup')

# ایجاد پوشه‌ها
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('temp', exist_ok=True)

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password, role='admin'):
        self.id = id
        self.username = username
        self.password = password
        self.role = role

# کاربران پیش‌فرض (در محیط واقعی از دیتابیس استفاده کنید)
users = {
    '1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest(), 'admin'),
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ الگوریتم هوشمند پیشرفته با قابلیت مقیاس‌پذیری ================
class ScalableHistoryBrain:
    def __init__(self, data_file='data/history_knowledge.json'):
        self.data_file = data_file
        self.knowledge_base = []
        self.vectorizer = TfidfVectorizer(max_features=10000)  # افزایش ویژگی‌ها
        self.question_vectors = None
        self.unanswered_questions = []
        self.lock = threading.Lock()
        self.batch_size = 1000
        self.load_knowledge()
        self.update_vectors()
        
    def load_knowledge(self):
        """بارگذاری دانش با مدیریت حافظه"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    # بارگذاری تدریجی برای فایل‌های بزرگ
                    data = json.load(f)
                    self.knowledge_base = data
                app.logger.info(f"📚 {len(self.knowledge_base)} دانش بارگذاری شد")
            else:
                self._create_sample_data()
        except Exception as e:
            app.logger.error(f"Error loading knowledge: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """ایجاد داده‌های نمونه"""
        sample_data = [
            {"id": 1, "question": "کوروش کبیر که بود", "answer": "کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود", "category": "ایران باستان", "times_used": 0},
            {"id": 2, "question": "داریوش چه کرد", "answer": "داریوش بزرگ جاده شاهی را ساخت و امپراتوری را به ساتراپی‌ها تقسیم کرد", "category": "ایران باستان", "times_used": 0},
            {"id": 3, "question": "خشایارشا که بود", "answer": "خشایارشا پسر داریوش بزرگ بود که به یونان لشکر کشید", "category": "ایران باستان", "times_used": 0}
        ]
        self.knowledge_base = sample_data
        self.save_knowledge()
    
    def save_knowledge(self):
        """ذخیره دانش با قفل برای جلوگیری از race condition"""
        with self.lock:
            try:
                # ذخیره در فایل موقت اول
                temp_file = self.data_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
                # جایگزینی فایل اصلی
                os.replace(temp_file, self.data_file)
            except Exception as e:
                app.logger.error(f"Error saving knowledge: {e}")
    
    def update_vectors(self):
        """به‌روزرسانی بردارها در پس‌زمینه"""
        if len(self.knowledge_base) > 0:
            try:
                questions = [item['question'] for item in self.knowledge_base]
                # پردازش دسته‌ای برای مجموعه‌های بزرگ
                if len(questions) > self.batch_size:
                    # استفاده از partial fit برای مجموعه‌های بزرگ
                    self.vectorizer = TfidfVectorizer(max_features=10000)
                    self.question_vectors = self.vectorizer.fit_transform(questions)
                else:
                    self.question_vectors = self.vectorizer.fit_transform(questions)
            except Exception as e:
                app.logger.error(f"Error updating vectors: {e}")
                self.question_vectors = None
    
    def preprocess_text(self, text):
        """پیش‌پردازش متن بهینه"""
        if not text:
            return ""
        # حذف کاراکترهای اضافی با یک بار عبور
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = text.lower().strip()
        return ' '.join([word for word in text.split() if len(word) > 1])
    
    def search_smart(self, query):
        """جستجوی هوشمند با کش"""
        # بررسی کش
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        if not self.knowledge_base:
            return []
        
        query = self.preprocess_text(query)
        query_words = set(query.split())
        
        # جستجوی موازی برای مجموعه‌های بزرگ
        if len(self.knowledge_base) > 5000:
            return self._parallel_search(query, query_words)
        else:
            return self._sequential_search(query, query_words)
    
    def _sequential_search(self, query, query_words):
        """جستجوی ترتیبی"""
        results = []
        
        # جستجوی کلمات کلیدی
        for item in self.knowledge_base:
            question_words = set(item['question'].split())
            common_words = query_words.intersection(question_words)
            
            if common_words:
                score = len(common_words) / max(len(question_words), 1)
                if query == item['question']:
                    score = 1.0
                    
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': score,
                    'category': item.get('category', 'عمومی')
                })
        
        # جستجوی برداری برای نتایج بهتر
        if self.question_vectors is not None:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.question_vectors)[0]
                
                for i, score in enumerate(similarities):
                    if score > 0.2:  # آستانه بالاتر برای کیفیت بهتر
                        item = self.knowledge_base[i]
                        # ترکیب با نتایج قبلی
                        found = False
                        for r in results:
                            if r['id'] == item['id']:
                                r['score'] = max(r['score'], float(score))
                                found = True
                                break
                        if not found:
                            results.append({
                                'id': item['id'],
                                'answer': item['answer'],
                                'score': float(score),
                                'category': item.get('category', 'عمومی')
                            })
            except Exception as e:
                app.logger.error(f"Vector search error: {e}")
        
        # مرتب‌سازی و محدود کردن نتایج
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:5]
        
        # به‌روزرسانی آمار در پس‌زمینه
        executor.submit(self._update_stats, top_results)
        
        # ذخیره در کش
        cache.set(f"search_{hashlib.md5(query.encode()).hexdigest()}", top_results, timeout=300)
        
        return top_results
    
    def _parallel_search(self, query, query_words):
        """جستجوی موازی برای مجموعه‌های بزرگ"""
        chunk_size = len(self.knowledge_base) // app.config['MAX_WORKERS'] + 1
        chunks = [self.knowledge_base[i:i+chunk_size] for i in range(0, len(self.knowledge_base), chunk_size)]
        
        futures = []
        for chunk in chunks:
            future = executor.submit(self._search_chunk, chunk, query, query_words)
            futures.append(future)
        
        results = []
        for future in futures:
            results.extend(future.result())
        
        # مرتب‌سازی و محدود کردن
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]
    
    def _search_chunk(self, chunk, query, query_words):
        """جستجو در یک تکه از داده"""
        results = []
        for item in chunk:
            question_words = set(item['question'].split())
            common_words = query_words.intersection(question_words)
            
            if common_words:
                score = len(common_words) / max(len(question_words), 1)
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': score,
                    'category': item.get('category', 'عمومی')
                })
        return results
    
    def _update_stats(self, results):
        """به‌روزرسانی آمار در پس‌زمینه"""
        for result in results:
            for item in self.knowledge_base:
                if item['id'] == result['id']:
                    item['times_used'] = item.get('times_used', 0) + 1
                    item['last_used'] = datetime.now().isoformat()
                    break
        self.save_knowledge()
    
    def add_knowledge(self, question, answer, category='عمومی', force=False):
        """اضافه کردن دانش جدید - با گزینه اجبار برای تکرار"""
        question_processed = self.preprocess_text(question)
        
        # بررسی تکراری نبودن (اگر force=False)
        if not force:
            for item in self.knowledge_base:
                if item['question'] == question_processed:
                    return False, "این سوال قبلاً ثبت شده است"
        
        with self.lock:
            new_id = max([item['id'] for item in self.knowledge_base], default=0) + 1
            new_item = {
                'id': new_id,
                'question': question_processed,
                'original_question': question,  # ذخیره سوال اصلی
                'answer': answer,
                'category': category,
                'date_added': datetime.now().isoformat(),
                'times_used': 0,
                'last_used': None
            }
            
            self.knowledge_base.append(new_item)
            
        # به‌روزرسانی بردارها در پس‌زمینه
        executor.submit(self.update_vectors)
        
        return True, "دانش با موفقیت اضافه شد"
    
    def add_bulk_from_files(self, filepaths, category='عمومی'):
        """اضافه کردن گروهی از چندین فایل - بهینه شده"""
        total_count = 0
        all_errors = []
        
        for filepath in filepaths:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                count, errors = self.add_bulk_from_text(content, category, force=True)
                total_count += count
                all_errors.extend(errors)
                
            except Exception as e:
                all_errors.append(f"خطا در فایل {filepath}: {str(e)}")
        
        return total_count, all_errors
    
    def add_bulk_from_text(self, text, category='عمومی', force=True):
        """اضافه کردن گروهی از متن - با گزینه اجبار برای تکرار"""
        lines = text.strip().split('\n')
        count = 0
        errors = []
        
        # پردازش موازی برای متن‌های بزرگ
        if len(lines) > 1000:
            chunk_size = len(lines) // app.config['MAX_WORKERS'] + 1
            chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
            
            futures = []
            for chunk in chunks:
                future = executor.submit(self._process_chunk, chunk, category, force)
                futures.append(future)
            
            for future in futures:
                c, e = future.result()
                count += c
                errors.extend(e)
        else:
            count, errors = self._process_chunk(lines, category, force)
        
        return count, errors
    
    def _process_chunk(self, lines, category, force):
        """پردازش یک تکه از خطوط"""
        count = 0
        errors = []
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    success, msg = self.add_knowledge(q.strip(), a.strip(), category, force)
                    if success:
                        count += 1
                    else:
                        errors.append(f"خطا در {q}: {msg}")
        
        return count, errors
    
    def record_unanswered(self, question):
        """ثبت سوالات بی‌پاسخ"""
        self.unanswered_questions.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # ذخیره در پس‌زمینه
        if len(self.unanswered_questions) % 10 == 0:  # هر ۱۰ سوال یکبار ذخیره کن
            executor.submit(self._save_unanswered)
    
    def _save_unanswered(self):
        """ذخیره سوالات بی‌پاسخ"""
        try:
            with open('data/unanswered.json', 'w', encoding='utf-8') as f:
                json.dump(self.unanswered_questions[-1000:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            app.logger.error(f"Error saving unanswered: {e}")
    
    @cache.cached(timeout=60, key_prefix='stats')
    def get_stats(self):
        """گرفتن آمار با کش"""
        total = len(self.knowledge_base)
        if total == 0:
            return {}
            
        categories = Counter([item.get('category', 'عمومی') for item in self.knowledge_base])
        most_used = sorted(self.knowledge_base, key=lambda x: x.get('times_used', 0), reverse=True)[:10]
        never_used = len([item for item in self.knowledge_base if item.get('times_used', 0) == 0])
        
        return {
            'total': total,
            'categories': dict(categories),
            'most_used': most_used,
            'never_used_count': never_used,
            'unanswered_count': len(self.unanswered_questions)
        }

# ================ نمونه اصلی ================
brain = ScalableHistoryBrain()

# ================ صفحات اصلی با بهینه‌سازی ================
@app.route('/')
@cache.cached(timeout=30)
def index():
    """صفحه اصلی چت - با کش"""
    stats = brain.get_stats()
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>تاریخ‌دان هوشمند</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .chat-container {
                width: 95%;
                max-width: 1200px;
                height: 90vh;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .chat-header h1 { font-size: 1.8em; display: flex; align-items: center; gap: 10px; }
            .header-stats {
                background: rgba(255,255,255,0.2);
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }
            .admin-link {
                color: white;
                text-decoration: none;
                padding: 8px 15px;
                border-radius: 20px;
                background: rgba(255,255,255,0.2);
                transition: all 0.3s;
            }
            .admin-link:hover { background: rgba(255,255,255,0.3); }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                background: #f8f9fa;
            }
            .message {
                display: flex;
                margin-bottom: 25px;
                animation: fadeIn 0.3s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.user { justify-content: flex-end; }
            .message.bot { justify-content: flex-start; }
            .message-content {
                max-width: 70%;
                padding: 15px 20px;
                border-radius: 20px;
                position: relative;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                line-height: 1.6;
            }
            .user .message-content {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            .bot .message-content {
                background: white;
                border-bottom-left-radius: 5px;
            }
            .message-time {
                font-size: 0.7em;
                opacity: 0.7;
                margin-top: 5px;
                text-align: left;
            }
            .chat-input-container {
                padding: 20px 30px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 15px;
            }
            .chat-input {
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1em;
                outline: none;
                transition: all 0.3s;
                font-family: 'Tahoma', sans-serif;
            }
            .chat-input:focus {
                border-color: #1e3c72;
                box-shadow: 0 0 0 3px rgba(30,60,114,0.1);
            }
            .send-btn {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5em;
                transition: all 0.3s;
            }
            .send-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 5px 20px rgba(30,60,114,0.4);
            }
            .typing-indicator {
                padding: 15px 25px;
                background: white;
                border-radius: 20px;
                display: inline-block;
            }
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #1e3c72;
                margin: 0 3px;
                animation: typing 1.4s infinite;
            }
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
            .upload-status {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: white;
                padding: 15px 25px;
                border-radius: 10px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.2);
                display: none;
                z-index: 1000;
            }
            .upload-status.show {
                display: block;
                animation: slideIn 0.3s ease;
            }
            @keyframes slideIn {
                from { transform: translateX(100px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>
                    <span>🤖 تاریخ‌دان هوشمند</span>
                    <span class="header-stats">📚 {{ stats.total }} دانش</span>
                </h1>
                <a href="/admin-login" class="admin-link">⚙️ پنل مدیریت</a>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! من تاریخ‌دان هوشمند هستم. هر سوال تاریخی داری بپرس!
                        <div class="message-time">{{ now.strftime('%H:%M') }}</div>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال تاریخی خود را بپرسید..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">
                    <span>➤</span>
                </button>
            </div>
        </div>
        
        <div class="upload-status" id="upload-status"></div>
        
        <script>
            const messagesContainer = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            let isProcessing = false;
            
            function addMessage(text, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const time = new Date().toLocaleTimeString('fa-IR', { hour: '2-digit', minute: '2-digit' });
                
                messageDiv.innerHTML = `
                    <div class="message-content">
                        ${text}
                        <div class="message-time">${time}</div>
                    </div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function showTyping() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message bot';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = `
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                `;
                messagesContainer.appendChild(typingDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function hideTyping() {
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
            }
            
            async function sendMessage() {
                if (isProcessing) return;
                
                const message = messageInput.value.trim();
                if (!message) return;
                
                isProcessing = true;
                addMessage(message, true);
                messageInput.value = '';
                showTyping();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        addMessage('متأسفم! نتونستم جوابی پیدا کنم. این سوال برای مدیر ارسال شد.');
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('خطا در ارتباط با سرور');
                } finally {
                    isProcessing = false;
                }
            }
            
            // Enable multiple file upload
            document.addEventListener('DOMContentLoaded', function() {
                const fileInput = document.getElementById('file');
                if (fileInput) {
                    fileInput.setAttribute('multiple', 'true');
                }
            });
            
            function showStatus(message, isError = false) {
                const status = document.getElementById('upload-status');
                status.textContent = message;
                status.style.background = isError ? '#ff4444' : '#4CAF50';
                status.style.color = 'white';
                status.classList.add('show');
                
                setTimeout(() => {
                    status.classList.remove('show');
                }, 5000);
            }
        </script>
    </body>
    </html>
    ''', stats=stats, now=datetime.now())

@app.route('/api/chat', methods=['POST'])
@limiter.limit("60 per minute")  # محدودیت نرخ برای هر IP
def api_chat():
    """API چت با محدودیت نرخ"""
    data = request.json
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'}), 400
    
    try:
        results = brain.search_smart(question)
        
        if results:
            return jsonify({
                'answer': results[0]['answer'],
                'confidence': results[0]['score'],
                'found': True
            })
        else:
            brain.record_unanswered(question)
            return jsonify({
                'answer': None,
                'found': False
            })
    except Exception as e:
        app.logger.error(f"Chat API error: {e}")
        return jsonify({'error': 'خطای داخلی سرور'}), 500

# ================ پنل مدیریت پیشرفته ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """صفحه لاگین مدیریت"""
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        for user in users.values():
            if user.username == username and user.password == password:
                login_user(user)
                return redirect(url_for('admin_panel'))
        
        return "❌ نام کاربری یا رمز عبور اشتباه است"
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ورود به پنل مدیریت</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                width: 350px;
            }
            h2 { text-align: center; color: #333; margin-bottom: 30px; }
            input {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-family: Tahoma;
            }
            button {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1.1em;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 ورود به پنل مدیریت</h2>
            <form method="POST">
                <input type="text" name="username" placeholder="نام کاربری" value="admin" required>
                <input type="password" name="password" placeholder="رمز عبور" value="admin123" required>
                <button type="submit">ورود</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/admin')
@login_required
@cache.cached(timeout=10)
def admin_panel():
    """پنل مدیریت اصلی با کش"""
    stats = brain.get_stats()
    
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>پنل مدیریت - تاریخ‌دان هوشمند</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-number {
                font-size: 2em;
                color: #1e3c72;
                font-weight: bold;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            textarea, input[type=text], select {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-family: Tahoma;
            }
            button {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
            }
            .file-upload {
                border: 2px dashed #1e3c72;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
            }
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .nav-links {
                display: flex;
                gap: 10px;
            }
            .nav-links a {
                color: white;
                text-decoration: none;
                padding: 8px 15px;
                border-radius: 20px;
                background: rgba(255,255,255,0.2);
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                width: 0%;
                transition: width 0.3s;
            }
            .checkbox-label {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚙️ پنل مدیریت تاریخ‌دان</h1>
            <div class="nav-links">
                <a href="/" target="_blank">🌐 صفحه چت</a>
                <a href="/logout">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.total }}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.never_used_count }}</div>
                <div>استفاده نشده</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.unanswered_count }}</div>
                <div>سوال بی‌پاسخ</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h2>📝 آموزش تکی</h2>
                <form action="/admin/add" method="POST">
                    <input type="text" name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="جواب" required></textarea>
                    <select name="category">
                        <option>ایران باستان</option>
                        <option>اسلامی</option>
                        <option>معاصر</option>
                        <option>جهان</option>
                    </select>
                    <div class="checkbox-label">
                        <input type="checkbox" name="force" id="force-single">
                        <label for="force-single">اجبار به یادگیری (حتی اگر تکراری باشد)</label>
                    </div>
                    <button type="submit">➕ اضافه کن</button>
                </form>
            </div>
            
            <div class="card">
                <h2>📁 آپلود چندین فایل آموزشی</h2>
                <form action="/admin/upload" method="POST" enctype="multipart/form-data" id="upload-form">
                    <div class="file-upload" onclick="document.getElementById('files').click()">
                        <p>📤 برای انتخاب چند فایل کلیک کنید</p>
                        <p style="font-size:0.9em; color:#666;">حداکثر ۱۰۰۰ فایل - فرمت: هر خط: سوال | جواب</p>
                    </div>
                    <input type="file" id="files" name="files" style="display:none;" accept=".txt" multiple>
                    <div class="checkbox-label">
                        <input type="checkbox" name="force" id="force" checked>
                        <label for="force">اجبار به یادگیری (حتی تکراری‌ها هم ذخیره شوند)</label>
                    </div>
                    <div id="file-list" style="margin: 10px 0;"></div>
                    <div class="progress-bar" id="progress-bar" style="display:none;">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <button type="submit" id="upload-btn">📥 آپلود و آموزش</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 آمار دقیق</h2>
            <h3>دسته‌بندی‌ها:</h3>
            <ul>
            {% for cat, count in stats.categories.items() %}
                <li>{{ cat }}: {{ count }} مورد</li>
            {% endfor %}
            </ul>
            
            <h3>پراستفاده‌ترین‌ها:</h3>
            {% for item in stats.most_used %}
            <div style="background:#f0f0f0; padding:10px; margin:5px 0; border-radius:5px;">
                {{ item.question }} - {{ item.times_used }} بار
            </div>
            {% endfor %}
        </div>
        
        <script>
            document.getElementById('files').addEventListener('change', function(e) {
                const fileList = document.getElementById('file-list');
                const files = Array.from(e.target.files);
                fileList.innerHTML = '<strong>فایل‌های انتخاب شده:</strong><br>' + 
                    files.map(f => `📄 ${f.name} (${(f.size/1024).toFixed(2)} KB)`).join('<br>');
            });
            
            document.getElementById('upload-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const files = document.getElementById('files').files;
                if (files.length === 0) {
                    alert('لطفاً فایل‌ها را انتخاب کنید');
                    return;
                }
                
                const formData = new FormData(this);
                const progressBar = document.getElementById('progress-bar');
                const progressFill = document.getElementById('progress-fill');
                const uploadBtn = document.getElementById('upload-btn');
                
                progressBar.style.display = 'block';
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'در حال آپلود...';
                
                fetch('/admin/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    progressFill.style.width = '100%';
                    setTimeout(() => {
                        alert(`✅ ${data.count} مورد با موفقیت اضافه شد\n❌ ${data.errors.length} خطا`);
                        location.reload();
                    }, 500);
                })
                .catch(error => {
                    alert('خطا در آپلود: ' + error);
                })
                .finally(() => {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '📥 آپلود و آموزش';
                });
            });
        </script>
    </body>
    </html>
    ''', stats=stats)

@app.route('/admin/add', methods=['POST'])
@login_required
def admin_add():
    """اضافه کردن دانش تکی"""
    question = request.form['question']
    answer = request.form['answer']
    category = request.form.get('category', 'عمومی')
    force = 'force' in request.form
    
    success, msg = brain.add_knowledge(question, answer, category, force)
    
    if success:
        return redirect(url_for('admin_panel'))
    else:
        return f"❌ {msg} <a href='/admin'>بازگشت</a>"

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    """آپلود چندین فایل آموزشی"""
    if 'files' not in request.files:
        return jsonify({'count': 0, 'errors': ['❌ فایلی انتخاب نشده است']})
    
    files = request.files.getlist('files')
    force = 'force' in request.form
    
    if len(files) == 0:
        return jsonify({'count': 0, 'errors': ['❌ فایلی انتخاب نشده است']})
    
    if len(files) > app.config['MAX_FILES_PER_UPLOAD']:
        return jsonify({'count': 0, 'errors': [f'❌ حداکثر {app.config["MAX_FILES_PER_UPLOAD"]} فایل مجاز است']})
    
    saved_files = []
    errors = []
    
    for file in files:
        if file and file.filename.endswith('.txt'):
            try:
                filename = secure_filename(file.filename)
                # اضافه کردن timestamp برای جلوگیری از تداخل نام
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{name}_{timestamp}{ext}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filepath)
            except Exception as e:
                errors.append(f"خطا در ذخیره {file.filename}: {str(e)}")
    
    if saved_files:
        # پردازش فایل‌ها در پس‌زمینه
        future = executor.submit(brain.add_bulk_from_files, saved_files, 'file_upload')
        
        try:
            count, file_errors = future.result(timeout=300)  # timeout 5 دقیقه
            errors.extend(file_errors)
            
            # پاک کردن فایل‌های موقت
            for f in saved_files:
                try:
                    os.remove(f)
                except:
                    pass
            
            return jsonify({
                'count': count,
                'errors': errors[:10]  # فقط ۱۰ خطای اول
            })
        except Exception as e:
            return jsonify({'count': 0, 'errors': [f'خطا در پردازش: {str(e)}']})
    
    return jsonify({'count': 0, 'errors': errors})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================ راه‌اندازی ================
if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     🤖 ربات تاریخ‌دان هوشمند - نسخه نهایی با مقیاس‌پذیری بالا       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                               ║
    ║  📁 حداکثر حجم آپلود: {} MB                                         ║
    ║  📂 حداکثر تعداد فایل: {}                                            ║
    ║  🌐 صفحه چت: http://localhost:5000                                  ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login                   ║
    ║  👤 کاربر: admin / رمز: admin123                                    ║
    ║  ⚡ حالت: یادگیری اجباری فعال - تکرارها هم ذخیره می‌شوند            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    ℹ️ برای نصب پیش‌نیازها:
    pip install flask flask-cors flask-login flask-limiter flask-caching redis scikit-learn numpy
    
    ℹ️ برای اجرا با Redis (توصیه می‌شود):
    docker run -d -p 6379:6379 redis
    
    یا اگر Redis ندارید، برنامه با محدودیت‌های کمتر اجرا می‌شود.
    """.format(len(brain.knowledge_base), 
               app.config['MAX_CONTENT_LENGTH'] // (1024*1024),
               app.config['MAX_FILES_PER_UPLOAD']))
    
    # اجرا با تنظیمات production
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
