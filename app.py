import os
import json
import random
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# --- KONFIGURASI NLTK KHUSUS VERCEL ---
# Mengarahkan download ke folder /tmp karena Vercel tidak mengizinkan tulis di root
nltk_data_dir = '/tmp/nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download hanya komponen minimal
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)

# --- LOAD DATA ---
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

def calculate_similarity(user_tokens, pattern_tokens):
    """Menghitung skor kecocokan antara input user dan database"""
    words_match = set(user_tokens).intersection(set(pattern_tokens))
    if not pattern_tokens: return 0
    return len(words_match) / len(set(user_tokens).union(set(pattern_tokens)))

def get_bot_response(user_message):
    user_message = user_message.lower().strip()
    user_tokens = word_tokenize(user_message)
    
    # Priority 1: Menu Angka
    menu_map = {"1": "schedule", "2": "pricing", "3": "facilities", "4": "cancellation"}
    if user_message in menu_map:
        tag = menu_map[user_message]
        confidence = 1.0
    else:
        # Priority 2: Similarity Matching (Logika NLP)
        best_tag = None
        max_sim = 0
        
        for intent in intents_data['intents']:
            for pattern in intent['patterns']:
                pattern_tokens = word_tokenize(pattern.lower())
                sim = calculate_similarity(user_tokens, pattern_tokens)
                if sim > max_sim:
                    max_sim = sim
                    best_tag = intent['tag']
        
        tag = best_tag
        confidence = max_sim

    # Respon jika tidak ditemukan
    if not tag or confidence < 0.15:
        return {
            "reply": "Maaf, saya kurang mengerti. Coba tanyakan jadwal, harga, atau pilih angka 1-4.",
            "analysis": {"tag": "Unknown", "confidence": "0%", "tokens": user_tokens}
        }

    # Ambil Jawaban
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            res = random.choice(intent['responses'])
            if tag == "greetings":
                res += "<br><br>1. Jadwal<br>2. Harga<br>3. Fasilitas"
            
            return {
                "reply": res,
                "analysis": {
                    "tag": tag, 
                    "confidence": f"{round(confidence * 100, 2)}%", 
                    "tokens": user_tokens,
                    "method": "NLP Similarity Match"
                }
            }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response_route():
    msg = request.args.get('msg')
    # Mengembalikan data JSON lengkap untuk UI Dashboard
    return jsonify(get_bot_response(msg))

if __name__ == "__main__":
    app.run(debug=True)