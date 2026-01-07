from flask import Flask, render_template, request, jsonify # Tambahkan jsonify
import json
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# --- TRAINING MODEL (Sama seperti sebelumnya) ---
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

training_sentences = []
training_labels = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern.lower())
        training_labels.append(intent['tag'])

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(training_sentences, training_labels)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response_route():
    userText = request.args.get('msg').lower().strip()
    
    # NLP Analysis
    tokens = word_tokenize(userText)
    probs = model.predict_proba([userText])
    confidence = np.max(probs)
    tag = model.predict([userText])[0]
    
    # Ambil Respon
    response_text = "Maaf, saya tidak mengerti."
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            response_text = random.choice(intent['responses'])
            if tag == "greetings":
                response_text += "<br><br>1. Jadwal<br>2. Harga<br>3. Fasilitas"

    # Kirim hasil analisis ke UI
    return jsonify({
        "reply": response_text,
        "analysis": {
            "tag": tag,
            "confidence": f"{round(confidence * 100, 2)}%",
            "tokens": tokens,
            "model": "Multinomial Naive Bayes"
        }
    })

if __name__ == "__main__":
    app.run(debug=True)