from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

nltk.download('punkt')

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================
with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)

sentences = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# =========================
# NAIVE BAYES TRAINING
# =========================
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(sentences)
y = labels

model = MultinomialNB()
model.fit(X, y)

all_words = set(vectorizer.get_feature_names_out())

# =========================
# GROQ CONFIG
# =========================
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# UTIL FUNCTIONS
# =========================
def predict_class(text):
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    best_idx = np.argmax(probs)
    tag = classes[best_idx]
    confidence = probs[best_idx]

    tokens = nltk.word_tokenize(text.lower())

    prob_map = dict(zip(classes, probs.round(4)))

    return tag, confidence, tokens, prob_map


def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Maaf, saya belum mengerti."


def is_related_to_json(tokens):
    return any(token in all_words for token in tokens)


def ask_groq(question):
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah asisten umum yang menjawab pertanyaan di luar konteks layanan feri."
            },
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    return completion.choices[0].message.content


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
def chatbot_response():
    msg = request.args.get("msg")

    tag, confidence, tokens, probabilities = predict_class(msg)
    related = is_related_to_json(tokens)

    use_groq = False

    if confidence < 0.6 or not related:
        reply = ask_groq(msg)
        use_groq = True
    else:
        reply = get_response(tag)

    return jsonify({
        "reply": reply,
        "analysis": {
            "tag": tag,
            "confidence": round(float(confidence), 3),
            "tokens": tokens,
            "probabilities": probabilities,
            "fallback_to_groq": use_groq
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
