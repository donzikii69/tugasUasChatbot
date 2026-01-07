from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS
import json
import random

app = Flask(__name__)
# CORS(app)  # Mengizinkan akses dari frontend Vite

# Load data
with open("intents.json", "r") as file:
    intents_data = json.load(file)


def get_bot_response(user_message):
    user_message = user_message.lower().strip()

    # Menu Mapping
    menu_options = {"1": "schedule", "2": "pricing", "3": "services", "4": "contact"}

    # Cek input angka
    if user_message in menu_options:
        target_tag = menu_options[user_message]
        for intent in intents_data["intents"]:
            if intent["tag"] == target_tag:
                return random.choice(intent["responses"])

    # Cek pattern keyword
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_message:
                res = random.choice(intent["responses"])
                if intent["tag"] == "greetings":
                    res += "\n\n1. Schedule\n2. Pricing\n3. Services\n4. Contact"
                return res

    return "I'm sorry, I don't understand. Type '1' for Schedule or 'Hello' for menu."


# API Endpoint untuk Frontend
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message")
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    bot_res = get_bot_response(user_msg)
    return jsonify({"reply": bot_res})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
# print(f"Looking for templates in: {os.path.join(app.root_path, 'templates')}")