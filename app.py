from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Flask app
app = Flask(__name__)

# functions generate response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# path for home
@app.route("/")
def home():
    return render_template("index.html")  # use the index.html template

# path for chat
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if user_input:
        response = generate_response(user_input)
        return jsonify({"response": response})
    return jsonify({"error": "No message provided!"}), 400

# Start server
if __name__ == "__main__":
    app.run(debug=True)
