from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Flask app
app = Flask(__name__)

# functions generate response
def generate_response(prompt):
    # tokenize prompt with attention_mask
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt", padding=True)

    # generate response with attention_mask
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        no_repeat_ngram_size=2
    )

    # decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


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
