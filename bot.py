from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Завантаження передтренованої моделі GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Функція для генерації відповіді
def generate_response(prompt):
    # Токенізація введеного тексту
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    
    # Генерація відповіді
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # Декодування відповіді
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Головний цикл чат-бота
print("Привіт! Я твій чат-бот. Як я можу допомогти?")
while True:
    user_input = input("Ти: ")
    if user_input.lower() == 'вихід':
        print("До зустрічі!")
        break
    response = generate_response(user_input)
    print(f"Чат-бот: {response}")
