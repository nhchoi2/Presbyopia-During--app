import openai

API_KEY = "your-api-key"

def ask_ai(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        api_key=API_KEY
    )
    return response["choices"][0]["message"]["content"]
