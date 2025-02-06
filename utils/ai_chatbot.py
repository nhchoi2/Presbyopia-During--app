import openai

API_KEY = "your-api-key"

def ask_ai(question):
    client = openai.OpenAI(api_key=API_KEY)  # 새로운 API 클라이언트 사용
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
