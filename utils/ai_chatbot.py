import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_ai_response(user_input: str) -> str:
    """
    사용자의 질문(user_input)에 대해 OpenAI GPT 계열 모델로부터 답변을 받아 반환합니다.
    실제 사용 시 모델 엔진이나 파라미터를 조정하세요.
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_input,
            max_tokens=100,
            temperature=0.7
        )
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return f"에러가 발생했습니다: {e}"
