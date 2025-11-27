import os
from openai import OpenAI

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY")

class LLM:
    def __init__(self,
                 server_name: str = "deepseek",
                 model: str = "deepseek-chat",
                 max_tokens: int = 1024,
                 temperature: float = 0.7,
                 top_p: float = 1.0):
        if server_name == "deepseek":
            self.client = OpenAI(api_key=DEEPSEEK_KEY,
                                 base_url="https://api.deepseek.com")
        elif server_name == "openai":
            self.client = OpenAI(api_key=OPENAI_KEY)
        elif server_name == "gemini":
            self.client = OpenAI(api_key=GEMINI_KEY,
                                 base_url="https://generativelanguage.googleapis.com/v1beta/openai")
        else:
            raise ValueError("server_name must be openai | deepseek | gemini")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def chat(self, user: str, system: str = "You are a helpful CUDA optimization assistant.") -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return resp.choices[0].message.content
    
    def change_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def change_top_p(self, top_p: float) -> None:
        self.top_p = top_p