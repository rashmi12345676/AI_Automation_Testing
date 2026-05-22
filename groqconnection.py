from deepeval.evaluate import evaluate
from dotenv import load_dotenv
from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM
import os
import json

from posthog import api_key
# Improved Groq wrapper to handle JSON requirements
class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3-70b-8192"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name
  

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        use_json = "llama" in self.model_name.lower()
        chat_completion = self.client.chat.completions.create(
            messages= [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Respond strictly in JSON format."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
            model=self.model_name,
            # Force JSON mode if using a compatible model
            response_format={"type": "json_object"} if use_json else None
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name