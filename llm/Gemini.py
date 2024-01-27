import os
import google.generativeai as genai


def configure_api(api_key, proxy_url=None):
    os.environ['https_proxy'] = proxy_url if proxy_url else None
    os.environ['http_proxy'] = proxy_url if proxy_url else None
    genai.configure(api_key=api_key)
    
    
class Gemini:
    def __init__(self, model_path='gemini-pro', api_key=None, proxy=None):
        configure_api(api_key, proxy)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        self.model = genai.GenerativeModel(model_path, safety_settings=safety_settings)
        
    def chat(self, message):
        times = 0
        while True:
            try:
                response = self.model.generate_content(message)
                return response.text
            except:
                times += 1
                if times > 5:
                    raise Exception('Failed to generate text.')
        