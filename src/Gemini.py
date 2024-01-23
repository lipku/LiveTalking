import os
import google.generativeai as genai


def configure_api(api_key, proxy_url=None):
    os.environ['https_proxy'] = proxy_url if proxy_url else None
    os.environ['http_proxy'] = proxy_url if proxy_url else None
    genai.configure(api_key=api_key)
    
    
class Gemini:
    def __init__(self, model_path='gemini-pro', api_key=None, proxy_url=None):
        configure_api(api_key, proxy_url)
        self.model = genai.GenerativeModel(model_path)
        
    def generate(self, question):
        response = self.model.generate_content(question)
        return response
        