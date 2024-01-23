from src.Linly import Linly
from src.Qwen import Qwen
from src.Gemini import Gemini

def test_Linly(question = "如何应对压力？", mode='offline', model_path="Linly-AI/Chinese-LLaMA-2-7B-hf"):
    llm = Linly(mode, model_path)
    answer = llm.generate(question)
    print(answer)
    
def test_Qwen(question = "如何应对压力？", mode='offline', model_path="Qwen/Qwen-1_8B-Chat"):
    llm = Qwen(mode, model_path)
    answer = llm.generate(question)
    print(answer)
    
def test_Gemini(question = "如何应对压力？", model_path='gemini-pro', api_key=None, proxy_url=None):
    llm = Gemini(model_path, api_key, proxy_url)
    answer = llm.generate(question)
    print(answer)
    
    
if __name__ == '__main__':
    test_Linly()
    # test_Qwen()
    # test_Gemini()
