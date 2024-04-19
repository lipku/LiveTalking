from llm.Qwen import Qwen
from llm.Gemini import Gemini
from llm.ChatGPT import ChatGPT
from llm.VllmGPT import VllmGPT

def test_Qwen(question = "如何应对压力？", mode='offline', model_path="Qwen/Qwen-1_8B-Chat"):
    llm = Qwen(mode, model_path)
    answer = llm.generate(question)
    print(answer)

def test_Gemini(question = "如何应对压力？", model_path='gemini-pro', api_key=None, proxy_url=None):
    llm = Gemini(model_path, api_key, proxy_url)
    answer = llm.generate(question)
    print(answer)

class LLM:
    def __init__(self, mode='offline'):
        self.mode = mode

    def init_model(self, model_name, model_path, api_key=None, proxy_url=None):
        if model_name not in ['Qwen', 'Gemini', 'ChatGPT', 'VllmGPT']:
            raise ValueError("model_name must be 'ChatGPT', 'VllmGPT', 'Qwen', or 'Gemini'(其他模型还未集成)")

        if model_name == 'Gemini':
            llm = Gemini(model_path, api_key, proxy_url)
        elif model_name == 'ChatGPT':
            llm = ChatGPT(model_path, api_key=api_key)
        elif model_name == 'Qwen':
            llm = Qwen(model_path=model_path, api_key=api_key, api_base=proxy_url)
        elif model_name == 'VllmGPT':
            llm = VllmGPT()
        return llm


    def test_Qwen(self, question="如何应对压力？", model_path="Qwen/Qwen-1_8B-Chat", api_key=None, proxy_url=None):
        llm = Qwen(model_path=model_path, api_key=api_key, api_base=proxy_url)
        answer = llm.chat(question)
        print(answer)

    def test_Gemini(self, question="如何应对压力？", model_path='gemini-pro', api_key=None, proxy_url=None):
        llm = Gemini(model_path, api_key, proxy_url)
        answer = llm.chat(question)
        print(answer)

if __name__ == '__main__':
    llm = LLM()
    # llm.test_Gemini(api_key='你的API Key', proxy_url=None)
    # llm = LLM().init_model('Gemini', model_path= 'gemini-pro',api_key='AIzaSyBWAWfT8zsyAZcRIXLS5Vzlw8KKCN9qsAg', proxy_url='http://172.31.71.58:7890')
    # response = llm.chat("如何应对压力？")
    # llm = LLM().init_model('VllmGPT', model_path= 'THUDM/chatglm3-6b')
    # response = llm.chat("如何应对压力？")
    # print(response)

    llm.test_Qwen(api_key="none", proxy_url="http://10.1.1.113:18000/v1")
