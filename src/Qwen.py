import os
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Qwen:
    def __init__(self, mode='api', model_path="Qwen/Qwen-1_8B-Chat") -> None:
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        self.url = "http://ip:port" # local server: http://ip:port
        self.headers = {
            "Content-Type": "application/json"
        }
        self.data = {
            "question": "北京有什么好玩的地方？"
        }
        self.prompt = '''请用少于25个字回答以下问题'''
        self.mode = mode
        
        self.model, self.tokenizer = self.init_model(model_path)
    
    def init_model(self, path = "Qwen/Qwen-1_8B-Chat"):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", 
                                                     device_map="auto", 
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer   
    
    def generate(self, question):
        if self.mode != 'api':
            self.data["question"] = f"{self.prompt} ### Instruction:{question}  ### Response:"
            try:
                response, history = self.model.chat(self.tokenizer, self.data["question"], history=None)
                print(history)
                return response
            except:
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict(question)
    def predict(self, question):
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        pass 
    
def test():
    llm = Qwen(mode='offline',model_path="Qwen/Qwen-1_8B-Chat")
    answer = llm.generate("如何应对压力？")
    print(answer)

if __name__ == '__main__':
    test()
