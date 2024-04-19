import os
import openai

'''
`huggingface`连接不上可以使用 `modelscope`
`pip install modelscope`
'''
from modelscope import AutoModelForCausalLM, AutoTokenizer
#from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Qwen:
    def __init__(self, model_path="Qwen/Qwen-1_8B-Chat", api_base=None, api_key=None) -> None:
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        # 默认本地推理
        self.local = True

        # api_base和api_key不为空时使用openapi的方式
        if api_key is not None and api_base is not None:
            openai.api_base = api_base
            openai.api_key = api_key
            self.local = False
            return

        self.model, self.tokenizer = self.init_model(model_path)
        self.data = {}

    def init_model(self, path="Qwen/Qwen-1_8B-Chat"):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat",
                                                     device_map="auto",
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer

    def chat(self, question):
        # 优先调用qwen openapi的方式
        if not self.local:
            # 不使用流式回复的请求
            response = openai.ChatCompletion.create(
                model="Qwen",
                messages=[
                    {"role": "user", "content": question}
                ],
                stream=False,
                stop=[]
            )
            return response.choices[0].message.content

        # 默认本地推理
        self.data["question"] = f"{question} ### Instruction:{question}  ### Response:"
        try:
            response, history = self.model.chat(self.tokenizer, self.data["question"], history=None)
            print(history)
            return response
        except:
            return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"


def test():
    llm = Qwen(model_path="Qwen/Qwen-1_8B-Chat")
    answer = llm.chat(question="如何应对压力？")
    print(answer)


if __name__ == '__main__':
    test()
