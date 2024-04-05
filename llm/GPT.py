import openai


class GPT():
    def __init__(self, model_path = 'gpt-3.5-turbo', api_key = None, base_url = None):
        openai.api_key = api_key
        self.model_path = model_path
        if base_url != None:
            openai.base_url = base_url

    def chat(self, message):
        response = openai.ChatCompletion.create(
            model=self.model_path,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response['choices'][0]['message']['content']

if __name__ == '__main__':
    llm = GPT('gpt-3.5-turbo', '你的API Key','https://openai.api2d.net/v1')
    response = llm.chat("如何应对压力？")