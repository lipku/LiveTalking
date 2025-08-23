from openai import OpenAI
import asyncio
from typing import AsyncGenerator, Optional

ori_client = OpenAI(
    api_key="sk-SoswpWotthi046yQ05723a7eE4F34fA6B0819824F994219d",
    base_url="https://one-api.modelbest.co/v1"
)

def get_answer_from_query(query, system_prompt="You are a helpful assistant.", model="gemini-2.5-flash-nothinking", temperature=0.2, client=ori_client):
    """同步获取 LLM 回复"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=temperature,
        max_tokens=4096
    )
    return completion.choices[0].message.content

def get_answer_stream(query, system_prompt="You are a helpful assistant.", model="gemini-2.5-flash-nothinking", temperature=0.2, client=ori_client):
    """流式获取 LLM 回复"""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=temperature,
        max_tokens=4096,
        stream=True
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content
    
    return full_response

async def get_answer_async(query, system_prompt="You are a helpful assistant.", model="gemini-2.5-flash-nothinking", temperature=0.2, client=ori_client):
    """异步获取 LLM 回复"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        get_answer_from_query,
        query,
        system_prompt,
        model,
        temperature,
        client
    )

if __name__ == "__main__":
    for i in range(1):
        query = f"你好"
        print(get_answer_from_query(query, model="gemini-2.5-flash-nothinking"))