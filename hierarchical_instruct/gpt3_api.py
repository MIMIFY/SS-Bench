import json
import tqdm
import os
import random
import openai
from openai import OpenAI
from datetime import datetime
import argparse
import time
    

def make_requests(
        engine, prompts, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, base_url=None
    ):
    client = OpenAI(
        api_key=api_key, 
        base_url= base_url
    )
    
    response = None
    target_length = max_tokens
    # if api_key is not None:
    #     OpenAI.api_key = api_key
    # if organization is not None:
    #     # openai.organization = organization
    #     OpenAI.base_url = organization
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries: # 重新尝试的次数
        #使用 while 循环来处理可能的 API 调用错误。retry_cnt 跟踪重试次数，backoff_time 控制重试之间的等待时间
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompts}
                    ],
                # prompt=prompts,
                # max_tokens=2048,
                # top_p=0.95,
                # frequency_penalty=0,
                # presence_penalty=0,
                # stop=None

                max_tokens=target_length,
                temperature=temperature, #控制生成文本的随机性。较低的值（如 0）会导致更确定性的输出，而较高的值（如 1 或更高）会增加随机性和创造性
                top_p=top_p, # top_p 是一个概率值，用于控制生成过程中的随机性。它决定了在每个步骤中，模型将考虑多少概率最高的 token
                frequency_penalty=frequency_penalty, #控制重复 token 的惩罚。较高的值会降低模型重复使用相同 token 的概率。
                presence_penalty=presence_penalty, #控制特定 token 出现频率的惩罚。较高的值会降低模型生成特定 token 的概率。
                stop=stop_sequences, # 一个列表，包含模型在生成文本时应停止的 token。当模型生成这些 token 时，它会停止生成更多的 token
                logprobs=False, #如果设置为 1，模型将返回每个生成 token 的对数概率。这可以用于进一步的分析或后处理。
                n=n, # 如果 prompt 是一个列表，n 指定了每个提示应该生成多少个独立的文本。这允许并行生成多个响应。
                # best_of=best_of, # 如果 n 大于 1，best_of 指定了从生成的 n 个响应中选择最佳响应的数量。模型将返回这些最佳响应。
            )
            print("调用API一次，成功完毕")
            
            break
        except openai.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
    
    
    if response is not None:
        # 转换返回response的格式，转换成可json化的字典格式
        response.choices[0].message = response.choices[0].message.__dict__
        response.choices[0] = response.choices[0].__dict__
        response.usage = response.usage.__dict__
        response = response.__dict__
    data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }

    return [data] # 返回一个结果列表

   
if __name__ == "__main__":
    # 简单测试 make_requests 函数
    test_prompt = "请用一句话介绍自闭症儿童社交故事的意义。"
    result = make_requests(
        engine="gpt-3.5-turbo",  # 或你自己的模型名称
        prompt=test_prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=None,
        logprobs=None,
        n=1,
        best_of=None,
        api_key="Your OpenAI API Key",  # 填写自己的API Key
        base_url="Your OpenAI API Base URL"  # 填写自己的API Base
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))