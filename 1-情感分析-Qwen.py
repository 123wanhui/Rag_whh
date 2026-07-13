import dashscope
import os
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY') 

# 封装模型响应函数
def get_response(messages):
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message'
    )
    return response
    
review = '这款音效特别好 给你意想不到的音质。'
messages=[
    {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向"},
    {"role": "user", "content": review}
  ]

response = get_response(messages)

if response.status_code == 200 and response.output is not None:
    if hasattr(response.output, 'choices') and response.output.choices:
        print(response.output.choices[0].message.content)
    elif hasattr(response.output, 'text') and response.output.text:
        print(response.output.text)
    else:
        print("未找到响应内容")
else:
    print(f"API调用失败，状态码: {response.status_code}")
    print(f"错误信息: {response.message}")