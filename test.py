import requests

url = "https://dpapi.cn/v1/chat/completions"
headers = {
    "Authorization": "sk-uBhYPi9uqJgsRvVF32F7A4B8960841618b1b248eD6195c46",
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "你好，你是谁"}],
}

response = requests.post(url, headers=headers, json=data)

if response.status_code != 200:
    raise Exception(f"Error: {response.status_code}, {response.json()}")
json_data = response.json()
content = json_data["choices"][0]["message"]["content"]


print("响应结果：", content)