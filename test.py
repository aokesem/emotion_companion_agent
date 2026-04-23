from langchain_openai import OpenAIEmbeddings
import os
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="2603860153fd484ead5ff4e05791e05f.J1X7dyCEFjtKuFfS",
    dimensions=1536
)

vector = embeddings.embed_query("我感觉有点累，精神很疲惫")
print(len(vector))  