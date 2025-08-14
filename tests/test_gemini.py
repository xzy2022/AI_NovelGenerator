import os

# 临时设置环境变量（仅在当前 Python 进程有效）
os.environ["GEMINI_API_KEY"] = "AIzaSyBh3bTTGln7enmSzlrKC_5RGuBSog-c-Og"  # 替换为实际 Key


from google import genai
# import google.generativeai as genai  

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="用一句话解释什么是AI"
)
print(response.text)