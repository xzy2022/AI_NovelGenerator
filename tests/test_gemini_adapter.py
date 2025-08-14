import os
import logging
from llm_adapters import BaseLLMAdapter
from google import genai
from google.genai import types
from typing import Optional
# from llm_adapters import GeminiAdapter


class GeminiAdapter(BaseLLMAdapter):
    """
    适配 Google Gemini (Google Generative AI) 接口
    注意事项:
    1. model_name 建议使用 'gemini-1.0-pro' 或 'gemini-2.0-flash'
    2. base_url 默认为 'https://generativelanguage.googleapis.com/v1beta'
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout * 1000  # 转换为毫秒

        # 初始化 Gemini 客户端
        self.client = genai.Client(api_key=self.api_key)


    def invoke(self, prompt: str) -> str:
        try:
            # 使用新的 generate_content 方法
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts":[{"text": prompt}]}],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,

                ),
            )
            
            if response and response.candidates and len(response.candidates) > 0:
                # 获取第一个候选结果的文本内容
                content = response.candidates[0].content
                if content and content.parts:
                    return content.parts[0].text
            
            logging.warning("No text response from Gemini API.")
            return ""
            
        except Exception as e:
            logging.error(f"Gemini API 调用失败: {e}")
            return ""

# 配置日志
logging.basicConfig(level=logging.INFO)

def test_gemini_adapter():
    try:
        # 配置参数
        api_key = "AIzaSyBh3bTTGln7enmSzlrKC_5RGuBSog-c-Og"  # 替换为实际的 API key
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        model_name = "gemini-2.0-flash"  # 或使用 "gemini-2.0-flash"
        max_tokens = 1000
        temperature = 0.7
        
        # 初始化适配器
        adapter = GeminiAdapter(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # 测试提示词
        test_prompt = "请用一句话解释什么是人工智能。"
        
        # 调用适配器
        logging.info("正在调用 Gemini API...")
        response = adapter.invoke(test_prompt)
        
        # 输出结果
        logging.info("Gemini 响应:")
        logging.info(response)
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    test_gemini_adapter()
