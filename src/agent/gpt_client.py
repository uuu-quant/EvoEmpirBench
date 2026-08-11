import os

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class GPTClient:
    def __init__(self, key=None, url=None, model=None):
        openai_key = os.getenv("OPENAI_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL") or "gpt-4"
        self.APIKEY = key or openai_key or deepseek_key

        if url:
            self.baseURL = url
        elif openai_key:
            self.baseURL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or DEFAULT_OPENAI_BASE_URL
        elif deepseek_key:
            self.baseURL = os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        else:
            self.baseURL = DEFAULT_OPENAI_BASE_URL

        if not self.APIKEY:
            raise ValueError(
                "Missing API key. Set OPENAI_API_KEY/DEEPSEEK_API_KEY in the environment "
                "or pass key=... explicitly."
            )
        self.client = self.init_connection()

    def init_connection(self):
        return OpenAI(
            api_key=self.APIKEY,
            base_url=self.baseURL
        )

    def get_response(self, messages: list, model=None):
        # 如果传入了model参数，则使用它，否则保持当前的self.model
        if model:
            self.model = model
        
        # 确保self.model不为None
        if self.model is None:
            self.model = "gpt-4"  # 设置默认模型
            
        # 支持不同模型系列的标识
        model_name = self.model.lower()  # 转换为小写
        is_qwen = 'qwen' in model_name
        is_gpt4_1 = 'gpt-4.1' in model_name or 'gpt-4-1' in model_name
        
        # 根据模型调整参数
        temperature = 0.3  # 默认温度
        
        if is_gpt4_1:
            # 对于GPT-4.1模型的特殊处理
            temperature = 0.2  # 稍微降低温度，更加确定性
            
        try:
            if is_qwen:
                # 对于 Qwen 系列模型使用流式输出
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )
                
                # 收集完整的响应
                collected_content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        collected_content += chunk.choices[0].delta.content
                
                if "该请求" in collected_content or "sorry" in collected_content:
                    return 'Failed'
                return collected_content
            else:
                # 对于其他模型(包括GPT-4和GPT-4.1)使用普通输出
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                if "该请求" in completion.choices[0].message.content or "sorry" in completion.choices[0].message.content:
                    return 'Failed'
                return completion.choices[0].message.content
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return 'Failed'

    def infer_data(self, data):
        return self.get_response(data) 
