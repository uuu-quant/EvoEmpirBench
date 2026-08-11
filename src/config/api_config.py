import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量。本仓库不提交真实 .env。
load_dotenv()

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# OpenAI兼容API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
)

# 默认API类型（可选 "deepseek" 或 "openai"）
DEFAULT_API_TYPE = os.getenv("DEFAULT_API_TYPE", "openai")

# 如果默认API是DeepSeek但没有设置API密钥，打印警告信息
if DEFAULT_API_TYPE == "deepseek" and not DEEPSEEK_API_KEY:
    print("警告: 未设置DEEPSEEK_API_KEY环境变量。请在.env文件中设置，或者在运行时提供。")

# 如果默认API是OpenAI但没有设置API密钥，打印警告信息
if DEFAULT_API_TYPE == "openai" and not OPENAI_API_KEY:
    print("警告: 未设置OPENAI_API_KEY环境变量。请在.env文件中设置，或者在运行时提供。") 
