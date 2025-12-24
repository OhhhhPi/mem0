import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TECHNIQUES = ["mem0", "rag", "langmem", "zep", "openai"]

METHODS = ["add", "search"]


def get_llm_client():
    """获取用于生成回答的 LLM 客户端 (DeepSeek)"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def get_embedding_client():
    """获取用于 embedding 的客户端 (Qwen)"""
    return OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL"),
    )


def get_judge_client():
    """获取用于 LLM judge 的客户端 (Qwen)"""
    return OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL"),
    )
