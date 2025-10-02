from langchain_ollama import ChatOllama
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

grok = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_GPT_OSS_KEY"),
    model="x-ai/grok-4-fast:free",
)

gpt_oss = ChatOllama(model="gpt-oss:120b-cloud")

deepseek = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_DEEPSEEK_KEY"),
    model="deepseek/deepseek-chat-v3.1:free",
)

groq = ChatGroq(model_name="llama-3.1-8b-instant")

gigachat_pro = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    model="GigaChat-Pro",
)

gigachat = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    model="GigaChat-2",
)
