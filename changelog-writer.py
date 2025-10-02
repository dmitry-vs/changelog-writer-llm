from pprint import pprint
from dotenv import load_dotenv
from github import Github, Auth
import os
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from models import grok, gpt_oss, groq, deepseek, gigachat_pro, gigachat
from langgraph.graph import StateGraph, START

load_dotenv()

# Define a tool to get commits and tags from GitHub
@tool
def get_commits_and_tags() -> list:
    """
    Получить список коммитов из основной ветки с информацией о тегах.
        
    Возвращает:
        Список словарей, содержащих SHA коммита, сообщение, временную метку коммита и тег (если есть)
    """
    
    try:
        auth = Auth.Token(os.getenv("GITHUB_API_TOKEN"))
        g = Github(auth=auth)
        repo = g.get_repo(os.getenv("GITHUB_REPO"))
        commits = repo.get_commits(sha="main")
        tags = repo.get_tags()
        tag_mapping = {tag.commit.sha: tag.name for tag in tags}
        commit_list = []
        for commit in commits:
            commit_data = {
                'sha': commit.sha,
                'message': commit.commit.message,
                'date': commit.commit.committer.date,
                'tag': tag_mapping.get(commit.sha, None)
            }
            commit_list.append(commit_data)
        return commit_list
    except Exception as e:
        return f"Ошибка при получении коммитов: {str(e)}"

# create state class for graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    changelog: str
    needs_human_review: bool

# Setup LLM
llm = gpt_oss

# Bind tools to LLM
tools = [get_commits_and_tags]
llm_with_tools = llm.bind_tools(tools)

# Define chatbot node
def chatbot(state: State):
    """Chatbot node that processes messages and decides whether to use tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Define human review node
def human_review(state: State):
    """Human review node for changelog correction."""
    print("\n" + "="*80)
    print("🤖 СГЕНЕРИРОВАННЫЙ CHANGELOG:")
    print("="*80)
    print(state["changelog"])
    print("="*80)
    
    print("\n📝 ВАРИАНТЫ ДЕЙСТВИЙ:")
    print("1. Принять changelog как есть (введите 'accept')")
    print("2. Исправить changelog (введите 'edit' и затем что нужно исправить)")
    print("3. Попросить перегенерировать (введите 'regenerate')")
    print("4. Выйти (введите 'quit')")
    
    while True:
        user_input = input("\nВаш выбор: ").strip().lower()
        
        if user_input == 'accept':
            print("✅ Changelog принят!")
            print("👋 Выход из программы.")
            exit(0)
        elif user_input == 'edit':
            print("\n📝 Введите требуемые исправления для changelog (завершите ввод пустой строкой):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            changelog_corrections = "\n".join(lines)
            if changelog_corrections.strip():
                print("💼 Changelog отправлен на доработку")
                return {
                    "messages": [{"role": "user", "content": f"Пользователь запросил доработку changelog:\n{changelog_corrections}"}],
                    "changelog": changelog_corrections,
                    "needs_human_review": False
                }
            else:
                print("❌ Пустой changelog. Попробуйте снова.")
        elif user_input == 'regenerate':
            print("🔄 Запрашиваем перегенерацию changelog...")
            return {
                "messages": [{"role": "user", "content": "Пользователь просит перегенерировать changelog. Пожалуйста, создай новый changelog на основе тех же коммитов, но с улучшениями."}],
                "needs_human_review": True
            }
        elif user_input == 'quit':
            print("👋 Выход из программы.")
            exit(0)
        else:
            print("❌ Неверный ввод. Попробуйте снова.")

# Define changelog extraction node
def extract_changelog(state: State):
    """Extract changelog from LLM response and prepare for human review."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'content') and last_message.content:
        changelog = last_message.content
        return {
            "changelog": changelog,
            "needs_human_review": True
        }
    return {"needs_human_review": False}

# Define conditional function for routing
def should_review(state: State):
    """Determine if changelog needs human review."""
    if state.get("needs_human_review", False):
        return "human_review"
    else:
        return "chatbot"

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("extract_changelog", extract_changelog)
graph_builder.add_node("human_review", human_review)

# Add conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Add edges
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "extract_changelog")
graph_builder.add_conditional_edges(
    "extract_changelog",
    should_review,
)
graph_builder.add_edge("human_review", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Prepare prompts and initial state
system_prompt = """
Ты эксперт по работе с GitHub.
Ты умеешь выполнять экспертный анализ коммитов и тегов.
Ты умеешь генерировать текст changelog на основе коммитов и тегов на русском языке.
"""

user_prompt = """
Используй доступный инструмент для получения списка коммитов с тегами из GitHub.

Теги представляют собой номера версий проекта в формате semver.
Они состоят из трех частей: major, minor, patch.
Примеры:
1.0.0
1.0.1
2.5.19

Тебе нужно найти в списке коммитов последний по времени коммит, у которого есть тег (назовем его "last").

Затем тебе нужно найти в списке коммитов коммит, у которого есть тег, minor версия такая же как у "last", и patch версия равна 0 (назовем его "first").
Если "first" не найден, то в качестве "first" возьми самый первый коммит в репозитории.

Составь список коммитов между "last" и "first", назовем его "commits".
При этом включи в список коммиты "first" и "last".
Порядок следования коммитов должен быть хронологическим по убыванию (сначала самые новые коммиты).
Другие коммиты не нужно включать, их вообще больше не нужно использовать.

Сгенерируй changelog на основе "commits".
Текст должен быть на русском языке.
Текст должен быть написан в формате markdown.
Текст должен содержать в качестве подзаголовка номер версии, которая соответствует "x".
Текст должен содержать следующие разделы в виде списков: 
- добавлено (если есть элементы, описание того, что было добавлено)
- исправлено (если есть элементы, описание того, что было исправлено)
- другое (если есть элементы, описание того, что было сделано, но не добавлено и не исправлено)
Все элементы должны быть написаны только на русском языке (где это возможно).
Порядок элементов в списках должен соответствовать порядку коммитов в списке "commits".

Пример:
[начало примера]
## 2.15.47

### ✅ Добавлено
- добавить сайдбар
- создать header
- добавить footer

### 🔧 Исправлено
- поправить стили таблицы
- убрать лишние обращения к сторонним сервисам

### 🔄 Другое
- обновить версии пакетов
- повысить покрытие тестами
[конец примера]

Выведи полученный changelog в виде строки.
"""

initial_state = {
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "changelog": "",
    "needs_human_review": False
}

# Run the agent
print("🚀 Запуск генерации changelog...")
response = graph.invoke(initial_state, config={"recursion_limit": 50})

# Final output
if response.get("changelog"):
    print("\n" + "="*80)
    print("📋 ФИНАЛЬНЫЙ CHANGELOG:")
    print("="*80)
    print(response["changelog"])
    print("="*80)
else:
    print("❌ Changelog не был сгенерирован.")