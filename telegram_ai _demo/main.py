import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from datetime import datetime
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import json
from pydantic import SecretStr

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key is None:
    raise ValueError("TAVILY_API_KEY environment variable is not set.")
tavily_tool = TavilySearchAPIWrapper(tavily_api_key=SecretStr(tavily_api_key))

def tavily_search(query: str) -> str:
    return json.dumps(tavily_tool.results(query), indent=2)

# --- Load Environment ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in your environment or .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in your environment or .env file.")

# --- LangChain Setup ---
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=openai_model)
# Remove DuckDuckGoSearchRun and use TavilySearchTool
# search_tool = DuckDuckGoSearchRun()

def get_current_date_time(query: str) -> str:
    """Return the current date and/or time if the query is about today's date or current time."""
    now = datetime.now()
    query_lower = query.lower()
    if "time" in query_lower and "date" in query_lower:
        return now.strftime("Today's date is %A, %B %d, %Y and the current time is %I:%M %p.")
    elif "time" in query_lower:
        return now.strftime("The current time is %I:%M %p.")
    else:
        return now.strftime("Today's date is %A, %B %d, %Y.")

def llm_only_search(query: str) -> str:
    llm_response = llm.invoke(query)
    content = getattr(llm_response, 'content', llm_response)
    if isinstance(content, list):
        content = '\n'.join(str(item) for item in content)
    elif not isinstance(content, str):
        content = str(content)
    return content

tools = [
    Tool(
        name="Web Search",
        func=tavily_search,
        description="Use this tool to find and return direct links (URLs) to web pages, including scorecards, YouTube videos, and the latest news or updates using Tavily. Always include the most relevant and recent links in your answer. Use this for current events, live scores, and breaking news."
    ),
    Tool(
        name="LLM",
        func=llm_only_search,
        description="Use this tool to answer user queries using the LLM model only. Do not perform any web search."
    ),
    Tool(
        name="Current Date and Time",
        func=get_current_date_time,
        description="Use this tool ONLY if the user explicitly asks for today's date, the current date, the current time, or both. Do NOT use for greetings or unrelated questions."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Telegram Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("👋 Hello! I'm your AI Assistant. Ask me anything and I'll search the web for you!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        user_input = update.message.text.strip().lower()
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greet in user_input for greet in greetings):
            await update.message.reply_text("👋 Hello! How can I assist you today?")
            return
        try:
            response = agent.run(update.message.text)
        except Exception as e:
            response = f"❌ Sorry, I ran into an error: {str(e)}"
        await update.message.reply_text(response)

# --- Main Application ---
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot is running...")
    app.run_polling()
