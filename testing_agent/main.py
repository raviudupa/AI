import os
from dotenv import load_dotenv
import telebot

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from datetime import datetime
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import json
from pydantic import SecretStr
import PyPDF2
import io
from crewai import Agent, Task, Crew, Process

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

# Global variable to store uploaded files
user_files = {}

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_txt(file_bytes):
    """Extract text from text file"""
    try:
        return file_bytes.decode('utf-8')
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def analyze_document_with_crewai(text, file_name):
    """Use CrewAI to analyze and summarize the document"""
    try:
        if len(text) > 8000:
            text = text[:8000] + "... [truncated for analysis]"
        
        # Create CrewAI agents
        document_analyzer = Agent(
            role='Document Analyst',
            goal='Extract key information, main topics, and important details from documents',
            backstory='An expert in document analysis with years of experience in processing various types of documents and extracting meaningful insights.',
            llm=llm,
            verbose=True
        )
        
        summary_writer = Agent(
            role='Summary Writer',
            goal='Create comprehensive, well-structured summaries that capture the essence of documents',
            backstory='A professional content writer and editor who specializes in creating clear, concise summaries that highlight the most important information.',
            llm=llm,
            verbose=True
        )
        
        # Create tasks
        analysis_task = Task(
            description=f"""Analyze the following document content and extract:
            1. Main topics and themes
            2. Key points and arguments
            3. Important facts and figures
            4. Structure and organization
            5. Any notable insights or conclusions
            
            Document: {file_name}
            Content: {text}
            
            Provide a detailed analysis that can be used to create a comprehensive summary.""",
            expected_output='A detailed analysis of the document covering main topics, key points, and important insights.',
            agent=document_analyzer
        )
        
        summary_task = Task(
            description="""Using the analysis provided, create a comprehensive summary that includes:
            1. Executive summary (2-3 sentences)
            2. Main topics covered
            3. Key points and insights
            4. Important conclusions or recommendations
            5. Overall document structure
            
            Make the summary clear, well-organized, and easy to understand.""",
            expected_output='A comprehensive, well-structured summary of the document.',
            agent=summary_writer
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[document_analyzer, summary_writer],
            tasks=[analysis_task, summary_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return f"📄 **CrewAI Analysis of {file_name}**\n\n{result}"
        
    except Exception as e:
        return f"Error in CrewAI analysis: {str(e)}"

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

def summarize_uploaded_file(user_id: int) -> str:
    """Summarize the user's uploaded file using CrewAI"""
    if user_id not in user_files:
        return "No file uploaded. Please upload a file first."
    
    file_info = user_files[user_id]
    file_bytes = file_info['content']
    file_name = file_info['name']
    
    # Extract text based on file type
    if file_name.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_bytes)
    elif file_name.lower().endswith('.txt'):
        text = extract_text_from_txt(file_bytes)
    else:
        return f"Unsupported file type: {file_name}. Please upload a PDF or TXT file."
    
    if text.startswith("Error"):
        return text
    
    # Use CrewAI for analysis and summarization
    summary = analyze_document_with_crewai(text, file_name)
    return summary

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
    ),
    Tool(
        name="CrewAI Document Analysis",
        func=lambda query: summarize_uploaded_file(int(query.split()[-1])) if query.split()[-1].isdigit() else "User ID not provided",
        description="Use this tool when user asks to analyze or summarize their uploaded document. Uses CrewAI agents for comprehensive document analysis and summarization."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Telegram Bot Setup ---
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- Telegram Bot Handlers ---
@bot.message_handler(commands=['start'])
def start(message):
    welcome_msg = """👋 Hello! I'm your AI Assistant with CrewAI document analysis!

📄 **Document Analysis Features:**
• Upload PDF or TXT files for advanced analysis
• Uses CrewAI agents for comprehensive document processing
• Get detailed summaries with key insights and main topics
• Professional document analysis and content extraction

🔍 **Search Features:**
• Ask me anything and I'll search the web for you
• Get current news and updates
• Find information on any topic

Upload a document and say "analyze" or "summarize" to get started!"""
    bot.reply_to(message, welcome_msg)

@bot.message_handler(content_types=['document'])
def handle_document(message):
    """Handle document uploads"""
    user_id = message.from_user.id
    document = message.document
    
    # Check file type
    if not (document.file_name.lower().endswith('.pdf') or document.file_name.lower().endswith('.txt')):
        bot.reply_to(message, "❌ Please upload only PDF or TXT files.")
        return
    
    try:
        # Download the file
        file_info = bot.get_file(document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Store file information
        user_files[user_id] = {
            'name': document.file_name,
            'content': downloaded_file,
            'size': document.file_size
        }
        
        bot.reply_to(message, f"✅ File '{document.file_name}' uploaded successfully!\n\nNow say 'analyze', 'summarize', or 'summarize this file' to get a CrewAI-powered analysis.")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error uploading file: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.text:
        user_input = message.text.strip().lower()
        user_id = message.from_user.id
        
        # Check for analysis/summarization requests
        analysis_keywords = ["analyze", "summarize", "summary", "summarise", "summarize this", "summarize file", "document analysis"]
        if any(keyword in user_input for keyword in analysis_keywords):
            if user_id in user_files:
                bot.reply_to(message, "🤖 Starting CrewAI document analysis... This may take a moment.")
                summary = summarize_uploaded_file(user_id)
                bot.send_message(message.chat.id, summary)
            else:
                bot.reply_to(message, "📄 No file uploaded. Please upload a PDF or TXT file first, then ask me to analyze or summarize it.")
            return
        
        # Check for greetings
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greet in user_input for greet in greetings):
            bot.reply_to(message, "👋 Hello! How can I assist you today?")
            return
        
        try:
            response = agent.run(message.text)
        except Exception as e:
            response = f"❌ Sorry, I ran into an error: {str(e)}"
        bot.reply_to(message, response)

# --- Main Application ---
if __name__ == "__main__":
    print("🤖 Bot is running with CrewAI document analysis...")
    bot.polling()
