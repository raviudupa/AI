from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.tools import tool

app = FastAPI(title="AI Tutor API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class GenerateQuestionsRequest(BaseModel):
    difficulty: str = "medium"
    num_questions: int = 5
    chapter: str = "all"  # "all" or specific chapter name

class StudyGuideRequest(BaseModel):
    topic: str = "general"
    focus: str = "highlights"  # "highlights" or "comprehensive"

class ConversationEntry(BaseModel):
    question: str
    answer: str
    timestamp: str

# Global state (in production, use a proper database)
pdf_tutor_system = None
conversation_history = []
generated_questions = []  # Track generated questions

# File to store conversation history
HISTORY_FILE = "conversation_history.json"
QUESTIONS_FILE = "generated_questions.json"

def load_conversation_history():
    """Load conversation history from file"""
    global conversation_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversation_history = [ConversationEntry(**entry) for entry in data]
        else:
            conversation_history = []
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        conversation_history = []

def save_conversation_history():
    """Save conversation history to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([entry.dict() for entry in conversation_history], f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

def load_generated_questions():
    """Load generated questions from file"""
    global generated_questions
    try:
        if os.path.exists(QUESTIONS_FILE):
            with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
                generated_questions = json.load(f)
        else:
            generated_questions = []
    except Exception as e:
        print(f"Error loading generated questions: {e}")
        generated_questions = []

def save_generated_questions():
    """Save generated questions to file"""
    try:
        with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(generated_questions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving generated questions: {e}")

# Load existing history and questions on startup
load_conversation_history()
load_generated_questions()

class PDFTutorSystem:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.pdf_content = None
        self.current_pdf_name = None
        self.chapters = []  # List of detected chapters
        self.setup_models()
    
    def setup_models(self):
        """Initialize the language models and embeddings"""
        try:
            # Check if API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
            
            # Initialize OpenAI models
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=api_key
            )
            
            self.embeddings = OpenAIEmbeddings(
                api_key=api_key
            )
            
        except Exception as e:
            raise Exception(f"Error setting up models: {str(e)}")
    
    def process_pdf(self, pdf_file, filename: str) -> bool:
        """Process uploaded PDF and create vector store"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file)
                tmp_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store with unique collection name
            collection_name = f"pdf_tutor_{filename.replace('.pdf', '').replace(' ', '_')}"
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=collection_name
            )
            
            # Store content for reference
            self.pdf_content = "\n".join([doc.page_content for doc in documents])
            self.current_pdf_name = filename
            
            # Extract chapters from the content
            self.extract_chapters()
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return True
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def extract_chapters(self):
        """Extract chapter information from the PDF content"""
        try:
            if not self.pdf_content:
                self.chapters = []
                return
            
            # Use LLM to identify chapters
            chapter_prompt = f"""
            Analyze the following document content and identify all chapters or major sections.
            Look for patterns like:
            - Chapter 1, Chapter 2, etc.
            - Section 1, Section 2, etc.
            - Part 1, Part 2, etc.
            - Any numbered or titled major divisions
            
            Document content (first 3000 characters):
            {self.pdf_content[:3000]}
            
            Return ONLY a JSON array of chapter names, like:
            ["Chapter 1: Introduction", "Chapter 2: Methods", "Chapter 3: Results"]
            
            If no clear chapters are found, return an empty array [].
            """
            
            response = self.llm.invoke(chapter_prompt)
            try:
                # Try to parse as JSON
                import json
                self.chapters = json.loads(response.content.strip())
            except:
                # If JSON parsing fails, try to extract from text
                content = response.content.strip()
                if content.startswith('[') and content.endswith(']'):
                    # Remove brackets and split by comma
                    content = content[1:-1]
                    self.chapters = [chapter.strip().strip('"') for chapter in content.split(',') if chapter.strip()]
                else:
                    self.chapters = []
                    
        except Exception as e:
            print(f"Error extracting chapters: {e}")
            self.chapters = []
    
    def create_agents(self):
        """Create CrewAI agents for different tutoring tasks"""
        
        # PDF Analyzer Agent
        pdf_analyzer = Agent(
            role='PDF Content Analyzer',
            goal='Analyze and understand the content of uploaded PDF documents',
            backstory="""You are an expert content analyzer with deep knowledge in 
            document processing and information extraction. You excel at understanding 
            complex documents and extracting key insights. You can identify main themes, 
            key concepts, and the overall structure of academic and technical documents.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.analyze_content_tool],
            llm=self.llm
        )
        
        # Question Generator Agent
        question_generator = Agent(
            role='Educational Question Generator',
            goal='Generate relevant and educational questions based on PDF content',
            backstory="""You are an expert educator who creates engaging and 
            thought-provoking questions. You know how to test understanding at 
            different levels - from basic recall to advanced analysis. You create 
            questions that encourage critical thinking and deep understanding.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.generate_questions_tool],
            llm=self.llm
        )
        
        # Tutor Agent
        tutor = Agent(
            role='Interactive AI Tutor',
            goal='Provide clear, helpful explanations and answer questions about the PDF content',
            backstory="""You are a patient and knowledgeable tutor who excels at 
            explaining complex concepts in simple terms. You adapt your teaching 
            style to the student's level and provide detailed, accurate answers. 
            You always cite specific parts of the document when answering questions.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.answer_question_tool],
            llm=self.llm
        )
        
        # Study Guide Generator Agent
        study_guide_generator = Agent(
            role='Study Guide Creator',
            goal='Create comprehensive study guides and learning materials from PDF content',
            backstory="""You are an expert curriculum designer who creates effective 
            study materials. You know how to organize information in ways that 
            facilitate learning and retention. You create structured guides with 
            clear sections, key points, and practical examples.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.create_study_guide_tool],
            llm=self.llm
        )
        
        return pdf_analyzer, question_generator, tutor, study_guide_generator
    
    @tool
    def analyze_content_tool(self, query: str) -> str:
        """Analyze the PDF content and provide insights"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        # Use RAG to get relevant content
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(query)
        
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        return f"Based on the PDF content: {context}"
    
    @tool
    def generate_questions_tool(self, difficulty: str = "medium", num_questions: int = 5) -> str:
        """Generate educational questions based on the PDF content"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        # Get key topics from the PDF
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.get_relevant_documents("main topics concepts")
        
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        Based on the following PDF content, generate {num_questions} {difficulty} level questions:
        
        Content: {context}
        
        Generate questions that:
        1. Test understanding of key concepts
        2. Are appropriate for {difficulty} difficulty
        3. Cover different aspects of the material
        4. Encourage critical thinking
        5. Include a mix of question types (multiple choice, short answer, essay)
        
        Format each question with a number and make them clear and specific.
        For multiple choice questions, provide 4 options (A, B, C, D).
        """
        
        return self.llm.invoke(prompt).content
    
    @tool
    def answer_question_tool(self, question: str) -> str:
        """Answer questions about the PDF content using RAG"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        # Create RAG chain
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        prompt_template = """
        You are a helpful AI tutor. Answer the following question based on the provided context.
        If the answer cannot be found in the context, say so clearly.
        
        Provide a comprehensive answer that:
        1. Directly addresses the question
        2. Cites specific information from the context
        3. Explains concepts clearly
        4. Provides additional insights when relevant
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain.invoke({"query": question})["result"]
    
    @tool
    def create_study_guide_tool(self, topic: str = "general") -> str:
        """Create a comprehensive study guide from the PDF content"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        # Get relevant content
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        relevant_docs = retriever.get_relevant_documents(topic)
        
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        Create a comprehensive study guide based on the following PDF content:
        
        Content: {context}
        
        The study guide should include:
        1. Executive Summary
        2. Key Concepts and Definitions
        3. Main Topics and Themes
        4. Important Details and Examples
        5. Study Tips and Strategies
        6. Practice Questions
        7. Further Reading Suggestions
        
        Format it clearly with headers, bullet points, and numbered lists where appropriate.
        Make it easy to follow and understand.
        """
        
        return self.llm.invoke(prompt).content
    
    def generate_summary(self) -> str:
        """Generate a comprehensive summary of the PDF"""
        if not self.pdf_content:
            return "No PDF content available."
        
        summary_prompt = f"""
        Please provide a comprehensive summary of the following document content.
        Include:
        1. Main topics and themes
        2. Key concepts and ideas
        3. Important details and findings
        4. Structure and organization of the content
        5. Target audience and purpose
        6. Key takeaways
        
        Document content:
        {self.pdf_content[:4000]}  # Limit content length
        
        Summary:
        """
        
        return self.llm.invoke(summary_prompt).content
    
    def generate_questions(self, difficulty: str = "medium", num_questions: int = 5, chapter: str = "all") -> str:
        """Generate questions using a simpler approach with RAG"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        try:
            # Determine search query based on chapter
            if chapter == "all":
                search_query = "main topics concepts key points"
            else:
                search_query = f"chapter {chapter} content topics concepts"
            
            # Get key topics from the PDF
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            relevant_docs = retriever.get_relevant_documents(search_query)
            
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Get previously generated questions to avoid duplicates
            global generated_questions
            previous_questions = "\n".join(generated_questions) if generated_questions else "No previous questions."
            
            chapter_info = f"Focus on chapter: {chapter}" if chapter != "all" else "Cover all chapters"
            
            prompt = f"""
            Based on the following PDF content, generate {num_questions} {difficulty} level questions:
            
            {chapter_info}
            
            Content: {context}
            
            IMPORTANT: Avoid generating questions that are similar to these previously generated questions:
            {previous_questions}
            
            Generate questions that:
            1. Test understanding of key concepts
            2. Are appropriate for {difficulty} difficulty
            3. Cover different aspects of the material
            4. Encourage critical thinking
            5. Include a mix of question types (multiple choice, short answer, essay)
            6. Are UNIQUE and different from previous questions
            7. Focus specifically on the requested chapter content
            
            Format each question with a number and make them clear and specific.
            For multiple choice questions, provide 4 options (A, B, C, D).
            
            Make sure the questions are directly related to the content provided and are not duplicates.
            """
            
            response = self.llm.invoke(prompt)
            new_questions = response.content
            
            # Extract and store the new questions
            lines = new_questions.split('\n')
            question_lines = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('Q:') or line.startswith('Question:')):
                    question_lines.append(line)
            
            # Add new questions to the tracking list
            generated_questions.extend(question_lines)
            save_generated_questions()
            
            return new_questions
            
        except Exception as e:
            return f"Error generating questions: {str(e)}"
    
    def create_study_guide(self, topic: str = "general", focus: str = "highlights") -> str:
        """Create a study guide or highlight important areas using RAG"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        try:
            # Get relevant content
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
            relevant_docs = retriever.get_relevant_documents(topic)
            
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            if focus == "highlights":
                prompt = f"""
                Analyze the following PDF content and identify the MOST IMPORTANT areas that should be learned and understood.
                
                Content: {context}
                
                Create a focused "Learning Highlights" guide that includes:
                
                🎯 **CRITICAL CONCEPTS TO MASTER**
                - List the 5-7 most important concepts that are fundamental to understanding this material
                - Explain why each concept is crucial
                
                ⚡ **KEY TAKEAWAYS**
                - Highlight the main points that students must remember
                - Focus on actionable insights and practical applications
                
                🔍 **AREAS REQUIRING SPECIAL ATTENTION**
                - Identify complex topics that need extra study time
                - Point out common misconceptions or tricky areas
                
                📚 **ESSENTIAL KNOWLEDGE CHECKLIST**
                - Create a checklist of what students should be able to explain or demonstrate
                - Include both theoretical understanding and practical application
                
                💡 **STUDY PRIORITIES**
                - Rank topics by importance for exam/assessment preparation
                - Suggest time allocation for different sections
                
                Format with clear headers, bullet points, and emphasis on what's truly important.
                Focus on the topic: {topic}
                """
            else:
                prompt = f"""
                Create a comprehensive study guide based on the following PDF content:
                
                Content: {context}
                
                The study guide should include:
                1. Executive Summary
                2. Key Concepts and Definitions
                3. Main Topics and Themes
                4. Important Details and Examples
                5. Study Tips and Strategies
                6. Practice Questions
                7. Further Reading Suggestions
                
                Format it clearly with headers, bullet points, and numbered lists where appropriate.
                Make it easy to follow and understand.
                Focus on the topic: {topic}
                """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error creating study guide: {str(e)}"
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the PDF content using RAG"""
        if not self.vectorstore:
            return "No PDF content available. Please upload a PDF first."
        
        try:
            # Create RAG chain
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            
            prompt_template = """
            You are a helpful AI tutor. Answer the following question based on the provided context.
            If the answer cannot be found in the context, say so clearly.
            
            Provide a comprehensive answer that:
            1. Directly addresses the question
            2. Cites specific information from the context
            3. Explains concepts clearly
            4. Provides additional insights when relevant
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            return qa_chain.invoke({"query": question})["result"]
            
        except Exception as e:
            return f"Error answering question: {str(e)}"

# API Routes
@app.get("/")
async def root():
    return {"message": "AI Tutor API is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_tutor_system
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Initialize tutor system
        pdf_tutor_system = PDFTutorSystem()
        
        # Process PDF
        success = pdf_tutor_system.process_pdf(content, file.filename)
        
        if success:
            return {
                "message": "PDF processed successfully",
                "filename": file.filename,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary")
async def get_summary():
    global pdf_tutor_system
    
    if not pdf_tutor_system:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
    
    try:
        summary = pdf_tutor_system.generate_summary()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-questions")
async def generate_questions(request: GenerateQuestionsRequest):
    global pdf_tutor_system
    
    if not pdf_tutor_system:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
    
    try:
        questions = pdf_tutor_system.generate_questions(
            request.difficulty, 
            request.num_questions,
            request.chapter
        )
        return {"questions": questions}
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    global pdf_tutor_system, conversation_history
    
    if not pdf_tutor_system:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
    
    try:
        answer = pdf_tutor_system.answer_question(request.question)
        
        # Add to conversation history
        conversation_entry = ConversationEntry(
            question=request.question,
            answer=answer,
            timestamp=datetime.now().isoformat()
        )
        conversation_history.append(conversation_entry)
        
        # Save to file
        save_conversation_history()
        
        return {"answer": answer}
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

@app.post("/create-study-guide")
async def create_study_guide(request: StudyGuideRequest):
    global pdf_tutor_system
    
    if not pdf_tutor_system:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
    
    try:
        study_guide = pdf_tutor_system.create_study_guide(request.topic, request.focus)
        return {"study_guide": study_guide}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-history")
async def get_conversation_history():
    global conversation_history
    return {"conversation": conversation_history}

@app.delete("/conversation-history")
async def clear_conversation_history():
    global conversation_history
    conversation_history = []
    save_conversation_history()  # Save empty history to file
    return {"message": "Conversation history cleared"}

@app.get("/generated-questions")
async def get_generated_questions():
    global generated_questions
    return {"questions": generated_questions}

@app.delete("/generated-questions")
async def clear_generated_questions():
    global generated_questions
    generated_questions = []
    save_generated_questions()  # Save empty questions to file
    return {"message": "Generated questions cleared"}

@app.get("/status")
async def get_status():
    global pdf_tutor_system, generated_questions
    return {
        "pdf_uploaded": pdf_tutor_system is not None,
        "current_pdf": pdf_tutor_system.current_pdf_name if pdf_tutor_system else None,
        "conversation_count": len(conversation_history),
        "generated_questions_count": len(generated_questions),
        "chapters": pdf_tutor_system.chapters if pdf_tutor_system else []
    }

@app.get("/chapters")
async def get_chapters():
    global pdf_tutor_system
    if not pdf_tutor_system:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
    
    return {"chapters": pdf_tutor_system.chapters}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 