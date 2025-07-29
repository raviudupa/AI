import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
import openai
import re
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from tavily import TavilyClient
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
import tempfile
import subprocess
import random
import json
import logging
import difflib
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from tempfile import NamedTemporaryFile, gettempdir
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import time

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

openai.api_key = OPENAI_API_KEY

# Set up LLM
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.6)
# Set up Tavily
tavily = TavilyClient(TAVILY_API_KEY)

# Difficulty to temperature mapping
DIFFICULTY_TEMPERATURE = {
    "easy": 0.4,
    "moderate": 0.6,
    "hard": 0.8,
    "very hard": 1.0
}
# Difficulty to threshold mapping for answer correctness
DIFFICULTY_THRESHOLD = {
    "easy": 0.3,
    "moderate": 0.5,
    "hard": 0.6,
    "very hard": 0.7  # fallback for very hard
}

app = Flask(__name__)

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# In-memory user state (for demo; use persistent store in production)
user_states = {}

# Helper: split audio into 60s chunks

def convert_webm_to_wav(input_path, output_path):
    cmd = [
        'ffmpeg', '-y', '-i', input_path, output_path
    ]
    subprocess.run(cmd, check=True)

def split_audio_into_chunks(audio_path, chunk_length_ms=60000):
    # Validate file type
    valid_exts = ['.webm', '.wav', '.mp3']
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in valid_exts:
        raise ValueError(f"Unsupported audio format: {ext}")
    temp_dir = gettempdir()
    base_uuid = str(uuid.uuid4())
    # Convert webm to wav if needed
    if ext == '.webm':
        try:
            audio = AudioSegment.from_file(audio_path, format='webm')
            wav_path = os.path.join(temp_dir, f"{base_uuid}.wav")
            audio.export(wav_path, format='wav')
            audio_path = wav_path
        except Exception:
            # fallback: use ffmpeg CLI
            wav_path = os.path.join(temp_dir, f"{base_uuid}.wav")
            cmd = [
                'ffmpeg', '-y', '-i', audio_path, wav_path
            ]
            subprocess.run(cmd, check=True)
            audio = AudioSegment.from_file(wav_path, format='wav')
            audio_path = wav_path
    else:
        audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_path = os.path.join(temp_dir, f"{base_uuid}_chunk_{i//chunk_length_ms}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio_chunks(chunk_paths):
    full_text = ""
    for chunk_path in chunk_paths:
        with open(chunk_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            full_text += transcript.text + " "
        os.remove(chunk_path)
    return full_text.strip()

GLOBAL_HISTORY_FILE = 'asked_questions_history.json'

def load_global_history():
    if os.path.exists(GLOBAL_HISTORY_FILE):
        with open(GLOBAL_HISTORY_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_global_history(history):
    with open(GLOBAL_HISTORY_FILE, 'w') as f:
        json.dump(list(history), f)

PROMPT_TEMPLATES = [
    "{topic} interview question, difficulty: {level}",
    "{topic} technical interview question, difficulty: {level}",
    "{topic} coding interview question, {level}",
    "{topic} scenario interview question, {level}",
    "Uncommon {topic} interview question, {level}",
    "Challenging {topic} interview question, {level}",
    "Behavioral {topic} interview question, {level}",
    "Situational {topic} interview question, {level}",
    "Practical {topic} interview question, {level}",
    "Real-world {topic} interview question, {level}"
]

QUESTION_START_WORDS = (
    "what", "how", "why", "explain", "describe", "tell", "give", "define",
    "when", "where", "who", "which", "can", "is", "are", "do", "does", "could", "would", "should"
)
LIST_KEYWORDS = [
    "top 10", "top ten", "10 most", "most difficult", "most common", "frequently asked",
    "list of", "sample questions", "interview questions and answers", "questions asked in",
    "multiple questions", "common interview questions", "top interview questions", "answers:"
]

def generate_llm_unique_question(level, topic, asked_questions=None):
    if asked_questions is None:
        asked_questions = []
    global_history = load_global_history()
    for _ in range(10):
        prompt = (
            f"Generate a unique, clear, and never-before-asked {level} interview question about {topic}. "
            "Do not repeat any previous questions. Only output the question itself."
        )
        llm_dynamic = ChatOpenAI(model=OPENAI_MODEL, temperature=0.8)
        question = llm_dynamic.invoke(prompt).content.strip()
        if not question.endswith('?'):
            question += '?'
        if question not in asked_questions and question not in global_history:
            global_history.add(question)
            save_global_history(global_history)
            return question
    return question  # fallback

def generate_question(level, topic="general", asked_questions=None, force_llm=False):
    if asked_questions is None:
        asked_questions = []
    global_history = load_global_history()
    temperature = DIFFICULTY_TEMPERATURE.get(level, 0.6)
    if force_llm:
        return generate_llm_unique_question(level, topic, asked_questions)
    for _ in range(10):
        try:
            search_prompt = random.choice(PROMPT_TEMPLATES).format(topic=topic, level=level)
            results = tavily.search(search_prompt, num_results=10)
            content = random.choice(results['results'])['content'].strip()
            question = content
            if not question.endswith('?'):
                continue
            if len(question) < 10 or len(question) > 200:
                continue
            if any(x in question.lower() for x in LIST_KEYWORDS):
                continue
            if not question.lower().startswith(QUESTION_START_WORDS):
                continue
            if question in asked_questions or question in global_history:
                continue
            global_history.add(question)
            save_global_history(global_history)
            return question
        except Exception:
            continue
    # fallback to LLM
    return generate_llm_unique_question(level, topic, asked_questions)

def evaluate_answer(question, answer):
    from langchain.chat_models import ChatOpenAI
    import re

    # Use a lower temperature for consistent scoring
    eval_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

    # Updated scoring prompt with clear evaluation criteria
    prompt = f"""
You are an expert interview evaluator. Rate the candidate's answer to the question below on a scale from 1 to 10, based on:

- ✅ Accuracy: Does the answer correctly address the question?
- ✅ Completeness: Does it cover important points?
- ✅ Clarity: Is the explanation understandable?

Only return a single number between 1 and 10. No explanation or extra words.

Question: {question}
Answer: {answer}
Score:"""

    try:
        result = eval_llm.invoke(prompt).content.strip()
        match = re.search(r'\d+', result)
        if match:
            score = int(match.group())
            return min(10, max(1, score))  # Clamp between 1 and 10
    except Exception as e:
        print("Evaluation Error:", e)

    return 5  # fallback score if evaluation fails

def preprocess_for_tts(text):
    # Replace multiple commas or dashes with a single one
    text = re.sub(r',\s*,+', ', ', text)
    text = re.sub(r'-\s*-+', '-', text)
    # Replace ellipses with a period
    text = text.replace('...', '.')
    # Split long sentences (>25 words) into shorter ones
    sentences = re.split(r'(?<=[.?!])\s+', text)
    processed = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 25:
            # Split into chunks of 20 words
            for i in range(0, len(words), 20):
                processed.append(' '.join(words[i:i+20]) + '.')
        else:
            processed.append(sentence)
    return ' '.join(processed)

def generate_tts_audio(text, voice="alloy"):
    tts_text = preprocess_for_tts(text)
    tts_response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=tts_text
    )
    return io.BytesIO(tts_response.content)

def keyword_accuracy_evaluation(question, user_answer, model_answer, threshold=0.7):
    if not user_answer.strip():
        return {
            "keywords": {},
            "accuracy": 0,
            "verdict": "not accurate"
        }
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        STOPWORDS = set(stopwords.words('english'))
    prompt = (
        f"List the 3-7 most important technical keywords or concepts (single words or short phrases, comma-separated, no explanations) from the following answer to the question:\n"
        f"Question: {question}\n"
        f"Answer: {model_answer}\n"
        "Keywords:"
    )
    keywords_str = llm.invoke(prompt).content.strip()
    print("Extracted keywords:", keywords_str)
    keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
    # Filter out stopwords and very short keywords
    keywords = [kw for kw in keywords if kw not in STOPWORDS and len(kw) > 2]
    print("Filtered keywords:", keywords)
    user_answer_lower = user_answer.lower()
    user_words = set(user_answer_lower.split())
    keyword_results = {}
    for kw in keywords:
        found = False
        for word in user_words:
            if kw == word or (difflib.SequenceMatcher(None, kw, word).ratio() > 0.9):
                found = True
                break
        keyword_results[kw] = found
    if keywords:
        matched = sum(keyword_results.values())
        accuracy = matched / len(keywords)
    else:
        accuracy = 0
    verdict = "accurate" if accuracy >= threshold else "not accurate"
    return {
        "keywords": keyword_results,
        "accuracy": accuracy,
        "verdict": verdict
    }

INTERVIEW_FLOW = {
    # 'total_questions': 6,  # Remove this line
    'progression': ['easy', 'moderate', 'moderate', 'hard', 'hard', 'very hard']
}

@app.route("/start", methods=["POST"])
def start():
    user_id = request.json.get("user_id")
    state = user_states.get(user_id)
    if not state:
        user_states[user_id] = {
            "level": "easy",
            "step": 0,
            "score": 0,
            "questions": 0,
            "correct": 0,
            "topic": None,
            "asked_questions": []
        }
        state = user_states[user_id]
    greeting = "👋 Welcome to the AI Voice Interview!"
    if state.get("topic"):
        # If topic is already set, ask first question
        question = generate_question("easy", topic=state["topic"])
        state["level"] = "easy"
        state["last_question"] = question
        user_states[user_id] = state
        return jsonify({"greeting": greeting, "question": question, "level": "easy"})
    else:
        return jsonify({"greeting": greeting, "prompt": "Please set your interview topic to begin."})

@app.route("/set_topic", methods=["POST"])
def set_topic():
    import random
    user_id = request.json.get("user_id")
    topic = request.json.get("topic")
    if user_id not in user_states:
        user_states[user_id] = {
            "level": "easy",
            "step": 0,
            "score": 0,
            "questions": 0,
            "correct": 0,
            "topic": None,
            "asked_questions": []
        }
    user_states[user_id]["topic"] = topic
    # Ask first question immediately, use web search with LLM rephrasing
    asked_questions = user_states[user_id].get("asked_questions", [])
    # Randomize first question difficulty and temperature
    first_level = random.choice(["easy", "moderate"])
    first_temp = random.choice([0.6, 0.8, 1.0])
    for _ in range(5):
        try:
            results = tavily.search(f"{topic} interview question, difficulty: {first_level}", num_results=1)
            content = results['results'][0]['content'].strip()
            prompt = (
                f"Rewrite the following as a single, clear, complete {first_level} difficulty interview question about {topic}. "
                "Only output the question itself, do not include explanations, lists, or multiple questions.\n\n"
                f"Text:\n{content}"
            )
            llm_dynamic = ChatOpenAI(model=OPENAI_MODEL, temperature=first_temp)
            question = llm_dynamic.invoke(prompt).content.strip()
            if question.endswith('?') and question not in asked_questions:
                break
        except Exception:
            continue
    else:
        # fallback to LLM if all web attempts fail
        prompt = (
            f"Generate a single, clear, {first_level} difficulty interview question about {topic}. "
            "Do not include explanations, lists, or multiple questions. Only output the question itself."
        )
        llm_dynamic = ChatOpenAI(model=OPENAI_MODEL, temperature=first_temp)
        question = llm_dynamic.invoke(prompt).content.strip()
        if not question.endswith('?'):
            question += '?'
    user_states[user_id]["level"] = first_level
    user_states[user_id]["last_question"] = question
    user_states[user_id]["asked_questions"] = asked_questions + [question]
    tts_url = f"/tts?text={question}"
    return jsonify({"message": f"Topic set to {topic}. Let's begin.", "question": question, "tts_url": tts_url, "level": first_level})

@app.route("/next_question", methods=["POST"])
def next_question():
    start = time.time()
    user_id = request.json.get("user_id")
    state = user_states.get(user_id)
    if not state or not state.get("topic"):
        return jsonify({"error": "Set topic first."}), 400
    progression = INTERVIEW_FLOW["progression"]
    qnum = state["questions"] if "questions" in state else 0
    if qnum < len(progression):
        level = progression[qnum]
    else:
        level = 'hard'
    asked_questions = user_states[user_id].get("asked_questions", [])
    # For first 5 questions, use LLM only
    if qnum < 5:
        question = generate_llm_unique_question(level, topic=state["topic"], asked_questions=asked_questions)
    else:
        question = generate_question(level, topic=state["topic"], asked_questions=asked_questions)
    user_states[user_id]["level"] = level
    user_states[user_id]["last_question"] = question
    user_states[user_id]["asked_questions"] = asked_questions + [question]
    tts_url = f"/tts?text={question}"
    print("/next_question took", time.time() - start, "seconds")
    return jsonify({"question": question, "tts_url": tts_url, "level": level})

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    start = time.time()
    try:
        user_id = request.form.get("user_id")
        if "audio" not in request.files:
            print("/upload_audio took", time.time() - start, "seconds")
            return jsonify({"error": "No audio file uploaded."}), 400
        audio_file = request.files["audio"]
        if audio_file.filename == '':
            print("/upload_audio took", time.time() - start, "seconds")
            return jsonify({"error": "No selected file."}), 400
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_in:
            audio_file.save(temp_in)
            temp_in_path = temp_in.name
        file_size = os.path.getsize(temp_in_path)
        logging.info(f"Received file: {audio_file.filename}, size: {file_size} bytes, mimetype: {audio_file.mimetype}")
        if file_size == 0:
            os.remove(temp_in_path)
            print("/upload_audio took", time.time() - start, "seconds")
            return jsonify({"error": "Uploaded file is empty. Please ensure your microphone is working and try again."}), 400
        # Try to open with AudioSegment to check validity
        try:
            AudioSegment.from_file(temp_in_path)
        except Exception as e:
            os.remove(temp_in_path)
            logging.error(f"Invalid audio file: {e}")
            print("/upload_audio took", time.time() - start, "seconds")
            return jsonify({"error": "Uploaded file is not a valid audio file. Please try recording again."}), 400
        chunk_paths = []
        try:
            chunk_paths = split_audio_into_chunks(temp_in_path, chunk_length_ms=60000)
            answer = transcribe_audio_chunks(chunk_paths)
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            for path in [temp_in_path] + chunk_paths:
                if os.path.exists(path): os.remove(path)
            print("/upload_audio took", time.time() - start, "seconds")
            return jsonify({"error": f"Transcription failed: {e}"}), 500
        if os.path.exists(temp_in_path): os.remove(temp_in_path)
        print("/upload_audio took", time.time() - start, "seconds")
        return jsonify({"transcript": answer})
    except Exception as e:
        logging.error(f"Upload audio error: {e}")
        print("/upload_audio took", time.time() - start, "seconds")
        return jsonify({"error": str(e)}), 500

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    start = time.time()
    user_id = request.json.get("user_id")
    answer = request.json.get("answer")
    # Add explicit end flag
    end_interview = request.json.get("end_interview", False)
    state = user_states.get(user_id)
    if not state or not state.get("last_question"):
        print("/submit_answer took", time.time() - start, "seconds")
        return jsonify({"error": "No question to answer."}), 400
    question = state["last_question"]
    if "answered_count" not in state:
        state["answered_count"] = 0
    if "skipped" not in state:
        state["skipped"] = 0
    if not answer.strip():
        state["questions"] += 1
        state["skipped"] += 1
        prompt = (
            f"Provide a very short, concise, and neatly formatted (e.g., bullet points) model answer to the following interview question.\n"
            f"Question: {question}\n"
            "Answer:"
        )
        correct_answer = llm.invoke(prompt).content.strip()
        if end_interview:
            # Only end if explicitly requested
            correct = state.get("correct", 0)
            answered = state.get("answered_count", 0)
            skipped = state.get("skipped", 0)
            accuracy = 0.0
            if "all_answers" in state:
                all_answers = state["all_answers"]
            else:
                all_answers = []
            msg = f"Interview complete! You attempted {answered+skipped} questions and answered {answered} of them. You answered {correct} correctly. Skipped {skipped}. Accuracy: {accuracy:.1f}%"
            improvement_prompt = (
                f"You are an expert interview coach. The candidate attempted {answered+skipped} questions and answered {answered} of them, with {correct} correct.\n"
                f"Here are the questions and their answers (blank means skipped):\n"
                + "\n".join([f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}" for i, qa in enumerate(all_answers)]) +
                "\nBased on this, suggest 2-3 specific improvements for their next interview. Be concise and encouraging. If possible, mention which areas to focus on based on their actual answers."
            )
            improvement_msg = llm.invoke(improvement_prompt).content.strip()
            user_states[user_id] = {"level": "easy", "step": 0, "score": 0, "questions": 0, "correct": 0, "topic": state.get("topic"), "asked_questions": []}
            print("/submit_answer took", time.time() - start, "seconds")
            return jsonify({
                "message": msg,
                "improvement": improvement_msg,
                "interview_complete": True,
                "accuracy": accuracy,
                "answers_transcribed": [qa['answer'] for qa in all_answers],
                "skipped": skipped
            })
        else:
            if "all_answers" not in state:
                state["all_answers"] = []
            state["all_answers"].append({"question": question, "answer": answer.strip()})
            user_states[user_id] = state
            print("/submit_answer took", time.time() - start, "seconds")
            return jsonify({
                "feedback": "No answer detected. Please record a longer response or try the next question.",
                "score": 0,
                "interview_complete": False,
                "transcription": answer,
                "correct_answer": correct_answer
            })
    # Dynamic threshold based on difficulty
    level = state.get("level", "moderate")
    score = evaluate_answer(question, answer)
    state["questions"] += 1
    state["answered_count"] += 1
    if "all_answers" not in state:
        state["all_answers"] = []
    state["all_answers"].append({"question": question, "answer": answer.strip()})
    correct_answer = None
    improvement_points = None
    threshold = DIFFICULTY_THRESHOLD.get(level, 0.6)
    # Generate model answer for keyword evaluation (only once)
    prompt = (
        f"Provide a very short, concise, and neatly formatted (e.g., bullet points) model answer to the following interview question.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    model_answer = llm.invoke(prompt).content.strip()
    keyword_eval = keyword_accuracy_evaluation(question, answer, model_answer, threshold=threshold)
    # Generate contextual follow-up question
    followup_prompt = f"""
You are an expert interviewer. Given the following interview question and the candidate's answer, generate a single, clear, and relevant follow-up question to probe deeper or clarify their response. Only output the follow-up question itself.

Previous Question: {question}
Candidate's Answer: {answer}
Follow-up Question:
"""
    followup_question = llm.invoke(followup_prompt).content.strip()
    if score / 10 > threshold:
        state["correct"] = state.get("correct", 0) + 1
        feedback = "This answer is considered correct!"
    else:
        improvement_prompt = (
            f"You are an expert interview evaluator. The candidate answered the following question:\n"
            f"Question: {question}\n"
            f"Candidate's answer: {answer}\n"
            "List in 2-3 short bullet points what important points or concepts were missing or could be improved. Be concise."
        )
        improvement_points = llm.invoke(improvement_prompt).content.strip()
        feedback = "This answer was not accurate enough."
    # In /submit_answer, remove total_questions check for ending the interview. Only end if user triggers 'End Test' or 'Submit Test'.
    user_states[user_id] = state
    response = {
        "feedback": feedback,
        "score": score,
        "interview_complete": False,
        "transcription": answer,
        "keyword_evaluation": keyword_eval,
        "followup_question": followup_question
    }
    if model_answer:
        response["correct_answer"] = model_answer
    if improvement_points:
        response["improvement_points"] = improvement_points
    print("/submit_answer took", time.time() - start, "seconds")
    return jsonify(response)

@app.route("/get_model_answer", methods=["POST"])
def get_model_answer():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400
    prompt = (
        f"Provide a very short, concise, and neatly formatted (e.g., bullet points) model answer to the following interview question.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    model_answer = llm.invoke(prompt).content.strip()
    return jsonify({"model_answer": model_answer})

@app.route("/explain_answer", methods=["POST"])
def explain_answer():
    data = request.json
    question = data.get("question")
    user_answer = data.get("user_answer")
    if not question or not user_answer:
        return jsonify({"error": "Question and user_answer required."}), 400
    prompt = (
        f"You are an expert interview evaluator. Given the following question and the candidate's answer, explain in 2-4 sentences why the answer is correct or incorrect, and what a perfect answer would include. Be clear and concise.\n"
        f"Question: {question}\n"
        f"Candidate's Answer: {user_answer}\n"
        "Explanation:"
    )
    explanation = llm.invoke(prompt).content.strip()
    # Also provide the model answer
    model_prompt = (
        f"Provide a very short, concise, and neatly formatted (e.g., bullet points) model answer to the following interview question.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    model_answer = llm.invoke(model_prompt).content.strip()
    return jsonify({"explanation": explanation, "model_answer": model_answer})

@app.route("/tts", methods=["GET", "POST"])
def tts():
    if request.method == "POST":
        data = request.json
        text = data.get("text", "")
        # Ignore incoming voice, always use 'alloy'
        voice = "alloy"
    else:  # GET
        text = request.args.get("text", "")
        # Ignore incoming voice, always use 'alloy'
        voice = "alloy"
    if not text:
        return jsonify({"error": "No text provided."}), 400
    # Preprocess text for TTS
    audio_io = generate_tts_audio(text, voice=voice)
    return send_file(audio_io, mimetype="audio/mpeg", as_attachment=False, download_name="tts.mp3")

@app.route("/tts_voices", methods=["GET"])
def tts_voices():
    voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
    return jsonify({"voices": voices})

@app.route("/get_improvement", methods=["POST"])
def get_improvement():
    data = request.json
    all_answers = data.get("all_answers", [])
    correct = data.get("correct", 0)
    answered = data.get("answered", 0)
    total = data.get("total", 0)
    prompt = (
        f"You are an expert interview coach. The candidate attempted {total} questions and answered {answered} of them, with {correct} correct.\n"
        f"Here are the questions and their answers (blank means skipped):\n"
        + "\n".join([f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}" for i, qa in enumerate(all_answers)]) +
        "\nBased on this, suggest 2-3 specific improvements for their next interview. Be concise and encouraging. If possible, mention which areas to focus on based on their actual answers."
    )
    improvement_msg = llm.invoke(prompt).content.strip()
    return jsonify({"improvement": improvement_msg})

if __name__ == "__main__":
    app.run(debug=True) 