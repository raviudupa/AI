import re
import difflib
import random
import logging
from langchain_openai import ChatOpenAI

# You may need to import llm, OPENAI_MODEL, and other dependencies from your main config if needed.

def evaluate_answer(question, answer):
    from langchain.chat_models import ChatOpenAI
    import re
    import logging
    from ai_interview.backend.main import OPENAI_MODEL
    # Use a lower temperature for consistent scoring
    eval_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
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
        logging.warning(f"Evaluation Error: {e}")
    return 5  # fallback score if evaluation fails

def generate_question(level, topic="general", asked_questions=None):
    from ai_interview.backend.main import PROMPT_TEMPLATES, load_global_history, save_global_history, llm, tavily, OPENAI_MODEL, user_states
    if asked_questions is None:
        asked_questions = []
    global_history = load_global_history()
    temperature = 0.6  # You may want to make this configurable
    for _ in range(20):  # Try more times to avoid repeats
        try:
            search_prompt = random.choice(PROMPT_TEMPLATES).format(topic=topic, level=level)
            results = tavily.search(search_prompt, num_results=10)
            content = random.choice(results['results'])['content'].strip()
            question = content
            # Expanded filtering
            if not question.endswith('?'):
                continue
            if len(question) < 10 or len(question) > 200:
                continue
            if question in asked_questions or question in global_history:
                continue
            # Add to global history and user's asked_questions
            global_history.add(question)
            save_global_history(global_history)
            return question
        except Exception:
            continue
    # fallback to LLM
    prompt = (
        f"Generate a new, uncommon, unique, clear, {level} difficulty interview question about {topic}. "
        "Do not repeat previous questions. Only output the question itself."
    )
    llm_dynamic = ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)
    question = llm_dynamic.invoke(prompt).content.strip()
    if not question.endswith('?'):
        question += '?'
    if question not in asked_questions and question not in global_history:
        global_history.add(question)
        save_global_history(global_history)
        return question
    # If all else fails, return a message indicating no new question could be generated
    return "No new unique question could be generated. Please try a different topic or reset the interview."

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
    # You may need to import llm from your main config
    from ai_interview.backend.main import llm
    keywords_str = llm.invoke(prompt).content.strip()
    logging.info(f"Extracted keywords: {keywords_str}")
    keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
    keywords = [kw for kw in keywords if kw not in STOPWORDS and len(kw) > 2]
    logging.info(f"Filtered keywords: {keywords}")
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