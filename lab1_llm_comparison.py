import os
import time
import warnings
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from groq import Groq

# Подавляем системные предупреждения устаревших зависимостей Python
warnings.filterwarnings("ignore")

# Загрузка переменных окружения из .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
openrouter_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

prompt = "Объясни в трех предложениях, что такое машинное обучение."

results = []

# --- 1. GROQ ---
print("Запрос к Groq...")
try:
    groq_client = Groq(api_key=groq_api_key)
    
    # Исключаем служебные модели безопасности/guard и берем текстовую генеративную модель
    all_models = [m.id for m in groq_client.models.list().data]
    gen_models = [m for m in all_models if "guard" not in m and "whisper" not in m]
    target_model = gen_models[0] if gen_models else "llama-3.3-70b-versatile"
    
    start_time = time.time()
    groq_response = groq_client.chat.completions.create(
        model=target_model, 
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_time = round(time.time() - start_time, 2)
    
    usage = groq_response.usage
    results.append({
        "platform": f"Groq ({target_model})",
        "time": elapsed_time,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    })
    print(f"✓ Groq: Успешно ({target_model})!")
except Exception as e:
    results.append({
        "platform": "Groq",
        "error": str(e)
    })

# --- 2. GEMINI ---
print("Запрос к Gemini...")
try:
    genai.configure(api_key=google_api_key)
    
    # Испольуем актуальную модель gemini-3.8-flash, запрашиваемую сервером
    model = genai.GenerativeModel("gemini-3.8-flash")
    
    start_time = time.time()
    gemini_response = model.generate_content(prompt)
    elapsed_time = round(time.time() - start_time, 2)

    prompt_tokens = gemini_response.usage_metadata.prompt_token_count if hasattr(gemini_response, 'usage_metadata') else "N/A"
    candidates_tokens = gemini_response.usage_metadata.candidates_token_count if hasattr(gemini_response, 'usage_metadata') else "N/A"
    total_tokens = gemini_response.usage_metadata.total_token_count if hasattr(gemini_response, 'usage_metadata') else "N/A"

    results.append({
        "platform": "Gemini (gemini-3.8-flash)",
        "time": elapsed_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": candidates_tokens,
        "total_tokens": total_tokens
    })
    print("✓ Gemini: Успешно (gemini-3.8-flash)!")
except Exception as e:
    results.append({
        "platform": "Gemini",
        "error": str(e)
    })

# --- 3. OPENROUTER ---
print("Запрос к OpenRouter...")
try:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    start_time = time.time()
    or_response = openrouter_client.chat.completions.create(
        model="openrouter/free", 
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_time = round(time.time() - start_time, 2)

    usage = or_response.usage
    results.append({
        "platform": "OpenRouter (Free)",
        "time": elapsed_time,
        "prompt_tokens": usage.prompt_tokens if usage else "N/A",
        "completion_tokens": usage.completion_tokens if usage else "N/A",
        "total_tokens": usage.total_tokens if usage else "N/A"
    })
    print("✓ OpenRouter: Успешно!")
except Exception as e:
    results.append({
        "platform": "OpenRouter",
        "error": str(e)
    })

# --- ВЫВОД РЕЗУЛЬТАТОВ ---
print("\n" + "=" * 80)
header = f"{'Платформа':<32} | {'Время (сек)':<11} | {'Промпт токены':<13} | {'Ответ токены':<12} | {'Всего токенов':<13}"
print(header)
print("=" * 80)

for res in results:
    if "error" in res:
        print(f"{res['platform']:<32} | ОШИБКА: {res['error']}")
    else:
        print(f"{res['platform']:<32} | {res['time']:<11} | {res['prompt_tokens']:<13} | {res['completion_tokens']:<12} | {res['total_tokens']:<13}")

print("=" * 80)