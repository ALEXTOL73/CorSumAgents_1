"""
Конфигурация проекта CorSumAgentsAI
Версия 5.7.0 - Новая формула CorScore: ΔWER + 8*ΔLev + (1 - Perpl/100)*0.5
"""
import os
from pathlib import Path

# =============================================================================
# 🌐 GLOBAL LANGUAGE CONFIGURATION
# =============================================================================
# ВАЖНО: 'ru' = русский (папка data/RUS/), 'en' = english (папка data/ENG/)
LANGUAGE = 'en'  # 'ru' for Russian → data/RUS/, 'en' for English → data/ENG/
DATA_LANG_DIR = 'RUS' if LANGUAGE.lower() == 'ru' else 'ENG'

# =============================================================================
# 🔹 Обработка Файлов
# =============================================================================
SKIP_PROCESSED = 1       # 1 - Пропустить обработанные файлы

# 🔹 LM Studio Настройки
# =============================================================================
LM_STUDIO_HOST = "127.0.0.1"
LM_STUDIO_PORT = 1234
MODEL_NAME = "gemma-4-12b-it"
REQUEST_TIMEOUT = 100
LM_STUDIO_MIN_REQUEST_INTERVAL = 0.2
LM_STUDIO_MAX_TOKENS = 4096

# =============================================================================
# ⭐ ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА
# =============================================================================
PARALLEL_TESTS_ENABLED = True
PARALLEL_TESTS_MAX_WORKERS = 4

# =============================================================================
# ⭐ РАННИЙ ОСТАНОВ (EARLY STOPPING) ДЛЯ АНСАМБЛЕЙ
# =============================================================================
EARLY_STOP_CORRECTION = True
EARLY_STOP_CORRECTION_THRESHOLD = 0.95      # если CorScore >= этого значения, прекращаем генерацию
EARLY_STOP_SUMMARY = True
EARLY_STOP_SUMMARY_THRESHOLD = 0.90

# =============================================================================
# 🔹 Настройки Агентов
# =============================================================================
ENSEMBLE_SIZE = 3
MAX_RETRIES = 1
TEMPERATURE_RANGE = (0.1, 0.8)
MAX_CROSS_VALIDATION_ITERATIONS = 1
CROSS_VALIDATION_SIMILARITY_THRESHOLD = 0.4
MAX_TOTAL_ITERATIONS = 20

# =============================================================================
# ⭐ ПОРОГИ ДЛЯ МЕТРИК И РАБОТЫ АГЕНТОВ
# =============================================================================
MIN_WER_IMPROVEMENT = 0.1
MIN_LEV_SIMILARITY_IMPROVEMENT = 0.001
MIN_LLM_JUDGE_SCORE = 6
MIN_SEMANTIC_SIMILARITY = 0.5
MIN_METEOR_SCORE = 0.3
MIN_PERPLEXITY = 0.7

REFLECTION_ENABLED = True   # ВЫКЛЮЧИЛ ДЛЯ УСКОРЕНИЯ
REFLECTION_THRESHOLD_SUMSCORE = 0.55
REFLECTION_THRESHOLD_CORSCORE = 0.4

CROSS_VALIDATION_EARLY_STOP_SUMSCORE = 0.85
AGGREGATION_IMPROVEMENT_THRESHOLD = 0.90
LEV_WEIGHT = 8
PERPLEXITY_WEIGHT = 0.3

# =============================================================================
# ⭐ АДАПТИВНАЯ КОРРЕКЦИЯ
# =============================================================================
ADAPTIVE_CORRECTION_ENABLED = True
MAX_ADAPTIVE_ATTEMPTS = 1
MAX_LEV_RETRY_ATTEMPTS = 1
TEMP_RETRY_TEMPS = [0.1, 0.3]
# ✅ LEV_RETRY_TEMPS будет установлен ниже через os.getenv
DELTA_LEV_THRESHOLD = 0.02
USE_SAVED_PROMPTS = True
USE_FEW_SHOT_PROMPT = True
USE_CHAIN_OF_THOUGHT_PROMPT = True

# =============================================================================
# ⭐ ДИНАМИЧЕСКИЕ ТЕМПЕРАТУРЫ И SELF-CONSISTENCY
# =============================================================================
DYNAMIC_TEMPERATURES_ENABLED = True
SELF_CONSISTENCY_ENABLED = True   # ИСПРАВИЛ ДЛЯ УСКОРЕНИЯ
SELF_CONSISTENCY_EXTRA_COUNT = 1
ERROR_PROFILE_ENABLED = True

# =============================================================================
# ⭐ НАСТРОЙКИ СУММАРИЗАЦИИ
# =============================================================================
SUMMARY_ENSEMBLE_SIZE = 3
SUMMARY_TEMPERATURE_RANGE = (0.2, 1.0)
SUMMARY_ADAPTIVE_ENABLED = True
SUMMARY_MAX_RETRY_ATTEMPTS = 1
SUMMARY_RETRY_TEMPS = [0.4, 0.6, 1.0]
SUMMARY_USE_SAVED_PROMPTS = True
SUMMARY_USE_FEW_SHOT = True
SUMMARY_USE_CHAIN_OF_THOUGHT = True
SUMMARY_AGGREGATION_ROUNDS = 1
SUMMARY_SELF_CONSISTENCY_ENABLED = False     # ИСПРАВИЛ ДЛЯ УСКОРЕНИЯ
SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT = 1
SUMMARY_DYNAMIC_TEMPERATURES = True

# =============================================================================
# ⭐ МНОГОРАУНДОВАЯ АГРЕГАЦИЯ
# =============================================================================
MAX_AGGREGATION_ROUNDS = 1

# =============================================================================
# ⭐ БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ
# =============================================================================
HYPERPARAMETER_OPTIMIZATION_ENABLED = True
OPTIMIZATION_WINDOW_SIZE = 20

# =============================================================================
# ⭐ ДИНАМИЧЕСКИЙ FEW-SHOT
# =============================================================================
DYNAMIC_FEW_SHOT_ENABLED = True
MAX_FEW_SHOT_EXAMPLES = 3
FEW_SHOT_SIMILARITY_THRESHOLD = 0.5  # порог схожести для подбора примеров

# Для суммаризации
SUMMARY_DYNAMIC_FEW_SHOT_ENABLED = True
SUMMARY_MAX_FEW_SHOT_EXAMPLES = 3
SUMMARY_FEW_SHOT_LENGTH_RATIO = 0.2  # допустимое отклонение длины ±30%

# CoT для оценки
JUDGE_USE_COT = True  # использовать Chain-of-Thought при оценке суммаризации

# =============================================================================
# ⭐ ИСПОЛЬЗОВАНИЕ ЛУЧШИХ ПРОМПТОВ
# =============================================================================
USE_BEST_PROMPTS_FROM_MEMORY = True
BEST_PROMPTS_LIMIT = 5

# =============================================================================
# ⭐ ВЕБ-МОНИТОРИНГ
# =============================================================================
WEB_MONITOR_ENABLED = True
WEB_MONITOR_HOST = "127.0.0.1"
WEB_MONITOR_PORT = 5000
WEB_MONITOR_UPDATE_INTERVAL = 30

# =============================================================================
# ⭐ КЭШИРОВАНИЕ ПРОМПТОВ
# =============================================================================
PROMPT_CACHE_ENABLED = True
PROMPT_CACHE_MAX_SIZE = 500
PROMPT_CACHE_MIN_IMPROVEMENT = 0.1

# =============================================================================
# ⭐ МЕТРИКИ
# =============================================================================
BERTSCORE_ENABLED = True          # ✅ Включаем BertScore
SUMSCORE_ENABLED = True

# ✅ НОВЫЕ ВЕСА ДЛЯ SUMSCORE (с BertScore)
SUMSCORE_WEIGHTS = {
    "g_eval": 0.3,
    "llm_judge": 0.2,
    "meteor": 0.5,
    "bertscore": 0.4
}
SUMSCORE_DENOMINATOR = sum(SUMSCORE_WEIGHTS.values())  # = 1.4

# Для совместимости с main.py
ADAPTIVE_LEV_RETRY_ENABLED = True
XLS_UPDATE_AFTER_EACH_DOC = True

# =============================================================================
# 🔹 Препроцессинг Текста
# =============================================================================
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 10
MAX_SENTENCE_LENGTH = 500

# =============================================================================
# 🔹 Логирование
# =============================================================================
LOG_LEVEL = "INFO"
LOG_MAX_BYTES = 10485760
LOG_BACKUP_COUNT = 5

# =============================================================================
# 🔹 Память Агентов
# =============================================================================
MEMORY_ENABLED = True
MEMORY_MAX_HISTORY = 1000
MEMORY_AUTO_SAVE_INTERVAL = 10

# =============================================================================
# ⭐ САМОРЕФЛЕКСИЯ (REFLECTION)
# =============================================================================
REFLECTION_MIN_SCORE = 7               # Минимальная оценка для принятия без повторной генерации
REFLECTION_MAX_ATTEMPTS = 2            # Максимум попыток рефлексии для одного теста
REFLECTION_TEMPERATURE = 0.2           # Температура для запроса рефлексии
REFLECTION_USE_IN_PROMPT = True        # Использовать предложение рефлексии в следующем промпте

# =============================================================================
# 🔹 Модели для Задач
# =============================================================================
CORRECTION_MODEL = "google/gemma-3-12b"
SUMMARIZATION_MODEL = "google/gemma-3-12b"
JUDGE_MODEL = "google/gemma-3-12b"

# =============================================================================
# 📁 БАЗОВЫЕ НАСТРОЙКИ
# =============================================================================
BASE_DIR = Path(__file__).parent

# =============================================================================
# 🔧 LM STUDIO CONFIGURATION (переменные окружения)
# =============================================================================
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions")
LM_STUDIO_HOST = os.getenv("LM_STUDIO_HOST", LM_STUDIO_HOST)
LM_STUDIO_PORT = int(os.getenv("LM_STUDIO_PORT", LM_STUDIO_PORT))
MODEL_NAME = os.getenv("MODEL_NAME", MODEL_NAME)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", REQUEST_TIMEOUT))
LM_STUDIO_MIN_REQUEST_INTERVAL = float(os.getenv("LM_STUDIO_MIN_REQUEST_INTERVAL", LM_STUDIO_MIN_REQUEST_INTERVAL))
LM_STUDIO_MAX_TOKENS = int(os.getenv("LM_STUDIO_MAX_TOKENS", LM_STUDIO_MAX_TOKENS))

# =============================================================================
# 🤖 AGENT CONFIGURATION (переменные окружения)
# =============================================================================
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", ENSEMBLE_SIZE))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", MAX_RETRIES))
TEMPERATURE_RANGE = tuple(float(x) for x in os.getenv("TEMPERATURE_RANGE", f"{TEMPERATURE_RANGE[0]},{TEMPERATURE_RANGE[1]}").split(","))
MAX_CROSS_VALIDATION_ITERATIONS = int(os.getenv("MAX_CROSS_VALIDATION_ITERATIONS", MAX_CROSS_VALIDATION_ITERATIONS))
CROSS_VALIDATION_SIMILARITY_THRESHOLD = float(os.getenv("CROSS_VALIDATION_SIMILARITY_THRESHOLD", CROSS_VALIDATION_SIMILARITY_THRESHOLD))

ADAPTIVE_CORRECTION_ENABLED = os.getenv("ADAPTIVE_CORRECTION_ENABLED", str(ADAPTIVE_CORRECTION_ENABLED)).lower() in ('true', '1', 'yes')
MAX_ADAPTIVE_ATTEMPTS = int(os.getenv("MAX_ADAPTIVE_ATTEMPTS", MAX_ADAPTIVE_ATTEMPTS))
MAX_LEV_RETRY_ATTEMPTS = int(os.getenv("MAX_LEV_RETRY_ATTEMPTS", MAX_LEV_RETRY_ATTEMPTS))
TEMP_RETRY_TEMPS = [float(t) for t in os.getenv("TEMP_RETRY_TEMPS", ",".join(map(str, TEMP_RETRY_TEMPS))).split(",")]
LEV_RETRY_TEMPS = TEMP_RETRY_TEMPS
DELTA_LEV_THRESHOLD = float(os.getenv("DELTA_LEV_THRESHOLD", DELTA_LEV_THRESHOLD))
USE_SAVED_PROMPTS = os.getenv("USE_SAVED_PROMPTS", str(USE_SAVED_PROMPTS)).lower() in ('true', '1', 'yes')
USE_FEW_SHOT_PROMPT = os.getenv("USE_FEW_SHOT_PROMPT", str(USE_FEW_SHOT_PROMPT)).lower() in ('true', '1', 'yes')
USE_CHAIN_OF_THOUGHT_PROMPT = os.getenv("USE_CHAIN_OF_THOUGHT_PROMPT", str(USE_CHAIN_OF_THOUGHT_PROMPT)).lower() in ('true', '1', 'yes')

DYNAMIC_TEMPERATURES_ENABLED = os.getenv("DYNAMIC_TEMPERATURES_ENABLED", str(DYNAMIC_TEMPERATURES_ENABLED)).lower() in ('true', '1', 'yes')
SELF_CONSISTENCY_ENABLED = os.getenv("SELF_CONSISTENCY_ENABLED", str(SELF_CONSISTENCY_ENABLED)).lower() in ('true', '1', 'yes')
SELF_CONSISTENCY_EXTRA_COUNT = int(os.getenv("SELF_CONSISTENCY_EXTRA_COUNT", SELF_CONSISTENCY_EXTRA_COUNT))
ERROR_PROFILE_ENABLED = os.getenv("ERROR_PROFILE_ENABLED", str(ERROR_PROFILE_ENABLED)).lower() in ('true', '1', 'yes')

MAX_AGGREGATION_ROUNDS = int(os.getenv("MAX_AGGREGATION_ROUNDS", MAX_AGGREGATION_ROUNDS))
AGGREGATION_IMPROVEMENT_THRESHOLD = float(os.getenv("AGGREGATION_IMPROVEMENT_THRESHOLD", AGGREGATION_IMPROVEMENT_THRESHOLD))
CROSS_VALIDATION_EARLY_STOP_SUMSCORE = float(os.getenv("CROSS_VALIDATION_EARLY_STOP_SUMSCORE", CROSS_VALIDATION_EARLY_STOP_SUMSCORE))
HYPERPARAMETER_OPTIMIZATION_ENABLED = os.getenv("HYPERPARAMETER_OPTIMIZATION_ENABLED", str(HYPERPARAMETER_OPTIMIZATION_ENABLED)).lower() in ('true', '1', 'yes')
OPTIMIZATION_WINDOW_SIZE = int(os.getenv("OPTIMIZATION_WINDOW_SIZE", OPTIMIZATION_WINDOW_SIZE))
USE_BEST_PROMPTS_FROM_MEMORY = os.getenv("USE_BEST_PROMPTS_FROM_MEMORY", str(USE_BEST_PROMPTS_FROM_MEMORY)).lower() in ('true', '1', 'yes')
BEST_PROMPTS_LIMIT = int(os.getenv("BEST_PROMPTS_LIMIT", BEST_PROMPTS_LIMIT))

REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", str(REFLECTION_ENABLED)).lower() in ('true', '1', 'yes')
REFLECTION_THRESHOLD_SUMSCORE = float(os.getenv("REFLECTION_THRESHOLD_SUMSCORE", REFLECTION_THRESHOLD_SUMSCORE))
REFLECTION_THRESHOLD_CORSCORE = float(os.getenv("REFLECTION_THRESHOLD_CORSCORE", REFLECTION_THRESHOLD_CORSCORE))

# Параметры суммаризации
SUMMARY_ENSEMBLE_SIZE = int(os.getenv("SUMMARY_ENSEMBLE_SIZE", SUMMARY_ENSEMBLE_SIZE))
SUMMARY_TEMPERATURE_RANGE = tuple(float(x) for x in os.getenv("SUMMARY_TEMPERATURE_RANGE", f"{SUMMARY_TEMPERATURE_RANGE[0]},{SUMMARY_TEMPERATURE_RANGE[1]}").split(","))
SUMMARY_ADAPTIVE_ENABLED = os.getenv("SUMMARY_ADAPTIVE_ENABLED", str(SUMMARY_ADAPTIVE_ENABLED)).lower() in ('true', '1', 'yes')
SUMMARY_MAX_RETRY_ATTEMPTS = int(os.getenv("SUMMARY_MAX_RETRY_ATTEMPTS", SUMMARY_MAX_RETRY_ATTEMPTS))
SUMMARY_RETRY_TEMPS = [float(t) for t in os.getenv("SUMMARY_RETRY_TEMPS", ",".join(map(str, SUMMARY_RETRY_TEMPS))).split(",")]
SUMMARY_USE_SAVED_PROMPTS = os.getenv("SUMMARY_USE_SAVED_PROMPTS", str(SUMMARY_USE_SAVED_PROMPTS)).lower() in ('true', '1', 'yes')
SUMMARY_USE_FEW_SHOT = os.getenv("SUMMARY_USE_FEW_SHOT", str(SUMMARY_USE_FEW_SHOT)).lower() in ('true', '1', 'yes')
SUMMARY_USE_CHAIN_OF_THOUGHT = os.getenv("SUMMARY_USE_CHAIN_OF_THOUGHT", str(SUMMARY_USE_CHAIN_OF_THOUGHT)).lower() in ('true', '1', 'yes')
SUMMARY_AGGREGATION_ROUNDS = int(os.getenv("SUMMARY_AGGREGATION_ROUNDS", SUMMARY_AGGREGATION_ROUNDS))
SUMMARY_SELF_CONSISTENCY_ENABLED = os.getenv("SUMMARY_SELF_CONSISTENCY_ENABLED", str(SUMMARY_SELF_CONSISTENCY_ENABLED)).lower() in ('true', '1', 'yes')
SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT = int(os.getenv("SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT", SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT))
SUMMARY_DYNAMIC_TEMPERATURES = os.getenv("SUMMARY_DYNAMIC_TEMPERATURES", str(SUMMARY_DYNAMIC_TEMPERATURES)).lower() in ('true', '1', 'yes')

# =============================================================================
# 📄 FILE PROCESSING CONFIGURATION
# =============================================================================
SKIP_PROCESSED = int(os.getenv("SKIP_PROCESSED", SKIP_PROCESSED))
XLS_UPDATE_AFTER_EACH_DOC = os.getenv("XLS_UPDATE_AFTER_EACH_DOC", str(XLS_UPDATE_AFTER_EACH_DOC)).lower() in ('true', '1', 'yes')

# =============================================================================
# 📊 METRICS CONFIGURATION
# =============================================================================
METRICS_CONFIG = {
    "wer": {"enabled": True},
    "levenshtein": {"enabled": True},
    "g_eval": {"enabled": True, "min_score": MIN_LLM_JUDGE_SCORE},
    "p_umfd": {"enabled": True},
    "meteor": {"enabled": True, "min_score": MIN_METEOR_SCORE},
    "perplexity": {"enabled": True, "max_threshold": MIN_PERPLEXITY},
    "bertscore": {"enabled": BERTSCORE_ENABLED}
}

RETRY_THRESHOLDS = {
    "min_wer_improvement": float(os.getenv("MIN_WER_IMPROVEMENT", MIN_WER_IMPROVEMENT)),
    "min_lev_similarity_improvement": float(os.getenv("MIN_LEV_SIMILARITY_IMPROVEMENT", MIN_LEV_SIMILARITY_IMPROVEMENT)),
    "min_llm_judge_score": int(os.getenv("MIN_LLM_JUDGE_SCORE", MIN_LLM_JUDGE_SCORE)),
    "min_semantic_similarity": float(os.getenv("MIN_SEMANTIC_SIMILARITY", MIN_SEMANTIC_SIMILARITY)),
    "min_meteor_score": float(os.getenv("MIN_METEOR_SCORE", MIN_METEOR_SCORE)),
    "max_perplexity": float(os.getenv("MIN_PERPLEXITY", MIN_PERPLEXITY))
}

# =============================================================================
# 📝 TEXT PREPROCESSING CONFIGURATION
# =============================================================================
TEXT_PREPROCESSING = {
    "max_text_length": int(os.getenv("MAX_TEXT_LENGTH", MAX_TEXT_LENGTH)),
    "min_text_length": int(os.getenv("MIN_TEXT_LENGTH", MIN_TEXT_LENGTH)),
    "max_sentence_length": int(os.getenv("MAX_SENTENCE_LENGTH", MAX_SENTENCE_LENGTH)),
    "normalize_whitespace": True,
    "sanitize_special_chars": True,
    "handle_mixed_languages": True,
    "auto_truncate": True
}

# =============================================================================
# 📂 DIRECTORY CONFIGURATION
# =============================================================================
DIRS = {
    "data": BASE_DIR / "data",
    "incorrect_texts": BASE_DIR / "data" / DATA_LANG_DIR / "Incorrect_texts",
    "etalon_texts": BASE_DIR / "data" / DATA_LANG_DIR / "etalon_texts",
    "etalon_summaries": BASE_DIR / "data" / DATA_LANG_DIR / "etalon_summaries",
    "test_cases": BASE_DIR / "data" / DATA_LANG_DIR / "test_cases",
    "correction": BASE_DIR / "data" / DATA_LANG_DIR / "correction",
    "summary": BASE_DIR / "data" / DATA_LANG_DIR / "summary",
    "correction_metrics": BASE_DIR / "data" / DATA_LANG_DIR / "correction_metrics",
    "summary_metrics": BASE_DIR / "data" / DATA_LANG_DIR / "summary_metrics",
    "memory": BASE_DIR / "data" / DATA_LANG_DIR / "memory",
    "metrics": BASE_DIR / "data" / DATA_LANG_DIR / "metrics",  # ✅ Папка для XLS метрик
    "logs": BASE_DIR / "logs",
    "agent_logs": BASE_DIR / "logs" / "agent_logs",
    "documentation": BASE_DIR / "documentation"
}

# =============================================================================
# 📝 LOGGING CONFIGURATION
# =============================================================================
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", LOG_LEVEL),
    "format": "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "system.log",
    "max_bytes": int(os.getenv("LOG_MAX_BYTES", LOG_MAX_BYTES)),
    "backup_count": int(os.getenv("LOG_BACKUP_COUNT", LOG_BACKUP_COUNT)),
    "console_output": True,
    "file_output": True,
    "agent_logs": True
}

# =============================================================================
# 📤 FILE OUTPUT CONFIGURATION
# =============================================================================
FILE_OUTPUT_CONFIG = {
    "encoding": "cp1251",  # windows-1251 для совместимости
    "sentence_per_line": True,
    "include_prompts": True,
    "include_ensemble_outputs": False,
    "filename_format": "{test_id}.txt"
}

# =============================================================================
# 🧠 MEMORY CONFIGURATION
# =============================================================================
MEMORY_CONFIG = {
    "enabled": MEMORY_ENABLED,
    "max_history_size": int(os.getenv("MEMORY_MAX_HISTORY", MEMORY_MAX_HISTORY)),
    "auto_save": True,
    "auto_save_interval": int(os.getenv("MEMORY_AUTO_SAVE_INTERVAL", MEMORY_AUTO_SAVE_INTERVAL)),
    "files": {
        "history": "correction_history.json",
        "errors": "common_errors.json",
        "patterns": "success_patterns.json",
        "stats": "domain_stats.json",
        "prompts": "best_prompts.json"
    }
}

# =============================================================================
# 🎯 MODEL SELECTION CONFIGURATION
# =============================================================================
MODEL_SELECTION = {
    "correction": os.getenv("CORRECTION_MODEL", CORRECTION_MODEL) or MODEL_NAME,
    "summarization": os.getenv("SUMMARIZATION_MODEL", SUMMARIZATION_MODEL) or MODEL_NAME,
    "judge": os.getenv("JUDGE_MODEL", JUDGE_MODEL) or MODEL_NAME,
    "parameters": {
        "correction": {"temperature_range": (0.1, 0.8), "max_tokens": 8192},
        "summarization": {"temperature_range": (0.4, 0.8), "max_tokens": 4096},
        "judge": {"temperature_range": (0.1, 0.3), "max_tokens": 4096}
    }
}

# =============================================================================
# 🤖 MODEL CACHE CONFIGURATION
# =============================================================================
SENTENCE_TRANSFORMERS_HOME = os.getenv("SENTENCE_TRANSFORMERS_HOME", str(Path.home() / ".cache" / "torch" / "sentence_transformers"))
HF_HOME = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# =============================================================================
# ⚙️ HELPER FUNCTIONS
# =============================================================================
def create_directories():
    for dir_path in DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)

def get_config_summary() -> dict:
    return {
        "lm_studio": {"url": LM_STUDIO_URL, "host": LM_STUDIO_HOST, "port": LM_STUDIO_PORT, "model": MODEL_NAME, "timeout": REQUEST_TIMEOUT, "min_interval": LM_STUDIO_MIN_REQUEST_INTERVAL},
        "agents": {"ensemble_size": ENSEMBLE_SIZE, "max_retries": MAX_RETRIES, "temperature_range": TEMPERATURE_RANGE, "cross_validation": MAX_CROSS_VALIDATION_ITERATIONS},
        "adaptive_correction": {"enabled": ADAPTIVE_CORRECTION_ENABLED, "max_attempts": MAX_ADAPTIVE_ATTEMPTS, "max_lev_retry_attempts": MAX_LEV_RETRY_ATTEMPTS, "temp_retry_temps": TEMP_RETRY_TEMPS, "delta_lev_threshold": DELTA_LEV_THRESHOLD, "use_saved_prompts": USE_SAVED_PROMPTS, "use_few_shot": USE_FEW_SHOT_PROMPT, "use_cot": USE_CHAIN_OF_THOUGHT_PROMPT},
        "dynamic_temperatures": {"enabled": DYNAMIC_TEMPERATURES_ENABLED, "self_consistency": SELF_CONSISTENCY_ENABLED, "self_consistency_extra": SELF_CONSISTENCY_EXTRA_COUNT},
        "aggregation": {"max_rounds": MAX_AGGREGATION_ROUNDS, "improvement_threshold": AGGREGATION_IMPROVEMENT_THRESHOLD},
        "cross_validation": {"early_stop_sumscore": CROSS_VALIDATION_EARLY_STOP_SUMSCORE},
        "optimization": {"enabled": HYPERPARAMETER_OPTIMIZATION_ENABLED, "window_size": OPTIMIZATION_WINDOW_SIZE},
        "best_prompts": {"use_from_memory": USE_BEST_PROMPTS_FROM_MEMORY, "limit": BEST_PROMPTS_LIMIT},
        "reflection": {"enabled": REFLECTION_ENABLED, "threshold_sumscore": REFLECTION_THRESHOLD_SUMSCORE, "threshold_corscore": REFLECTION_THRESHOLD_CORSCORE},
        "summarization": {
            "ensemble_size": SUMMARY_ENSEMBLE_SIZE,
            "temperature_range": SUMMARY_TEMPERATURE_RANGE,
            "adaptive": SUMMARY_ADAPTIVE_ENABLED,
            "max_retry_attempts": SUMMARY_MAX_RETRY_ATTEMPTS,
            "retry_temps": SUMMARY_RETRY_TEMPS,
            "use_saved_prompts": SUMMARY_USE_SAVED_PROMPTS,
            "use_few_shot": SUMMARY_USE_FEW_SHOT,
            "use_cot": SUMMARY_USE_CHAIN_OF_THOUGHT,
            "aggregation_rounds": SUMMARY_AGGREGATION_ROUNDS,
            "self_consistency_enabled": SUMMARY_SELF_CONSISTENCY_ENABLED,
            "self_consistency_extra": SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT,
            "dynamic_temperatures": SUMMARY_DYNAMIC_TEMPERATURES
        },
        "web_monitor": {"enabled": WEB_MONITOR_ENABLED, "host": WEB_MONITOR_HOST, "port": WEB_MONITOR_PORT, "update_interval": WEB_MONITOR_UPDATE_INTERVAL},
        "prompt_cache": {"enabled": PROMPT_CACHE_ENABLED, "max_size": PROMPT_CACHE_MAX_SIZE, "min_improvement": PROMPT_CACHE_MIN_IMPROVEMENT},
        "parallel_processing": {"enabled": PARALLEL_TESTS_ENABLED, "max_workers": PARALLEL_TESTS_MAX_WORKERS},
        "metrics": {"bertscore_enabled": BERTSCORE_ENABLED, "sumscore_enabled": SUMSCORE_ENABLED, "sumscore_weights": SUMSCORE_WEIGHTS},
        "file_processing": {"skip_processed": SKIP_PROCESSED, "xls_update_after_each": XLS_UPDATE_AFTER_EACH_DOC},
        "metrics_config": {"enabled": [k for k, v in METRICS_CONFIG.items() if v.get("enabled")], "thresholds": RETRY_THRESHOLDS},
        "preprocessing": {"max_length": TEXT_PREPROCESSING["max_text_length"], "min_length": TEXT_PREPROCESSING["min_text_length"]},
        "memory": {"enabled": MEMORY_CONFIG["enabled"], "max_history": MEMORY_CONFIG["max_history_size"]},
        "logging": {"level": LOG_CONFIG["level"]},
        "model_cache": {"sentence_transformers_home": SENTENCE_TRANSFORMERS_HOME, "hf_home": HF_HOME}
    }

def validate_config() -> list:
    warnings_list = []
    if not LM_STUDIO_URL.startswith("http"):
        warnings_list.append(f"LM_STUDIO_URL должен начинаться с http: {LM_STUDIO_URL}")
    if TEMPERATURE_RANGE[0] < 0 or TEMPERATURE_RANGE[1] > 1:
        warnings_list.append(f"TEMPERATURE_RANGE должен быть в [0, 1]: {TEMPERATURE_RANGE}")
    if not (0 <= CROSS_VALIDATION_SIMILARITY_THRESHOLD <= 1):
        warnings_list.append(f"CROSS_VALIDATION_SIMILARITY_THRESHOLD должен быть в [0, 1]")
    if not (0 <= MIN_METEOR_SCORE <= 1):
        warnings_list.append(f"MIN_METEOR_SCORE должен быть в [0, 1]: {MIN_METEOR_SCORE}")
    if not (0 <= MIN_PERPLEXITY <= 1):
        warnings_list.append(f"MIN_PERPLEXITY должен быть в [0, 1]: {MIN_PERPLEXITY}")
    if SKIP_PROCESSED not in [0, 1]:
        warnings_list.append(f"SKIP_PROCESSED должен быть 0 или 1: {SKIP_PROCESSED}")
    if LOG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        warnings_list.append(f"LOG_LEVEL должен быть DEBUG, INFO, WARNING, ERROR или CRITICAL: {LOG_LEVEL}")
    if WEB_MONITOR_PORT < 1024 or WEB_MONITOR_PORT > 65535:
        warnings_list.append(f"WEB_MONITOR_PORT должен быть 1024-65535: {WEB_MONITOR_PORT}")
    if PROMPT_CACHE_MAX_SIZE < 10 or PROMPT_CACHE_MAX_SIZE > 1000:
        warnings_list.append(f"PROMPT_CACHE_MAX_SIZE должен быть 10-1000: {PROMPT_CACHE_MAX_SIZE}")
    if PROMPT_CACHE_MIN_IMPROVEMENT < 0 or PROMPT_CACHE_MIN_IMPROVEMENT > 1:
        warnings_list.append(f"PROMPT_CACHE_MIN_IMPROVEMENT должен быть в [0, 1]: {PROMPT_CACHE_MIN_IMPROVEMENT}")
    if PARALLEL_TESTS_MAX_WORKERS < 1 or PARALLEL_TESTS_MAX_WORKERS > 4:
        warnings_list.append(f"PARALLEL_TESTS_MAX_WORKERS должен быть 1-4: {PARALLEL_TESTS_MAX_WORKERS}")
    total_weight = sum(SUMSCORE_WEIGHTS.values())
    if abs(total_weight - 1.4) > 0.01:
        warnings_list.append(f"Сумма весов SumScore должна быть 1.4 (сейчас: {total_weight})")
    if MAX_ADAPTIVE_ATTEMPTS < 1 or MAX_ADAPTIVE_ATTEMPTS > 5:
        warnings_list.append(f"MAX_ADAPTIVE_ATTEMPTS должен быть 1-5: {MAX_ADAPTIVE_ATTEMPTS}")
    if MAX_LEV_RETRY_ATTEMPTS < 1 or MAX_LEV_RETRY_ATTEMPTS > 5:
        warnings_list.append(f"MAX_LEV_RETRY_ATTEMPTS должен быть 1-5: {MAX_LEV_RETRY_ATTEMPTS}")
    if DELTA_LEV_THRESHOLD < 0:
        warnings_list.append(f"DELTA_LEV_THRESHOLD должен быть >= 0: {DELTA_LEV_THRESHOLD}")
    for temp in TEMP_RETRY_TEMPS:
        if temp < 0 or temp > 1:
            warnings_list.append(f"Температура в TEMP_RETRY_TEMPS должна быть в [0, 1]: {temp}")
    for name, path in DIRS.items():
        if not path.exists():
            warnings_list.append(f"Директория не существует: {name} = {path}")
    return warnings_list

def should_skip_file(test_id: str, task_type: str) -> bool:
    if SKIP_PROCESSED == 0:
        return False
    
    # Для combined: проверяем И correction И summary - ОБЕ должны быть полностью готовы
    if task_type == "combined":
        corr_exists = (DIRS["correction"] / f"{test_id}.txt").exists()
        corr_metrics_exists = (DIRS["correction_metrics"] / f"{test_id}.txt").exists()
        sum_exists = (DIRS["summary"] / f"{test_id}.txt").exists()
        sum_metrics_exists = (DIRS["summary_metrics"] / f"{test_id}.txt").exists()
        # Пропускаем ТОЛЬКО если ВСЕ 4 файла существуют
        if corr_exists and corr_metrics_exists and sum_exists and sum_metrics_exists:
            return True
        return False  # Если хоть чего-то нет - обрабатываем
    
    # Для только коррекции
    if task_type == "correction":
        corr_exists = (DIRS["correction"] / f"{test_id}.txt").exists()
        corr_metrics_exists = (DIRS["correction_metrics"] / f"{test_id}.txt").exists()
        if corr_exists and corr_metrics_exists:
            return True
    
    # Для только суммаризации
    if task_type == "summary":
        sum_exists = (DIRS["summary"] / f"{test_id}.txt").exists()
        sum_metrics_exists = (DIRS["summary_metrics"] / f"{test_id}.txt").exists()
        if sum_exists and sum_metrics_exists:
            return True
    
    return False

def get_processed_files(directory: Path) -> set:
    if not directory.exists():
        return set()
    return {f.stem for f in directory.glob("*.txt")}

def calculate_sumscore(g_eval: float, llm_judge: float, meteor: float, bertscore: float = 0.0) -> float:
    """Вычисление SumScore с новыми весами (0.3, 0.3, 0.5, 0.3) и нормализацией"""
    llm_judge_normalized = llm_judge / 10.0
    numerator = (g_eval * SUMSCORE_WEIGHTS["g_eval"] +
                 llm_judge_normalized * SUMSCORE_WEIGHTS["llm_judge"] +
                 meteor * SUMSCORE_WEIGHTS["meteor"] +
                 bertscore * SUMSCORE_WEIGHTS["bertscore"])
    denominator = SUMSCORE_DENOMINATOR
    sumscore = numerator / denominator
    return max(0.0, min(1.0, sumscore))

def get_sumscore_assessment(sumscore: float) -> str:
    if sumscore >= 0.85:
        return "⭐⭐⭐⭐⭐ ОТЛИЧНО"
    elif sumscore >= 0.70:
        return "⭐⭐⭐⭐ ХОРОШО"
    elif sumscore >= 0.60:
        return "⭐⭐⭐ УДОВЛЕТВОРИТЕЛЬНО"
    elif sumscore >= 0.50:
        return "⭐⭐ ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
    else:
        return "⭐ НЕУДОВЛЕТВОРИТЕЛЬНО"

def calculate_cor_score(delta_wer: float, delta_lev: float, perplexity: float) -> float:
    """
    ✅ НОВАЯ ФОРМУЛА CorScore:
    delta_WER + 6*delta_Lev + (1 - perplexity/100)*0.5
    """
    perplexity_term = (1.0 - perplexity / 100.0) * PERPLEXITY_WEIGHT
    return delta_wer + LEV_WEIGHT * delta_lev + perplexity_term

def print_config_quick_reference():
    print("\n" + "=" * 80)
    print("📋 БЫСТРАЯ СПРАВКА ПО НАСТРОЙКАМ CorSumAgentsAI v5.7.0")
    print("=" * 80)
    print(f"""
🔹 LM Studio: {LM_STUDIO_HOST}:{LM_STUDIO_PORT} | Модель: {MODEL_NAME}
🔹 Адаптивная коррекция: {ADAPTIVE_CORRECTION_ENABLED} | Попыток: {MAX_ADAPTIVE_ATTEMPTS}
🔹 Динамические температуры: {DYNAMIC_TEMPERATURES_ENABLED} | Self-consistency: {SELF_CONSISTENCY_ENABLED}
🔹 Многораундовая агрегация: {MAX_AGGREGATION_ROUNDS} раундов
🔹 Кросс-валидация: {MAX_CROSS_VALIDATION_ITERATIONS} итераций, ранний выход при SumScore ≥ {CROSS_VALIDATION_EARLY_STOP_SUMSCORE}
🔹 Байесовская оптимизация: {HYPERPARAMETER_OPTIMIZATION_ENABLED}
🔹 Использование лучших промптов из памяти: {USE_BEST_PROMPTS_FROM_MEMORY}
🔹 Рефлексия: {REFLECTION_ENABLED} (пороги SumScore={REFLECTION_THRESHOLD_SUMSCORE}, CorScore={REFLECTION_THRESHOLD_CORSCORE})
🔹 Суммаризация: ансамбль {SUMMARY_ENSEMBLE_SIZE}, темп.диапазон {SUMMARY_TEMPERATURE_RANGE}, адаптив: {SUMMARY_ADAPTIVE_ENABLED}
🔹 Веса SumScore: G-Eval={SUMSCORE_WEIGHTS['g_eval']}, LLM-Judge={SUMSCORE_WEIGHTS['llm_judge']}, METEOR={SUMSCORE_WEIGHTS['meteor']}, BertScore={SUMSCORE_WEIGHTS['bertscore']}
🔹 Формула CorScore: ΔWER + Lev_Weight*ΔLev + (1 - Perpl/100)*Perplexity_Weight
""")
    print("=" * 80 + "\n")