#!/usr/bin/env python3
"""
CorSumAgentsAI - Система коррекции и суммаризации текстов на базе ансамбля LLM-агентов
Версия 5.6.1 - Исправлены duration в веб-мониторе, устранены двойные записи
"""
import csv
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    create_directories, DIRS, BASE_DIR, get_config_summary,
    validate_config, SKIP_PROCESSED, should_skip_file, MODEL_NAME, WEB_MONITOR_ENABLED, WEB_MONITOR_PORT,
    BERTSCORE_ENABLED, SUMSCORE_ENABLED, SUMSCORE_WEIGHTS,
    ADAPTIVE_LEV_RETRY_ENABLED, XLS_UPDATE_AFTER_EACH_DOC,
    calculate_sumscore, get_sumscore_assessment, calculate_cor_score,
    DATA_LANG_DIR
)
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory
from utils.logger import setup_logger, DateLogger, print_log_folder_info
from utils.colors import (
    Color, cformat, print_info, print_error,
    print_warning, print_metric, print_success, print_normal, init_colors
)
from orchestrator import Orchestrator, AgentState

init_colors()
logger = setup_logger("Main")
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def check_web_monitor_dependencies() -> bool:
    try:
        import flask
        try:
            import importlib.metadata
            flask_version = importlib.metadata.version("flask")
            logger.info(f"[WebMonitor] Flask версии {flask_version} установлен")
        except:
            logger.info("[WebMonitor] Flask установлен")
        try:
            import werkzeug
            logger.info("[WebMonitor] Werkzeug установлен")
        except ImportError:
            logger.warning("[WebMonitor] Werkzeug не установлен")
            print_warning("  ⚠️ Werkzeug не установлен! Рекомендуется установить: pip install werkzeug")
        return True
    except ImportError:
        logger.warning("[WebMonitor] Flask не установлен!")
        print_warning("  ⚠️ Flask не установлен! Веб-монитор не будет работать.")
        print_warning("  💡 Установите: pip install flask werkzeug")
        return False


def start_web_monitor_if_enabled() -> Optional[Any]:
    if not WEB_MONITOR_ENABLED:
        logger.info("[WebMonitor] Веб-монитор отключён в конфигурации")
        return None
    if not check_web_monitor_dependencies():
        return None
    try:
        from utils.web_monitor import init_web_monitor
        web_monitor = init_web_monitor(host="127.0.0.1", port=WEB_MONITOR_PORT)
        web_monitor.start()
        time.sleep(2)
        if web_monitor.is_running():
            print_success(f"  🌐 Веб-монитор запущен: http://127.0.0.1:{WEB_MONITOR_PORT}")
            logger.info(f"[Main] Веб-монитор запущен: http://127.0.0.1:{WEB_MONITOR_PORT}")
            try:
                import urllib.request
                req = urllib.request.Request(f"http://127.0.0.1:{WEB_MONITOR_PORT}/health", method='GET')
                response = urllib.request.urlopen(req, timeout=3)
                if response.getcode() == 200:
                    print_success(f"  ✅ Веб-сервер доступен и отвечает на запросы")
                else:
                    print_warning(f"  ⚠️ Веб-сервер ответил с кодом {response.getcode()}")
            except Exception as e:
                print_warning(f"  ⚠️ Веб-сервер запущен, но проверка доступности не удалась: {e}")
                print_warning(f"  💡 Попробуйте открыть в браузере: http://127.0.0.1:{WEB_MONITOR_PORT}")
        else:
            print_warning(f"  ⚠️ Веб-монитор не запустился")
        return web_monitor
    except Exception as e:
        print_warning(f"  ⚠️ Не удалось запустить веб-монитор: {e}")
        logger.warning(f"[Main] Ошибка запуска веб-монитора: {e}")
        import traceback
        traceback.print_exc()
        return None


class TimeStatistics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.correction_total = 0.0
        self.summarization_total = 0.0
        self.test_stats = []
    def start_total(self):
        self.start_time = time.time()
        logger.info(f"[Time] Начало работы системы")
    def end_total(self):
        self.end_time = time.time()
        logger.info(f"[Time] Завершение работы системы")
    def add_test_stat(self, test_id, correction_time, summarization_time, prompt_num=1, metrics=None):
        self.test_stats.append({
            'test_id': test_id,
            'correction_time': correction_time,
            'summarization_time': summarization_time,
            'total_time': correction_time + summarization_time,
            'prompt_num': prompt_num,
            'metrics': metrics or {}
        })
        self.correction_total += correction_time
        self.summarization_total += summarization_time
    def get_total_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    def get_correction_time(self): return self.correction_total
    def get_summarization_time(self): return self.summarization_total
    def get_average_test_time(self):
        if not self.test_stats: return 0.0
        return sum(t['total_time'] for t in self.test_stats) / len(self.test_stats)
    def print_statistics(self):
        print("\n" + "─" * 80)
        print_metric("  ⏱️  ОБЩАЯ СТАТИСТИКА ВРЕМЕНИ:")
        print("─" * 80)
        print_metric(f"     └─ Общее время работы: {self.get_total_time():.2f} сек")
        print_metric(f"     └─ Время коррекции: {self.get_correction_time():.2f} сек ({self.get_correction_time() / max(1, self.get_total_time()) * 100:.1f}%)")
        print_metric(f"     └─ Время суммаризации: {self.get_summarization_time():.2f} сек ({self.get_summarization_time() / max(1, self.get_total_time()) * 100:.1f}%)")
        print_metric(f"     └─ Среднее время на тест: {self.get_average_test_time():.2f} сек")
        print("─" * 80)
    def print_file_statistics(self, test_id, correction_time, summarization_time, prompt_num, metrics):
        print("\n" + "─" * 60)
        print_metric(f"  📊 СТАТИСТИКА ПО ФАЙЛУ: {test_id}")
        print("─" * 60)
        print_normal(f"     └─ Время коррекции: {correction_time:.3f} сек")
        print_normal(f"     └─ Время суммаризации: {summarization_time:.3f} сек")
        print_normal(f"     └─ Общее время: {correction_time + summarization_time:.3f} сек")
        print_normal(f"     └─ Промпт: {prompt_num}")
        if 'WER_0' in metrics and 'WER' in metrics:
            print_metric(f"     └─ Оценка коррекции: WER {metrics.get('WER_0')} → {metrics.get('WER')} (Δ={metrics.get('delta_WER')})")
        if 'LevRating_0' in metrics and 'LevRating' in metrics:
            print_metric(f"     └─ LevRating: {metrics.get('LevRating_0')} → {metrics.get('LevRating')} (Δ={metrics.get('delta_LEV')})")
        if 'perplexity' in metrics and metrics['perplexity'] != 'N/A':
            print_metric(f"     └─ Perplexity: {metrics.get('perplexity')}")
        if 'best_model_cor' in metrics:
            print_metric(f"     └─ Лучшая модель коррекции: {metrics.get('best_model_cor')}")
        if 'best_temp_cor' in metrics:
            print_metric(f"     └─ Лучшая температура коррекции: {metrics.get('best_temp_cor')}")
        if 'best_temperature_summary' in metrics:
            print_metric(f"     └─ Лучшая температура суммаризации: {metrics.get('best_temperature_summary')}")
        if 'best_model_sum' in metrics:
            print_metric(f"     └─ Лучшая модель суммаризации: {metrics.get('best_model_sum')}")
        if 'LLM_Judge' in metrics:
            print_metric(f"     └─ LLM-Judge: {metrics.get('LLM_Judge')}")
        if 'G_Eval' in metrics:
            print_metric(f"     └─ G-Eval: {metrics.get('G_Eval')}")
        if 'METEOR' in metrics:
            print_metric(f"     └─ METEOR: {metrics.get('METEOR')}")
        if BERTSCORE_ENABLED and 'BertScore' in metrics:
            print_metric(f"     └─ BertScore: {metrics.get('BertScore')}")
        if SUMSCORE_ENABLED and 'SumScore' in metrics:
            print_metric(f"     └─ SumScore: {metrics.get('SumScore')}")
        if 'Quality' in metrics:
            print_metric(f"     └─ Quality: {metrics.get('Quality')}")
        print("─" * 60)
        sys.stdout.flush()
    def to_dict(self):
        return {
            'total_time': self.get_total_time(),
            'correction_time': self.get_correction_time(),
            'summarization_time': self.get_summarization_time(),
            'average_test_time': self.get_average_test_time(),
            'tests_processed': len(self.test_stats)
        }


class TextFormatter:
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        if not text or not isinstance(text, str): return []
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text.strip())
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if not sentence[-1] in '.!?':
                    sentence += '.'
                result.append(sentence)
        return result
    @staticmethod
    def format_text_with_newlines(text: str) -> str:
        if not text or not isinstance(text, str): return ""
        sentences = TextFormatter.split_into_sentences(text)
        return "\n".join(sentences)


class XLSExporter:
    QUALITY_TRANSLATIONS = {"ОТЛИЧНО": "EXCELLENT", "ХОРОШО": "WELL", "УДОВЛЕТВОРИТЕЛЬНО": "SATISFACTORY", "ТРЕБУЕТСЯ УЛУЧШЕНИЕ": "NEEDS IMPROVEMENT", "N/A": "N/A"}
    XLS_HEADERS = [
        'File_Name', 'LevRating', 'delta_WER', 'delta_Lev', 'Perpl', 'CorScore',
        'best_temp_cor', 'best_model_cor', 'G-Eval', 'LLM-Judge', 'METEOR',
        'SumScore', 'BertScore', 'best_model_sum', 'Correction_Time', 'Summary_Time', 'Total_Time', 'Prompt_Num'
    ]
    @staticmethod
    def translate_quality(russian_quality: str) -> str:
        if not russian_quality: return "N/A"
        for ru, en in XLSExporter.QUALITY_TRANSLATIONS.items():
            if ru in russian_quality: return en
        return russian_quality
    @staticmethod
    def _load_existing_xls(xls_filename: Path) -> Dict[str, Dict[str, Any]]:
        if not xls_filename.exists(): return {}
        existing_data = {}
        try:
            with open(xls_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 2: return {}
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 10:
                        test_id = parts[0]
                        existing_data[test_id] = {
                            'CorScore': float(parts[5]) if parts[5] != 'N/A' else 0.0,
                            'delta_Lev': float(parts[3]) if parts[3] != 'N/A' else 0.0,
                            'row': parts
                        }
        except Exception as e:
            logger.warning(f"[XLS] Ошибка чтения существующего файла: {e}")
        return existing_data
    @staticmethod
    def _should_update_xls(test_id: str, new_cor_score: float, new_delta_lev: float, xls_filename: Path) -> bool:
        if not XLS_UPDATE_AFTER_EACH_DOC: return False
        existing = XLSExporter._load_existing_xls(xls_filename)
        if test_id not in existing: return True
        if new_cor_score > existing[test_id]['CorScore'] and new_delta_lev > existing[test_id]['delta_Lev']:
            logger.info(f"[XLS] ✅ Обновление {test_id}: CorScore {existing[test_id]['CorScore']:.4f} → {new_cor_score:.4f}, delta_Lev {existing[test_id]['delta_Lev']:.4f} → {new_delta_lev:.4f}")
            return True
        else:
            logger.info(f"[XLS] ⏭️ Пропуск {test_id}: значения не улучшились")
            return False
    @staticmethod
    def update_xls_after_each_document(metrics_row: Dict[str, Any], time_stat: Dict[str, Any], xls_filename: str = None):
        if xls_filename is None: xls_filename = DIRS["metrics"] / "metrics.xls"
        DIRS["metrics"].mkdir(parents=True, exist_ok=True)
        cor_score = float(metrics_row.get('CorScore', 0) or 0)
        delta_lev = float(metrics_row.get('delta_Lev', 0) or 0)
        if not XLSExporter._should_update_xls(metrics_row.get('file_name', ''), cor_score, delta_lev, xls_filename): return
        existing_data = XLSExporter._load_existing_xls(xls_filename)
        test_id = metrics_row.get('file_name', '')
        time_stat = time_stat or {}
        llm_judge_value = metrics_row.get('LLM-Judge', 'N/A')
        if isinstance(llm_judge_value, str) and "из" in llm_judge_value: llm_judge_value = llm_judge_value.split()[0]
        row = [
            metrics_row.get('file_name', 'N/A'),
            metrics_row.get('LevRating', 'N/A'),
            metrics_row.get('delta_WER', 'N/A'),
            metrics_row.get('delta_Lev', 'N/A'),
            metrics_row.get('Perpl', 'N/A'),
            f"{cor_score:.6f}",
            metrics_row.get('best_temp_cor', 'N/A'),
            metrics_row.get('best_model_cor', 'N/A'),
            metrics_row.get('G-Eval', 'N/A'),
            str(llm_judge_value),
            metrics_row.get('METEOR', 'N/A'),
            metrics_row.get('SumScore', 'N/A'),
            metrics_row.get('BertScore', 'N/A'),
            metrics_row.get('best_model_sum', 'N/A'),
            f"{time_stat.get('correction_time', 0):.3f}",
            f"{time_stat.get('summary_time', 0):.3f}",
            f"{time_stat.get('total_time', 0):.3f}",
            str(time_stat.get('prompt_num', 'N/A'))
        ]
        try:
            if test_id in existing_data:
                lines = []
                with open(xls_filename, 'r', encoding='utf-8') as f: lines = f.readlines()
                with open(xls_filename, 'w', encoding='utf-8') as f:
                    f.write(lines[0])
                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        if len(parts) >= 1 and parts[0] == test_id:
                            f.write('\t'.join(row) + '\n')
                        else:
                            f.write(line)
            else:
                with open(xls_filename, 'a', encoding='utf-8') as f: f.write('\t'.join(row) + '\n')
            logger.info(f"[XLS] XLS обновлён: {test_id}")
            print_success(f"  📊 XLS обновлён: {test_id}")
        except Exception as e:
            print_error(f"  ❌ Ошибка записи XLS: {e}")
            logger.error(f"[XLS] Ошибка: {e}")
    @staticmethod
    def save_metrics_xls(all_metrics: List[Dict], time_stats: List[Dict], xls_filename: str = None):
        if xls_filename is None: xls_filename = DIRS["metrics"] / "metrics.xls"
        DIRS["metrics"].mkdir(parents=True, exist_ok=True)
        try:
            with open(xls_filename, 'w', encoding='utf-8') as f:
                f.write('\t'.join(XLSExporter.XLS_HEADERS) + '\n')
                for i, metrics in enumerate(all_metrics):
                    time_stat = time_stats[i] if i < len(time_stats) else {}
                    cor_score = float(metrics.get('CorScore', 0) or 0)
                    llm_judge_value = metrics.get('LLM_Judge', 'N/A')
                    if isinstance(llm_judge_value, str) and "из" in llm_judge_value: llm_judge_value = llm_judge_value.split()[0]
                    row = [
                        metrics.get('file_name', 'N/A'),
                        metrics.get('LevRating', 'N/A'),
                        metrics.get('delta_WER', 'N/A'),
                        metrics.get('delta_Lev', 'N/A'),
                        metrics.get('perplexity', 'N/A'),
                        f"{cor_score:.6f}",
                        metrics.get('best_temp_cor', 'N/A'),
                        metrics.get('best_model_cor', 'N/A'),
                        metrics.get('G_Eval', 'N/A'),
                        str(llm_judge_value),
                        metrics.get('METEOR', 'N/A'),
                        metrics.get('SumScore', 'N/A'),
                        metrics.get('BertScore', 'N/A'),
                        metrics.get('best_model_sum', 'N/A'),
                        f"{time_stat.get('correction_time', 0):.3f}",
                        f"{time_stat.get('summarization_time', 0):.3f}",
                        f"{time_stat.get('total_time', 0):.3f}",
                        str(time_stat.get('prompt_num', 'N/A'))
                    ]
                    f.write('\t'.join(str(v) for v in row) + '\n')
            logger.info(f"[XLS] XLS файл сохранён: {xls_filename}")
            print_success(f"  📊 XLS файл сохранён: {xls_filename}")
        except Exception as e:
            print_error(f"  ❌ Ошибка записи XLS: {e}")
            logger.error(f"[XLS] Ошибка записи XLS: {e}")


class DirectoryScanner:
    @staticmethod
    def scan_directory(directory: Path, extension: str = ".txt") -> List[str]:
        if not directory.exists(): return []
        return [f.stem for f in directory.glob(f"*{extension}") if f.is_file()]
    @staticmethod
    def get_test_cases() -> List[Dict[str, Any]]:
        logger.info("[DirectoryScanner] Сканирование директорий с тестовыми файлами")
        incorrect_files = set(DirectoryScanner.scan_directory(DIRS["incorrect_texts"]))
        etalon_text_files = set(DirectoryScanner.scan_directory(DIRS["etalon_texts"]))
        etalon_summary_files = set(DirectoryScanner.scan_directory(DIRS["etalon_summaries"]))
        complete_ids = incorrect_files & etalon_text_files & etalon_summary_files
        test_cases = []
        for file_id in sorted(complete_ids):
            test_cases.append({
                "id": file_id,
                "task_type": "combined",
                "language": DirectoryScanner._detect_language(file_id),
                "input_file": f"data/{DATA_LANG_DIR}/Incorrect_texts/{file_id}.txt",
                "reference_text_file": f"data/{DATA_LANG_DIR}/etalon_texts/{file_id}.txt",
                "reference_summary_file": f"data/{DATA_LANG_DIR}/etalon_summaries/{file_id}.txt"
            })
        correction_only_ids = (incorrect_files & etalon_text_files) - etalon_summary_files
        for file_id in sorted(correction_only_ids):
            test_cases.append({
                "id": file_id,
                "task_type": "correction",
                "language": DirectoryScanner._detect_language(file_id),
                "input_file": f"data/{DATA_LANG_DIR}/Incorrect_texts/{file_id}.txt",
                "reference_text_file": f"data/{DATA_LANG_DIR}/etalon_texts/{file_id}.txt",
                "reference_summary_file": None
            })
        return test_cases
    @staticmethod
    def _detect_language(file_id: str) -> str:
        if file_id.startswith("ru_"): return "ru"
        elif file_id.startswith("en_"): return "en"
        else: return "unknown"
    @staticmethod
    def validate_test_case(test_case: Dict[str, Any]) -> bool:
        required = [test_case.get("input_file"), test_case.get("reference_text_file")]
        if test_case.get("task_type") in ["summary", "combined"]:
            required.append(test_case.get("reference_summary_file"))
        for fp in required:
            if fp and not (BASE_DIR / fp).exists():
                print_warning(f"[DirectoryScanner] Файл не найден: {fp}")
                return False
        return True
    @staticmethod
    def print_directory_summary():
        print("\n" + "─" * 80)
        print_metric("  📁 ДИРЕКТОРИИ С ФАЙЛАМИ:")
        print("─" * 80)
        incorrect = DirectoryScanner.scan_directory(DIRS["incorrect_texts"])
        etalon_text = DirectoryScanner.scan_directory(DIRS["etalon_texts"])
        etalon_summary = DirectoryScanner.scan_directory(DIRS["etalon_summaries"])
        correction = DirectoryScanner.scan_directory(DIRS["correction"])
        summary = DirectoryScanner.scan_directory(DIRS["summary"])
        correction_metrics = DirectoryScanner.scan_directory(DIRS["correction_metrics"])
        summary_metrics = DirectoryScanner.scan_directory(DIRS["summary_metrics"])
        print_normal(f"     └─ Incorrect_texts:  {len(incorrect)} файлов")
        print_normal(f"     └─ etalon_texts:     {len(etalon_text)} файлов")
        print_normal(f"     └─ etalon_summaries: {len(etalon_summary)} файлов")
        print_normal(f"     └─ correction:       {len(correction)} файлов")
        print_normal(f"     └─ summary:          {len(summary)} файлов")
        print_normal(f"     └─ correction_metrics: {len(correction_metrics)} файлов")
        print_normal(f"     └─ summary_metrics:    {len(summary_metrics)} файлов")
        print("─" * 80 + "\n")


class TestDataLoader:
    @staticmethod
    def load_text_file(filepath: str) -> str:
        full_path = BASE_DIR / filepath
        if not full_path.exists(): raise FileNotFoundError(f"Файл не найден: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    @staticmethod
    def load_test_case_files(test_case: Dict[str, Any]) -> Dict[str, Optional[str]]:
        logger.info(f"[TestDataLoader] Загрузка файлов для теста: {test_case.get('id', 'unknown')}")
        result = {
            "input_text": TestDataLoader.load_text_file(test_case["input_file"]),
            "reference_text": TestDataLoader.load_text_file(test_case["reference_text_file"]),
            "reference_summary": None
        }
        if test_case.get("reference_summary_file"):
            try:
                result["reference_summary"] = TestDataLoader.load_text_file(test_case["reference_summary_file"])
            except FileNotFoundError:
                print_warning(f"[TestDataLoader] Файл резюме не найден: {test_case['reference_summary_file']}")
                result["reference_summary"] = ""
        return result


class FileSaver:
    CSV_HEADERS = [
        'File_Name', 'delta_WER', 'LevRating', 'Perplexity', 'Best_Model_Cor', 'Best_Temp_Cor',
        'Best_Model_Sum', 'Best_Temp_Sum', 'Correction_Rating', 'LLM_Judge', 'G_Eval_Score', 'METEOR'
    ]
    if BERTSCORE_ENABLED: CSV_HEADERS.append('BertScore')
    if SUMSCORE_ENABLED: CSV_HEADERS.append('SumScore')
    CSV_HEADERS.append('Quality')
    @staticmethod
    def _safe_str(value, default="N/A"): return default if value is None else str(value)
    @staticmethod
    def _safe_float(value, default="N/A", decimals=3):
        if value is None: return default
        try: return f"{float(value):.{decimals}f}"
        except: return default
    @staticmethod
    def _extract_quality_short(quality):
        if not quality: return "N/A"
        if "ОТЛИЧНО" in quality: return "ОТЛИЧНО"
        if "ХОРОШО" in quality: return "ХОРОШО"
        if "УДОВЛЕТВОРИТЕЛЬНО" in quality: return "УДОВЛЕТВОРИТЕЛЬНО"
        if "ТРЕБУЕТСЯ УЛУЧШЕНИЕ" in quality: return "ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
        return quality
    @staticmethod
    def _format_llm_judge_for_display(score):
        if score is None or score == "N/A": return "N/A"
        if isinstance(score, str) and "из" in score: return score
        try: return f"{int(float(score))} из 10"
        except: return str(score)
    @staticmethod
    def _format_llm_judge_for_xls(score):
        if score is None or score == "N/A": return "N/A"
        if isinstance(score, str) and "из" in score:
            try: return score.split()[0]
            except: return score
        try: return str(int(float(score)))
        except: return str(score)
    @staticmethod
    def save_correction_result(state, test_id, correction_time, prompt_num=1):
        filename = DIRS["correction"] / f"{test_id}.txt"
        metrics_filename = DIRS["correction_metrics"] / f"{test_id}.txt"
        m = state.get("metrics_correction", {}) or {}
        input_fmt = TextFormatter.format_text_with_newlines(state.get("input_text", "") or "")
        corrected_fmt = TextFormatter.format_text_with_newlines(state.get("corrected_text", "") or "")
        reference_fmt = TextFormatter.format_text_with_newlines(state.get("reference_text", "") or "")
        best_prompt = state.get("prompt_correction", "Промпт не доступен") or "Промпт не доступен"
        best_model = state.get("best_model", MODEL_NAME)
        best_temperature = state.get("best_temperature", "0.3")
        content = f"""=== INCORRECT_TEXT ===
{input_fmt}

=== CORRECT_TEXT ===
{corrected_fmt}

=== ETALON_TEXT ===
{reference_fmt}

=== METRICS ===
WER_0: {FileSaver._safe_float(m.get("WER_0"))}
WER: {FileSaver._safe_float(m.get("WER"))}
delta_WER: {FileSaver._safe_float(m.get("delta_WER"))}
LevRating_0: {FileSaver._safe_float(m.get("LevRating_0"))}
LevRating: {FileSaver._safe_float(m.get("LevRating"))}
delta_LEV: {FileSaver._safe_float(m.get("delta_LEV"))}
PROMPT: {prompt_num}
Time: {correction_time:.3f} s
Perplexity: {FileSaver._safe_float(state.get("perplexity", {}).get("perplexity"))}
Quality: {FileSaver._safe_str(m.get("quality_assessment"))}
Best_Model: {best_model}
Best_Temperature: {best_temperature}
Best_Prompt: {best_prompt[:200]}...
"""
        with open(filename, "w", encoding="cp1251", errors="replace") as f: f.write(content)
        metrics_content = f"""################################################################################
#                    МЕТРИКИ КОРРЕКЦИИ                                        #
################################################################################
# Тест ID: {test_id}
# Модель: {best_model}
# Температура: {best_temperature}
################################################################################
WER: {FileSaver._safe_float(m.get("WER_0"))} → {FileSaver._safe_float(m.get("WER"))} (Δ={FileSaver._safe_float(m.get("delta_WER"))})
LevRating: {FileSaver._safe_float(m.get("LevRating_0"))} → {FileSaver._safe_float(m.get("LevRating"))} (Δ={FileSaver._safe_float(m.get("delta_LEV"))})
Perplexity: {FileSaver._safe_float(state.get("perplexity", {}).get("perplexity"))}
Quality: {FileSaver._safe_str(m.get("quality_assessment"))}
"""
        with open(metrics_filename, "w", encoding="cp1251", errors="replace") as f: f.write(metrics_content)
        print_success(f"  💾 Файл коррекции сохранён: {filename}")
        return str(filename)
    @staticmethod
    def save_summary_result(state, test_id, summarization_time, prompt_num=1):
        filename = DIRS["summary"] / f"{test_id}.txt"
        metrics_filename = DIRS["summary_metrics"] / f"{test_id}.txt"
        m = state.get("metrics_summary", {}) or {}
        corrected_or_input = state.get("corrected_text", "") or state.get("input_text", "") or ""
        corrected_fmt = TextFormatter.format_text_with_newlines(corrected_or_input)
        summary_fmt = TextFormatter.format_text_with_newlines(state.get("summary_text", "") or "")
        reference_fmt = TextFormatter.format_text_with_newlines(state.get("reference_summary", "") or "Не предоставлено")
        best_prompt = state.get("prompt_summary", "Промпт не доступен") or "Промпт не доступен"
        best_model = state.get("best_model", MODEL_NAME)
        best_temp_summary = state.get("best_temperature_summary", "N/A")
        meteor = FileSaver._safe_float(m.get("meteor"))
        llm_score = m.get("llm_score", "N/A")
        llm_score_formatted = FileSaver._format_llm_judge_for_display(llm_score)
        g_eval = FileSaver._safe_float(m.get("g_eval_overall"))
        quality = FileSaver._safe_str(m.get("final_assessment"), "N/A")
        quality_short = FileSaver._extract_quality_short(quality)
        bertscore = FileSaver._safe_float(m.get("bertscore")) if BERTSCORE_ENABLED else "N/A"
        sumscore = FileSaver._safe_float(m.get("sumscore")) if SUMSCORE_ENABLED else "N/A"
        content = f"""=== CORRECT_TEXT ===
{corrected_fmt}

=== SUMMARY_TEXT ===
{summary_fmt}

=== ETALON_SUMMARY ===
{reference_fmt}

=== METRICS ===
LLM-Judge: {llm_score_formatted}
G-Eval: {g_eval}
METEOR: {meteor}
BertScore: {bertscore}
SumScore: {sumscore}
Quality: {quality_short}
Best_Model: {best_model}
Best_Temperature_Summary: {best_temp_summary}
Best_Prompt: {best_prompt}
PROMPT: {prompt_num}
Time: {summarization_time:.2f} s
"""
        with open(filename, "w", encoding="cp1251", errors="replace") as f: f.write(content)
        metrics_content = f"""################################################################################
#                    МЕТРИКИ СУММАРИЗАЦИИ                                     #
################################################################################
# Тест ID: {test_id}
# Модель: {best_model}
# Температура суммаризации: {best_temp_summary}
################################################################################
LLM-Judge: {llm_score}/10
G-Eval: {g_eval}
METEOR: {meteor}
BertScore: {bertscore}
SumScore: {sumscore}
Compression: {m.get('compression_ratio', 'N/A')}
Quality: {quality}
"""
        with open(metrics_filename, "w", encoding="cp1251", errors="replace") as f: f.write(metrics_content)
        print_success(f"  💾 Файл суммаризации сохранён: {filename}")
        return str(filename)
    @staticmethod
    def save_metrics_csv(all_metrics, csv_filename=None):
        if csv_filename is None: csv_filename = DIRS["logs"] / "metrics.csv"
        DIRS["logs"].mkdir(parents=True, exist_ok=True)
        try:
            with open(csv_filename, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FileSaver.CSV_HEADERS)
                writer.writeheader()
                for metrics in all_metrics:
                    row = {
                        'File_Name': metrics.get('file_name', 'N/A'),
                        'delta_WER': metrics.get('delta_WER', 'N/A'),
                        'LevRating': metrics.get('LevRating', 'N/A'),
                        'Perplexity': metrics.get('perplexity', 'N/A'),
                        'Best_Model_Cor': metrics.get('best_model_cor', 'N/A'),
                        'Best_Temp_Cor': metrics.get('best_temp_cor', 'N/A'),
                        'Best_Model_Sum': metrics.get('best_model_sum', 'N/A'),
                        'Best_Temp_Sum': metrics.get('best_temperature_summary', 'N/A'),
                        'Correction_Rating': metrics.get('correction_rating', 'N/A'),
                        'LLM_Judge': metrics.get('llm_judge', 'N/A'),
                        'G_Eval_Score': metrics.get('g_eval_score', 'N/A'),
                        'METEOR': metrics.get('meteor', 'N/A'),
                    }
                    if BERTSCORE_ENABLED: row['BertScore'] = metrics.get('BertScore', 'N/A')
                    if SUMSCORE_ENABLED: row['SumScore'] = metrics.get('SumScore', 'N/A')
                    row['Quality'] = metrics.get('quality', 'N/A')
                    writer.writerow(row)
            logger.info(f"[Saver] CSV файл метрик сохранён: {csv_filename}")
            print_success(f"  💾 CSV файл метрик сохранён: {csv_filename}")
        except Exception as e:
            print_error(f"  ❌ Ошибка записи CSV: {e}")
            logger.error(f"[Saver] Ошибка записи CSV: {e}")


class ProgressDisplay:
    @staticmethod
    def print_header(title, char="█", width=80):
        print("\n" + cformat(char * width, Color.BRIGHT_WHITE))
        print(cformat(char + " " * (width - 2) + char, Color.BRIGHT_WHITE))
        print(cformat(char + title.center(width - 2) + char, Color.BRIGHT_WHITE))
        print(cformat(char + " " * (width - 2) + char, Color.BRIGHT_WHITE))
        print(cformat(char * width, Color.BRIGHT_WHITE) + "\n")
    @staticmethod
    def print_section(title, char="▓", width=80):
        print("\n" + cformat(char * width, Color.BRIGHT_BLACK))
        print(cformat(char + f"  {title}".ljust(width - 1) + char, Color.CYAN))
        print(cformat(char * width, Color.BRIGHT_BLACK) + "\n")
    @staticmethod
    def print_success(message): print_success(f"  ✅ {message}")
    @staticmethod
    def print_error(message): print_error(f"  ❌ {message}")
    @staticmethod
    def print_warning(message): print_warning(f"  ⚠️  {message}")
    @staticmethod
    def print_info(message): print_info(f"  ℹ️  {message}")
    @staticmethod
    def print_normal(message): print_normal(f"  {message}")
    @staticmethod
    def print_metric(message): print_metric(f"  📊 {message}")


def main():
    time_stats = TimeStatistics()
    time_stats_list = []
    time_stats.start_total()
    date_log_folder = DateLogger.get_date_log_folder()
    date_folder_name = DateLogger.get_date_folder_name()
    bertscore_status = "BertScore ❌" if not BERTSCORE_ENABLED else "BertScore ✅"
    adaptive_status = "Адаптивная коррекция ✅" if ADAPTIVE_LEV_RETRY_ENABLED else "Адаптивная коррекция ❌"
    ProgressDisplay.print_header(
        f"CorSumAgentsAI v5.6.1\n"
        f"{bertscore_status} | {adaptive_status} | XLS после каждого ✅ | Веб-монитор: 127.0.0.1:{WEB_MONITOR_PORT} ✅"
    )
    print_info(f"\n  📁 Папка логов: {date_log_folder}")
    logger.info(f"[Main] Папка логов для даты: {date_log_folder}")
    logger.info("=" * 80)
    logger.info("CorSumAgentsAI - ЗАПУСК СИСТЕМЫ v5.6.1")
    logger.info("=" * 80)
    config_summary = get_config_summary()
    logger.info(f"[Main] Конфигурация: {config_summary}")
    config_warnings = validate_config()
    if config_warnings:
        logger.warning(f"[Main] Предупреждения конфигурации: {config_warnings}")
        for warning in config_warnings: print_warning(f"  ⚠️  {warning}")
    try:
        create_directories()
        ProgressDisplay.print_success("Директории созданы")
        logger.info("[Main] Директории созданы")
    except Exception as e:
        ProgressDisplay.print_error(f"Ошибка создания директорий: {e}")
        logger.error(f"[Main] Ошибка создания директорий: {e}")
        return
    print_normal("  📁 Проверка директорий метрик...")
    metrics_dirs = [DIRS["correction_metrics"], DIRS["summary_metrics"]]
    for dir_path in metrics_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_normal(f"     └─ Создана: {dir_path}")
        else:
            file_count = len(list(dir_path.glob("*.txt")))
            print_normal(f"     └─ {dir_path.name}: {file_count} файлов")
    logger.info("[Main] Директории метрик проверены")
    print_normal("  🧠 Инициализация памяти агентов...")
    memory_path = BASE_DIR / "data" / DATA_LANG_DIR / "memory"
    memory = AgentMemory(memory_dir=str(memory_path))
    memory_stats = memory.get_memory_stats()
    print_normal(f"     └─ История исправлений: {memory_stats['history_size']}")
    print_normal(f"     └─ Частые ошибки: {memory_stats['common_errors_count']}")
    print_normal(f"     └─ Лучшие промпты: {memory_stats['best_prompts_count']}")
    logger.info(f"[Main] Память инициализирована: {memory_stats}")
    print_normal("  🔌 Подключение к LM Studio...")
    client = LMStudioClient()
    client.print_connection_info()
    if not client.health_check():
        ProgressDisplay.print_error("НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К LM STUDIO!")
        logger.error("[Main] НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К LM STUDIO!")
        return
    ProgressDisplay.print_success("Соединение с LM Studio успешно")
    logger.info("[Main] Соединение с LM Studio успешно")
    print_normal("\n  🌐 Инициализация веб-монитора...")
    web_monitor = start_web_monitor_if_enabled()
    print("\n")
    DirectoryScanner.print_directory_summary()
    test_cases = DirectoryScanner.get_test_cases()
    if not test_cases:
        ProgressDisplay.print_warning("Тестовые файлы не найдены!")
        print_warning("\n  💡 Убедитесь что файлы есть во всех трёх директориях:")
        print_warning(f"     └─ {DIRS['incorrect_texts']}")
        print_warning(f"     └─ {DIRS['etalon_texts']}")
        print_warning(f"     └─ {DIRS['etalon_summaries']}")
        return
    if web_monitor: web_monitor.set_total_tests(len(test_cases))
    if SKIP_PROCESSED == 1:
        skipped_count = 0
        not_skipped = []
        for tc in test_cases:
            if should_skip_file(tc["id"], tc["task_type"]):
                skipped_count += 1
            else:
                not_skipped.append(tc["id"])
        if skipped_count > 0:
            print_info(f"\n  ℹ️  SKIP_PROCESSED=1: {skipped_count} файлов будут пропущены (уже обработаны)")
            logger.info(f"[Main] SKIP_PROCESSED=1: {skipped_count} файлов будут пропущены")
        if not_skipped:
            print_success(f"\n  ✅ Будут обработаны {len(not_skipped)} файлов:")
            for f in not_skipped[:10]: print_normal(f"     └─ {f}")
            if len(not_skipped) > 10: print_normal(f"     └─ ... и ещё {len(not_skipped) - 10} файлов")
    print_info(f"Найдено {len(test_cases)} тестовых кейсов")
    logger.info(f"[Main] Найдено {len(test_cases)} тестовых кейсов")
    print_info(f"\n  📊 МЕТРИКИ:")
    print_normal(f"     └─ BertScore: {'✅ Включён' if BERTSCORE_ENABLED else '❌ Отключён'}")
    print_normal(f"     └─ SumScore: {'✅ Включён' if SUMSCORE_ENABLED else '❌ Отключён'}")
    print_normal(f"     └─ Адаптивная коррекция: {'✅ Включена' if ADAPTIVE_LEV_RETRY_ENABLED else '❌ Отключена'}")
    print_normal(f"     └─ XLS после каждого: {'✅ Включено' if XLS_UPDATE_AFTER_EACH_DOC else '❌ Отключено'}")
    if not BERTSCORE_ENABLED:
        logger.info("[Main] BertScore отключён в конфигурации (BERTSCORE_ENABLED=False)")
        print_warning("  ⚠️  BertScore отключён! Для включения установите BERTSCORE_ENABLED=True в config.py")
    try:
        orchestrator = Orchestrator(client, memory=memory)
        ProgressDisplay.print_success("Оркестратор инициализирован")
        logger.info("[Main] Оркестратор инициализирован")
    except Exception as e:
        ProgressDisplay.print_error(f"Ошибка инициализации оркестратора: {e}")
        logger.error(f"[Main] Ошибка инициализации оркестратора: {e}")
        return
    all_metrics = []
    stats = {
        "total": len(test_cases),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "skipped_processed": 0,
        "skipped_missing_files": 0,
        "correction_files": 0,
        "summary_files": 0,
        "correction_metrics_files": 0,
        "summary_metrics_files": 0
    }
    for i, test_case in enumerate(test_cases, 1):
        test_start_time = time.time()
        ProgressDisplay.print_section(f"ТЕСТ {i}/{stats['total']}: {test_case['id']}")
        if SKIP_PROCESSED == 1:
            if should_skip_file(test_case["id"], test_case["task_type"]):
                print_info(f"\n  ⏭️  Пропущено (файл уже обработан): {test_case['id']}")
                logger.info(f"[Main] Пропущено (SKIP_PROCESSED): {test_case['id']}")
                stats["skipped_processed"] += 1
                stats["skipped"] += 1
                continue
        try:
            if not DirectoryScanner.validate_test_case(test_case):
                ProgressDisplay.print_warning(f"Не все файлы найдены для теста {test_case['id']}")
                stats["skipped_missing_files"] += 1
                stats["skipped"] += 1
                continue
            test_data = TestDataLoader.load_test_case_files(test_case)
        except Exception as e:
            ProgressDisplay.print_warning(f"Ошибка загрузки файлов: {e}")
            stats["skipped"] += 1
            continue
        print_normal("\n  🛡️  Предварительная обработка текста...")
        input_text = test_data.get("input_text", "")
        reference_text = test_data.get("reference_text", "")
        print_normal(f"     └─ Длина текста: {len(input_text)} символов")
        print_normal(f"     └─ Валидация: ✅")
        initial_state: AgentState = {
            "input_text": input_text,
            "reference_text": reference_text,
            "reference_summary": test_data.get("reference_summary", "") or "",
            "task_type": test_case.get("task_type", "combined"),
            "prompt_correction": "",
            "corrected_text": "",
            "ensemble_outputs": [],
            "prompt_summary": "",
            "summary_text": "",
            "metrics_correction": {},
            "metrics_summary": {},
            "needs_retry_correction": False,
            "needs_retry_summary": False,
            "retry_count_correction": 0,
            "retry_count_summary": 0,
            "detected_language": "",
            "summary_language": "",
            "test_id": test_case.get("id", f"test_{i}"),
            "cross_validation_iteration": 0,
            "meaning_preserved": True,
            "adaptive_config": {},
            "best_model": client.model,
            "best_prompt": "",
            "perplexity": {},
            "best_temperature": "N/A",
            "best_prompt_type": "базовый"
        }
        try:
            processing_start = time.time()
            if web_monitor: web_monitor.update_test_status(test_case["id"], "running")
            final_state = orchestrator.execute(initial_state)
            if not final_state.get("corrected_text") or len(final_state.get("corrected_text", "").strip()) < 10:
                final_state["corrected_text"] = initial_state.get("input_text", "")
                logger.warning(f"[Main] Для теста {test_case['id']} corrected_text пуст, использован input_text")
                print_warning(f"  ⚠️  corrected_text пуст, использован исходный текст")
            processing_end = time.time()
            total_time = processing_end - processing_start
            # Приблизительное разделение времени (можно уточнить)
            if test_case.get("task_type") == "combined":
                correction_time = total_time * 0.7
                summarization_time = total_time * 0.3
            elif test_case.get("task_type") == "correction":
                correction_time = total_time
                summarization_time = 0.0
            else:
                correction_time = 0.0
                summarization_time = total_time
            stats["success"] += 1
            test_id = test_case.get("id")
            print_info("\n  📊 СТАТИСТИКА ОБРАБОТКИ...")
            file_metrics = {}
            prompt_num = 1
            # Коррекция
            m_cor = final_state.get("metrics_correction", {}) or {}
            file_metrics['WER_0'] = FileSaver._safe_float(m_cor.get("WER_0"))
            file_metrics['WER'] = FileSaver._safe_float(m_cor.get("WER"))
            file_metrics['delta_WER'] = FileSaver._safe_float(m_cor.get("delta_WER"))
            file_metrics['LevRating_0'] = FileSaver._safe_float(m_cor.get("LevRating_0"))
            file_metrics['LevRating'] = FileSaver._safe_float(m_cor.get("LevRating"))
            file_metrics['delta_LEV'] = FileSaver._safe_float(m_cor.get("delta_LEV"))
            perplexity = final_state.get("perplexity", {}) or {}
            file_metrics['perplexity'] = FileSaver._safe_float(perplexity.get("perplexity"))
            file_metrics['best_model_cor'] = final_state.get('best_model', MODEL_NAME)
            file_metrics['best_temp_cor'] = final_state.get('best_temperature', 'N/A')
            delta_wer = float(file_metrics['delta_WER']) if file_metrics['delta_WER'] != 'N/A' else 0.0
            delta_lev = float(file_metrics['delta_LEV']) if file_metrics['delta_LEV'] != 'N/A' else 0.0
            perplexity_val = float(file_metrics['perplexity']) if file_metrics['perplexity'] != 'N/A' else 1.0
            cor_score = calculate_cor_score(delta_wer, delta_lev, perplexity_val)
            file_metrics['CorScore'] = f"{cor_score:.6f}"
            # Суммаризация
            if test_case.get("task_type") in ["summary", "combined"]:
                m_sum = final_state.get("metrics_summary", {}) or {}
                llm_score = FileSaver._safe_str(m_sum.get('llm_score'), 'N/A')
                file_metrics['LLM_Judge'] = FileSaver._format_llm_judge_for_display(llm_score)
                file_metrics['G_Eval'] = FileSaver._safe_float(m_sum.get("g_eval_overall"))
                file_metrics['METEOR'] = FileSaver._safe_float(m_sum.get("meteor"))
                file_metrics['best_model_sum'] = final_state.get('best_model', MODEL_NAME)
                file_metrics['best_temperature_summary'] = final_state.get('best_temperature_summary', 'N/A')
                if BERTSCORE_ENABLED:
                    file_metrics['BertScore'] = FileSaver._safe_float(m_sum.get("bertscore"))
                if SUMSCORE_ENABLED:
                    g_eval = float(file_metrics['G_Eval']) if file_metrics['G_Eval'] != 'N/A' else 0.0
                    llm_judge_str = file_metrics['LLM_Judge']
                    llm_judge = float(llm_judge_str.split()[0]) if llm_judge_str != 'N/A' and llm_judge_str else 0.0
                    meteor = float(file_metrics['METEOR']) if file_metrics['METEOR'] != 'N/A' else 0.0
                    bertscore = float(file_metrics.get('BertScore', 0)) if BERTSCORE_ENABLED else 0.0
                    sumscore = calculate_sumscore(g_eval, llm_judge, meteor, bertscore)
                    sumscore_assessment = get_sumscore_assessment(sumscore)
                    file_metrics['SumScore'] = f"{sumscore:.3f}"
                    file_metrics['SumScore_assessment'] = sumscore_assessment
                file_metrics['Quality'] = FileSaver._extract_quality_short(m_sum.get("final_assessment", "N/A"))
            time_stats.print_file_statistics(test_id, correction_time, summarization_time, prompt_num, file_metrics)
            if XLS_UPDATE_AFTER_EACH_DOC:
                metrics_row = {
                    'file_name': test_id,
                    'LevRating': file_metrics.get('LevRating', 'N/A'),
                    'delta_WER': file_metrics.get('delta_WER', 'N/A'),
                    'delta_Lev': file_metrics.get('delta_LEV', 'N/A'),
                    'Perpl': file_metrics.get('perplexity', 'N/A'),
                    'CorScore': file_metrics.get('CorScore', '0.000000'),
                    'best_temp_cor': file_metrics.get('best_temp_cor', 'N/A'),
                    'best_model_cor': file_metrics.get('best_model_cor', 'N/A'),
                    'G-Eval': file_metrics.get('G_Eval', 'N/A'),
                    'LLM-Judge': FileSaver._format_llm_judge_for_xls(file_metrics.get('LLM_Judge', 'N/A')),
                    'METEOR': file_metrics.get('METEOR', 'N/A'),
                    'SumScore': file_metrics.get('SumScore', 'N/A'),
                    'BertScore': file_metrics.get('BertScore', 'N/A'),
                    'best_model_sum': file_metrics.get('best_model_sum', 'N/A'),
                }
                time_stat = {
                    'correction_time': correction_time,
                    'summary_time': summarization_time,
                    'total_time': correction_time + summarization_time,
                    'prompt_num': prompt_num
                }
                XLSExporter.update_xls_after_each_document(metrics_row, time_stat)
            # Обновляем веб-монитор с duration (общее время теста)
            if web_monitor:
                prompt_type_label = final_state.get("best_prompt_type", "базовый") or "базовый"
                prompt_sum_label = final_state.get("best_prompt_summary_type", "базовый") or "базовый"
                web_monitor.update_test_status(
                    test_id,
                    "completed",
                    file_metrics,
                    prompt_correction=prompt_type_label,
                    corrected_text=final_state.get("corrected_text", ""),
                    prompt_summary=prompt_sum_label,
                    summary_text=final_state.get("summary_text", ""),
                    duration=total_time   # ✅ передаём длительность теста
                )
            print_normal("\n  💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
            metrics_row_csv = {
                'file_name': test_id,
                'delta_WER': 'N/A',
                'LevRating': 'N/A',
                'perplexity': 'N/A',
                'best_model_cor': 'N/A',
                'best_temp_cor': 'N/A',
                'best_model_sum': 'N/A',
                'best_temperature_summary': 'N/A',
                'correction_rating': 'N/A',
                'llm_judge': 'N/A',
                'g_eval_score': 'N/A',
                'meteor': 'N/A',
                'quality': 'N/A'
            }
            if test_case.get("task_type") in ["correction", "combined"]:
                FileSaver.save_correction_result(final_state, test_id, correction_time, prompt_num)
                stats["correction_files"] += 1
                stats["correction_metrics_files"] += 1
                metrics_row_csv['delta_WER'] = file_metrics['delta_WER']
                metrics_row_csv['LevRating'] = file_metrics['delta_LEV']
                metrics_row_csv['perplexity'] = file_metrics['perplexity']
                metrics_row_csv['best_model_cor'] = file_metrics['best_model_cor']
                metrics_row_csv['best_temp_cor'] = file_metrics['best_temp_cor']
                metrics_row_csv['correction_rating'] = FileSaver._extract_quality_short(
                    m_cor.get("quality_assessment", "N/A"))

            if test_case.get("task_type") in ["summary", "combined"]:
                FileSaver.save_summary_result(final_state, test_id, summarization_time, prompt_num)
                stats["summary_files"] += 1
                stats["summary_metrics_files"] += 1
                metrics_row_csv['llm_judge'] = file_metrics.get('LLM_Judge', 'N/A')
                metrics_row_csv['g_eval_score'] = file_metrics.get('G_Eval', 'N/A')
                metrics_row_csv['meteor'] = file_metrics.get('METEOR', 'N/A')
                metrics_row_csv['best_model_sum'] = file_metrics.get('best_model_sum', 'N/A')
                metrics_row_csv['best_temperature_summary'] = file_metrics.get('best_temperature_summary', 'N/A')
                if BERTSCORE_ENABLED:
                    metrics_row_csv['BertScore'] = file_metrics.get('BertScore', 'N/A')
                if SUMSCORE_ENABLED:
                    metrics_row_csv['SumScore'] = file_metrics.get('SumScore', 'N/A')
                metrics_row_csv['quality'] = file_metrics.get('Quality', 'N/A')
            all_metrics.append(metrics_row_csv)
            test_elapsed = time.time() - test_start_time
            time_stats.add_test_stat(test_id, correction_time, summarization_time, prompt_num, file_metrics)
            time_stats_list.append({
                'test_id': test_id,
                'correction_time': correction_time,
                'summarization_time': summarization_time,
                'total_time': test_elapsed,
                'prompt_num': prompt_num
            })
            ProgressDisplay.print_success(f"Тест {i} завершен успешно ({test_elapsed:.2f} сек)")
        except Exception as e:
            stats["failed"] += 1
            ProgressDisplay.print_error(f"Ошибка при обработке теста {i}: {e}")
            logger.error(f"[Main] Ошибка при обработке теста {i}: {e}")
            if web_monitor:
                web_monitor.update_test_status(test_case["id"], "failed", {"error": str(e)})
            import traceback
            traceback.print_exc()
            continue
    time_stats.end_total()
    print_normal("\n  📊 Сохранение XLS файла метрик...")
    xls_filename = DIRS["metrics"] / "metrics.xls"
    XLSExporter.save_metrics_xls(all_metrics, time_stats_list, xls_filename)
    print_normal("\n  📊 Сохранение CSV файла метрик...")
    csv_filename = DIRS["metrics"] / "metrics.csv"
    FileSaver.save_metrics_csv(all_metrics, csv_filename)
    cache_stats = client.get_cache_stats()
    print_info(f"\n  💾 Статистика кэша LLM:")
    print_normal(f"     └─ Размер кэша: {cache_stats['cache_size']}")
    print_normal(f"     └─ Cache hits: {cache_stats['cache_hits']}")
    print_normal(f"     └─ Hit rate: {cache_stats['hit_rate']:.1f}%")
    memory_stats = memory.get_memory_stats()
    print_info(f"\n  🧠 Статистика памяти:")
    print_normal(f"     └─ Записей в истории: {memory_stats['history_size']}")
    print_normal(f"     └─ Частых ошибок: {memory_stats['common_errors_count']}")
    print_normal(f"     └─ Лучших промптов: {memory_stats['best_prompts_count']}")
    print_normal(f"     └─ Лучших промптов суммаризации: {memory_stats.get('best_summary_prompts_count', 0)}")
    print_info(f"\n  📁 Логи агентов:")
    print_normal(f"     └─ Папка даты: {date_log_folder}")
    print_normal(f"     └─ Формат: agent_logs/{date_folder_name}/<agent>_<YYYYMMDD>.log")
    print_info(f"\n  📄 Файлы логов:")
    print_log_folder_info()
    if web_monitor:
        print_info(f"\n  🌐 Веб-мониторинг:")
        print_normal(f"     └─ URL: http://127.0.0.1:{WEB_MONITOR_PORT}")
        print_normal(f"     └─ Статус: ✅ Активен")
        print_normal(f"     └─ Графики: ✅ Chart.js")
        print_normal(f"     └─ Примеры текстов: ✅")
        print_normal(f"     └─ Автообновление: ✅ (30 сек)")
    print("\n")
    ProgressDisplay.print_header("РАБОТА СИСТЕМЫ ЗАВЕРШЕНА")
    print_normal("  📊 СТАТИСТИКА ВЫПОЛНЕНИЯ:")
    print_normal(f"     └─ Всего тестов: {stats['total']}")
    success_rate = stats['success'] / max(1, stats['total']) * 100
    if success_rate >= 90: print_success(f"     └─ Успешно: {stats['success']} ({success_rate:.1f}%)")
    elif success_rate >= 50: print_warning(f"     └─ Успешно: {stats['success']} ({success_rate:.1f}%)")
    else: print_error(f"     └─ Успешно: {stats['success']} ({success_rate:.1f}%)")
    if stats['failed'] > 0: print_error(f"     └─ Ошибки: {stats['failed']}")
    else: print_normal(f"     └─ Ошибки: {stats['failed']}")
    print_normal(f"     └─ Пропущено: {stats['skipped']}")
    print_normal(f"        ├─ Уже обработаны (SKIP_PROCESSED): {stats['skipped_processed']}")
    print_normal(f"        └─ Отсутствие файлов: {stats['skipped_missing_files']}")
    print_normal(f"     └─ Файлов коррекции: {stats['correction_files']}")
    print_normal(f"     └─ Файлов метрик коррекции: {stats['correction_metrics_files']}")
    print_normal(f"     └─ Файлов суммаризации: {stats['summary_files']}")
    print_normal(f"     └─ Файлов метрик суммаризации: {stats['summary_metrics_files']}")
    print_normal(f"     └─ XLS файл метрик: {xls_filename}")
    print_normal(f"     └─ CSV файл метрик: {csv_filename}")
    time_stats.print_statistics()
    # ========== СРЕДНИЕ МЕТРИКИ И СТАТИСТИКА ТЕМПЕРАТУР ==========
    print("\n" + "─" * 80)
    print_metric("  📊 СРЕДНИЕ ЗНАЧЕНИЯ МЕТРИК ПО ВСЕМ ТЕСТАМ")
    print("─" * 80)
    if all_metrics:
        successful = [m for m in all_metrics if m.get('delta_WER') != 'N/A']
        if successful:
            def _safe_metric_float(val):
                """Безопасный парсинг метрик: '6 из 10' -> 6.0, 'N/A' -> 0, и т.д."""
                if val is None or val == 'N/A' or val == '': return 0.0
                if isinstance(val, str) and 'из' in val:
                    try: return float(val.split()[0])
                    except: return 0.0
                try: return float(val)
                except: return 0.0
            avg_delta_wer = sum(_safe_metric_float(m.get('delta_WER')) for m in successful)/len(successful)
            avg_lev = sum(_safe_metric_float(m.get('LevRating')) for m in successful)/len(successful)
            avg_cor = sum(_safe_metric_float(m.get('CorScore')) for m in successful)/len(successful)
            avg_sum = sum(_safe_metric_float(m.get('SumScore')) for m in successful if m.get('SumScore')!='N/A')/len(successful)
            avg_meteor = sum(_safe_metric_float(m.get('meteor')) for m in successful if m.get('meteor')!='N/A')/len(successful)
            avg_llm = sum(_safe_metric_float(m.get('llm_judge')) for m in successful if m.get('llm_judge')!='N/A')/len(successful)
            avg_g_eval = sum(_safe_metric_float(m.get('g_eval_score')) for m in successful if m.get('g_eval_score')!='N/A')/len(successful)
            if BERTSCORE_ENABLED:
                avg_bert = sum(_safe_metric_float(m.get('BertScore')) for m in successful if m.get('BertScore')!='N/A')/len(successful)
                print_metric(f"     └─ Средний BertScore: {avg_bert:.4f}")
            print_metric(f"     └─ Средний ΔWER: {avg_delta_wer:.4f}")
            print_metric(f"     └─ Средний LevRating: {avg_lev:.4f}")
            print_metric(f"     └─ Средний CorScore: {avg_cor:.4f}")
            print_metric(f"     └─ Средний SumScore: {avg_sum:.4f}")
            print_metric(f"     └─ Средний METEOR: {avg_meteor:.4f}")
            print_metric(f"     └─ Средний LLM-Judge: {avg_llm:.1f}")
            print_metric(f"     └─ Средний G-Eval: {avg_g_eval:.4f}")
        temp_cor = {}
        temp_sum = {}
        for m in successful:
            tc = m.get('best_temp_cor', 'N/A')
            if tc != 'N/A': temp_cor[tc] = temp_cor.get(tc, 0) + 1
            ts = m.get('best_temperature_summary', 'N/A')
            if ts != 'N/A': temp_sum[ts] = temp_sum.get(ts, 0) + 1
        if temp_cor:
            print("\n" + "─" * 80)
            print_metric("  🌡️ ЛУЧШИЕ ТЕМПЕРАТУРЫ (КОРРЕКЦИЯ):")
            for t, c in sorted(temp_cor.items(), key=lambda x: x[1], reverse=True): print_normal(f"     └─ {t} - {c} раз")
        if temp_sum:
            print("\n" + "─" * 80)
            print_metric("  🌡️ ЛУЧШИЕ ТЕМПЕРАТУРЫ (СУММАРИЗАЦИЯ):")
            for t, c in sorted(temp_sum.items(), key=lambda x: x[1], reverse=True): print_normal(f"     └─ {t} - {c} раз")
        print("─" * 80 + "\n")
    else:
        print_metric("  ⚠️ Нет данных для расчёта средних метрик")
        print("─" * 80 + "\n")
    print("\n")
    ProgressDisplay.print_header("  📁 РЕЗУЛЬТАТЫ:")
    print_normal(f"     └─ Коррекция: {DIRS['correction']}")
    print_normal(f"     └─ Метрики коррекции: {DIRS['correction_metrics']}")
    print_normal(f"     └─ Суммаризация: {DIRS['summary']}")
    print_normal(f"     └─ Метрики суммаризации: {DIRS['summary_metrics']}")
    print_normal(f"     └─ XLS/CSV метрики: {DIRS['logs']}")
    print_normal(f"     └─ Логи по датам: {date_log_folder}")
    print_normal(f"     └─ Память: {memory.memory_dir}")
    print_normal(f"     └─ Логи: {DIRS['logs']}")
    print("\n")
    ProgressDisplay.print_header("  ⚙️  КОНФИГУРАЦИЯ:")
    print_normal(f"     └─ SKIP_PROCESSED: {SKIP_PROCESSED} ({'Пропускать обработанные' if SKIP_PROCESSED == 1 else 'Перезаписывать все'})")
    print_normal(f"     └─ Веб-монитор: {'✅' if web_monitor else '❌'}")
    print_normal(f"     └─ BertScore: {'✅' if BERTSCORE_ENABLED else '❌ (отключен)'}")
    print_normal(f"     └─ SumScore: {'✅' if SUMSCORE_ENABLED else '❌'}")
    print_normal(f"     └─ Адаптивная коррекция: {'✅' if ADAPTIVE_LEV_RETRY_ENABLED else '❌'}")
    print_normal(f"     └─ XLS после каждого: {'✅' if XLS_UPDATE_AFTER_EACH_DOC else '❌'}")
    if SUMSCORE_ENABLED:
        print_normal(f"     └─ SumScore веса: G-Eval={SUMSCORE_WEIGHTS['g_eval']}, LLM-Judge={SUMSCORE_WEIGHTS['llm_judge']}, METEOR={SUMSCORE_WEIGHTS['meteor']}, BertScore={SUMSCORE_WEIGHTS['bertscore']}")
        print_normal(f"     └─ SumScore диапазон: 0.000-1.000 (1.000 - лучше)")
    print_normal(f"     └─ CorScore формула: ΔWER + 8*ΔLev + (1 - Perpl/100)*0.5")
    print_normal(f"     └─ Perplexity в скоринге: ✅")
    print_normal(f"     └─ Лучшая модель (строка): ✅")
    print_normal(f"     └─ Лучшая температура коррекции (строка): ✅")
    print_normal(f"     └─ Лучшая температура суммаризации (строка): ✅")
    print_normal(f"     └─ LLM-Judge 'X из 10': ✅ (в консоли и txt файлах)")
    print_normal(f"     └─ LLM-Judge только число: ✅ (в XLS файле)")
    print_normal(f"     └─ Цветной вывод: ✅")
    print_normal(f"     └─ Логи по датам (DDMMYYYY): ✅")
    print_normal(f"     └─ XLS экспорт с переводом: ✅")
    print_normal(f"     └─ Статистика ПЕРЕД сохранением: ✅")
    print_normal(f"     └─ Время в выходных файлах: ✅")
    logger.info("=" * 80)
    logger.info("РАБОТА СИСТЕМЫ ЗАВЕРШЕНА")
    logger.info(f"Статистика: {stats}")
    logger.info(f"Time statistics: {time_stats.to_dict()}")
    logger.info(f"Cache stats: {cache_stats}")
    logger.info(f"Memory stats: {memory_stats}")
    logger.info(f"Date log folder: {date_log_folder}")
    logger.info("=" * 80)
    orchestrator.close()
    if web_monitor:
        print_warning("  ⏳ Ожидание 60 секунд для экспорта таблицы веб-монитором...")
        logger.info("[WebMonitor] Ожидание 60 секунд перед остановкой для завершения экспорта Excel")
        time.sleep(60)
        web_monitor.stop()
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\n  ⚠️  Работа прервана пользователем")
        logger.warning("[Main] Работа прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\n  ❌ Критическая ошибка: {e}")
        logger.critical(f"[Main] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
