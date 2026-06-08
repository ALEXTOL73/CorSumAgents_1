#!/usr/bin/env python3
"""
Веб-интерфейс мониторинга CorSumAgentsAI
Версия 5.8.7 – Фильтрация пустых тестов, исправлены промпты и длительность
"""
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from utils.logger import setup_logger

logger = setup_logger("WebMonitor")

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)

# Глобальное состояние
monitor_state = {
    "status": "idle",
    "total_tests": 0,
    "completed_tests": 0,
    "failed_tests": 0,
    "current_test": "",
    "start_time": None,
    "end_time": None,
    "tests": {},
    "metrics": {
        "avg_wer_improvement": 0,
        "avg_lev_improvement": 0,
        "avg_llm_judge": 0,
        "avg_meteor": 0,
        "avg_bertscore": 0,
        "avg_sumscore": 0,
        "avg_levrating": 0,
        "avg_geval": 0,
        "avg_corscore": 0
    },
    "last_update": None,
    "examples": {"outputs": []}
}

monitor_lock = threading.Lock()

# Пути (импорт из config или значения по умолчанию)
try:
    from config import DIRS, MODEL_NAME, calculate_cor_score, DATA_LANG_DIR
except ImportError:
    DIRS = {
        "correction_metrics": Path("data/correction_metrics"),
        "summary_metrics": Path("data/summary_metrics"),
        "correction": Path("data/correction"),
        "summary": Path("data/summary"),
        "logs": Path("logs"),
        "metrics": Path("data/metrics"),
    }
    MODEL_NAME = "gemma-3-12b-it"
    DATA_LANG_DIR = "RUS"  # fallback
    def calculate_cor_score(delta_wer, delta_lev, perplexity):
        # Примерная формула, можно заменить на свою
        return max(0.0, min(1.0, delta_wer * 0.5 + delta_lev * 0.3 + (1 - perplexity) * 0.2))


class WebMonitor:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self.app = None
        self.server = None
        self.server_thread = None
        self.running = False

        # Загружаем все существующие результаты с диска
        self._load_existing_results()

        try:
            from flask import Flask
            self.app = Flask(__name__)
            self._setup_routes()
            logger.info(f"[WebMonitor] Инициализирован: http://{host}:{port}")
        except ImportError as e:
            logger.warning(f"[WebMonitor] Flask не установлен! {e}")

    # -------------------------------------------------------------------------
    # Парсинг WER и LevRating (формат со стрелками)
    # -------------------------------------------------------------------------
    def _parse_wer_line(self, line: str) -> tuple:
        # ✅ Поддержка форматов: 
        #   WER: 0.276 → 0.105 (Δ=0.171) - правильный utf-8
        #   WER: 0.276 ? 0.105 (?=0.171) - encoding corrupted
        match = re.search(r'WER:\s*([\d.]+)\s+[^\s]+\s+([\d.]+)\s+\([Δ?]?=([\d.-]+)\)', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return 0.0, 0.0, 0.0

    def _parse_lev_line(self, line: str) -> tuple:
        # ✅ Поддержка форматов:
        #   LevRating: 0.935 → 0.963 (Δ=0.028) - правильный utf-8
        #   LevRating: 0.935 ? 0.963 (?=0.028) - encoding corrupted
        match = re.search(r'LevRating:\s*([\d.]+)\s+[^\s]+\s+([\d.]+)\s+\([Δ?]?=([\d.-]+)\)', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return 0.0, 0.0, 0.0

    # -------------------------------------------------------------------------
    # Загрузка метрик из Excel (metrics.xls)
    # -------------------------------------------------------------------------
    def _load_excel_metrics(self):
        excel_data = {}
        # ✅ Читаем metrics.xls из data/{DATA_LANG_DIR}/metrics/
        metrics_dir = DIRS.get("metrics", Path("data") / DATA_LANG_DIR / "metrics")
        xls_path = metrics_dir / "metrics.xls"
        if not xls_path.exists():
            logger.info(f"[WebMonitor] metrics.xls не найден в {metrics_dir}, CorScore будет вычислен")
            return excel_data

        try:
            # Пробуем cp1251 сначала, потом utf-8
            try:
                with open(xls_path, 'r', encoding='cp1251') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(xls_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            if len(lines) > 1:
                headers = lines[0].strip().split('\t')
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if not parts:
                        continue
                    test_id = parts[0]
                    metrics = {}
                    for i, h in enumerate(headers):
                        if i >= len(parts):
                            break
                        val = parts[i]
                        if val == 'N/A' or val == '':
                            continue
                        if h == 'CorScore':
                            metrics['CorScore'] = float(val)
                        elif h == 'G-Eval':
                            metrics['G-Eval'] = float(val)
                        elif h == 'LLM-Judge':
                            metrics['LLM_Judge'] = float(val)
                        elif h == 'METEOR':
                            metrics['METEOR'] = float(val)
                        elif h == 'SumScore':
                            metrics['SumScore'] = float(val)
                        elif h == 'BertScore':
                            metrics['BertScore'] = float(val)
                        elif h == 'delta_WER':
                            metrics['delta_WER'] = float(val)
                        elif h == 'delta_Lev':
                            metrics['delta_LEV'] = float(val)
                        elif h == 'LevRating':
                            metrics['LevRating'] = float(val)
                        elif h == 'Perpl':
                            metrics['perplexity'] = float(val)
                        elif h == 'Total_Time':
                            metrics['total_time'] = float(val)
                        elif h == 'Correction_Time':
                            try:
                                metrics['correction_time'] = float(val)
                            except ValueError:
                                pass
                        elif h == 'Summary_Time':
                            try:
                                metrics['summary_time'] = float(val)
                            except ValueError:
                                pass
                        elif h == 'Prompt_Num':
                            metrics['prompt_num'] = val
                        elif h == 'best_temp_cor':
                            metrics['best_temp_cor'] = val
                        elif h == 'best_temperature_summary':
                            metrics['best_temperature_summary'] = val
                        elif h == 'best_model_cor':
                            metrics['best_model_cor'] = val
                        elif h == 'best_model_sum':
                            metrics['best_model_sum'] = val
                    if metrics:
                        excel_data[test_id] = metrics
            logger.info(f"[WebMonitor] Загружено {len(excel_data)} записей из Excel")
        except Exception as e:
            logger.warning(f"Ошибка чтения metrics.xls: {e}")
        return excel_data

    # -------------------------------------------------------------------------
    # Главная загрузка всех тестов с диска
    # -------------------------------------------------------------------------
    def _load_existing_results(self):
        logger.info("[WebMonitor] Загрузка существующих результатов...")
        excel_data = self._load_excel_metrics()

        correction_metrics_dir = DIRS.get("correction_metrics", Path("data/correction_metrics"))
        summary_metrics_dir = DIRS.get("summary_metrics", Path("data/summary_metrics"))
        correction_dir = DIRS.get("correction", Path("data/correction"))
        summary_dir = DIRS.get("summary", Path("data/summary"))

        logger.info(f"[WebMonitor] correction_metrics_dir: {correction_metrics_dir}")
        logger.info(f"[WebMonitor] summary_metrics_dir: {summary_metrics_dir}")
        logger.info(f"[WebMonitor] correction_dir: {correction_dir}")
        logger.info(f"[WebMonitor] summary_dir: {summary_dir}")

        test_ids = set()
        if correction_metrics_dir.exists():
            test_ids.update(f.stem for f in correction_metrics_dir.glob("*.txt"))
        if summary_metrics_dir.exists():
            test_ids.update(f.stem for f in summary_metrics_dir.glob("*.txt"))
        test_ids.update(excel_data.keys())

        # ✅ Сортируем test_ids с учётом числового порядка (natural sort)
        def natural_sort_key(s):
            """Сортировка: 1_, 2_, 10_, 20_, 100_ вместо 1_, 10_, 100_, 2_, 20_"""
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
        sorted_test_ids = sorted(test_ids, key=natural_sort_key)

        tests_dict = {}
        examples_list = []

        for test_id in sorted_test_ids:
            metrics = {}
            prompt_correction = None
            prompt_summary = None
            duration = 0.0
            corrected_text = ""
            summary_text = ""

            # --- Чтение correction_metrics (WER, LevRating, Perplexity, Temperature, Time, Prompt) ---
            corr_metrics_file = correction_metrics_dir / f"{test_id}.txt"
            if corr_metrics_file.exists():
                try:
                    # Пробуем cp1251 сначала, потом utf-8
                    try:
                        content = corr_metrics_file.read_text(encoding='cp1251')
                    except UnicodeDecodeError:
                        content = corr_metrics_file.read_text(encoding='utf-8')
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith('WER:'):
                            w0, w1, dw = self._parse_wer_line(line)
                            metrics['WER_0'] = w0
                            metrics['WER'] = w1
                            metrics['delta_WER'] = dw
                        elif line.startswith('LevRating:'):
                            l0, l1, dl = self._parse_lev_line(line)
                            metrics['LevRating_0'] = l0
                            metrics['LevRating'] = l1
                            metrics['delta_LEV'] = dl
                        elif 'Perplexity:' in line and not line.startswith('#'):
                            m = re.search(r'Perplexity:\s*([\d.]+)', line)
                            if m:
                                metrics['perplexity'] = float(m.group(1))
                        # Температура из комментария (# Температура: 0.1) или из тела (Best_Temperature: 0.1)
                        elif ('Температура:' in line or 'Temperature:' in line or 'температура:' in line) and 'best_temp_cor' not in metrics:
                            m = re.search(r'[Тт]емпература[:\s]*([\d.]+)', line)
                            if not m:
                                m = re.search(r'Temperature[:\s]*([\d.]+)', line, re.IGNORECASE)
                            if m:
                                metrics['best_temp_cor'] = m.group(1)
                        elif line.startswith('Best_Temperature:') and 'best_temp_cor' not in metrics:
                            m = re.search(r'Best_Temperature:\s*([^\s]+)', line)
                            if m:
                                val = m.group(1).strip()
                                if val and val != 'N/A':
                                    metrics['best_temp_cor'] = val
                        # Время выполнения (Time: 246.437)
                        elif line.startswith('Time:') and duration == 0.0:
                            m = re.search(r'Time:\s*([\d.]+)', line)
                            if m:
                                duration = float(m.group(1))
                        # Тип промпта (# Промпт: базовый или Best_Prompt: базовый)
                        elif ('Промпт:' in line or 'Prompt:' in line) and not prompt_correction:
                            m = re.search(r'[Пп]ромпт[:\s]*(.+)', line)
                            if not m:
                                m = re.search(r'Prompt[:\s]*(.+)', line, re.IGNORECASE)
                            if m:
                                pval = m.group(1).strip()
                                # Отбрасываем декоративные символы # из комментариев
                                pval = pval.lstrip('#').strip()
                                if pval and pval != 'N/A' and not pval.startswith('###'):
                                    prompt_correction = pval
                        elif line.startswith('Best_Prompt:') and not prompt_correction:
                            m = re.search(r'Best_Prompt:\s*(.+)', line)
                            if m:
                                pval = m.group(1).strip()
                                if pval and pval != 'N/A':
                                    prompt_correction = pval
                except Exception as e:
                    logger.warning(f"Ошибка чтения {corr_metrics_file}: {e}")

            # --- Чтение summary_metrics (LLM-Judge, G-Eval, METEOR, BertScore, SumScore, Temperature, Time, Prompt) ---
            sum_metrics_file = summary_metrics_dir / f"{test_id}.txt"
            if sum_metrics_file.exists():
                try:
                    # Пробуем cp1251 сначала, потом utf-8
                    try:
                        content = sum_metrics_file.read_text(encoding='cp1251')
                    except UnicodeDecodeError:
                        content = sum_metrics_file.read_text(encoding='utf-8')
                    m = re.search(r'LLM-Judge:\s*(\d+)(?:/10)?', content)
                    if m:
                        metrics['LLM_Judge'] = int(m.group(1))
                    m = re.search(r'G-Eval:\s*([\d.]+)', content)
                    if m:
                        metrics['G-Eval'] = float(m.group(1))
                    m = re.search(r'METEOR:\s*([\d.]+)', content)
                    if m:
                        metrics['METEOR'] = float(m.group(1))
                    m = re.search(r'BertScore:\s*([\d.]+)', content)
                    if m:
                        metrics['BertScore'] = float(m.group(1))
                    m = re.search(r'SumScore:\s*([\d.]+)', content)
                    if m:
                        metrics['SumScore'] = float(m.group(1))
                    # Температура суммаризации (из комментария или Best_Temperature)
                    if 'best_temperature_summary' not in metrics:
                        m = re.search(r'[Тт]емпература\s+суммаризации[:\s]*([\d.]+)', content)
                        if not m:
                            m = re.search(r'Best_Temperature:\s*([^\s\n]+)', content)
                        if m:
                            val = m.group(1).strip()
                            if val and val != 'N/A':
                                metrics['best_temperature_summary'] = val
                    # Тип промпта суммаризации
                    if not prompt_summary:
                        m = re.search(r'[Пп]ромпт[:\s]*(.+)', content)
                        if not m:
                            m = re.search(r'^Best_Prompt:\s*(.+)', content, re.MULTILINE)
                        if m:
                            pval = m.group(1).strip().lstrip('#').strip()
                            if pval and pval != 'N/A' and not pval.startswith('###'):
                                prompt_summary = pval
                except Exception as e:
                    logger.warning(f"Ошибка чтения {sum_metrics_file}: {e}")

            # --- Чтение текстового файла коррекции (Best_Prompt, Time, corrected_text) ---
            corr_text_file = correction_dir / f"{test_id}.txt"
            if corr_text_file.exists():
                try:
                    # Пробуем cp1251 сначала, потом utf-8
                    try:
                        content = corr_text_file.read_text(encoding='cp1251')
                    except UnicodeDecodeError:
                        content = corr_text_file.read_text(encoding='utf-8')
                    # Best_Prompt
                    m = re.search(r'Best_Prompt:\s*(.+?)(?:\n|$)', content)
                    if m:
                        prompt_correction = m.group(1).strip()
                    # Best_Temperature (коррекция)
                    m = re.search(r'Best_Temperature:\s*([^\n]+)', content)
                    if m:
                        temp_val = m.group(1).strip()
                        if temp_val and temp_val != 'N/A':
                            metrics['best_temp_cor'] = temp_val
                    # Time
                    m = re.search(r'Time:\s*([\d.]+)\s*s', content)
                    if m:
                        duration = float(m.group(1))
                    # corrected_text для примеров
                    # ✅ Ищем текст между CORRECT_TEXT и ETALON_TEXT (гибкий regex)
                    m = re.search(r'=== CORRECT_TEXT ===\s*\n(.*?)\n=== ETALON_TEXT ===', content, re.DOTALL)
                    if m:
                        corrected_text = m.group(1).strip()
                    else:
                        # Fallback: пробуем без \n после ===
                        m = re.search(r'=== CORRECT_TEXT ===(.*?)=== ETALON_TEXT ===', content, re.DOTALL)
                        if m:
                            corrected_text = m.group(1).strip()
                except Exception as e:
                    logger.warning(f"Ошибка чтения {corr_text_file}: {e}")

            # --- Чтение текстового файла суммаризации (Best_Prompt, summary_text) ---
            sum_text_file = summary_dir / f"{test_id}.txt"
            if sum_text_file.exists():
                try:
                    # Пробуем cp1251 сначала, потом utf-8
                    try:
                        content = sum_text_file.read_text(encoding='cp1251')
                    except UnicodeDecodeError:
                        content = sum_text_file.read_text(encoding='utf-8')
                    m = re.search(r'Best_Prompt:\s*(.+?)(?:\n|$)', content)
                    if m:
                        prompt_summary = m.group(1).strip()
                    # Best_Temperature_Summary (суммаризация)
                    m = re.search(r'Best_Temperature_Summary:\s*([^\n]+)', content)
                    if m:
                        temp_val = m.group(1).strip()
                        if temp_val and temp_val != 'N/A':
                            metrics['best_temperature_summary'] = temp_val
                    m = re.search(r'=== SUMMARY_TEXT ===\n(.*?)\n=== ETALON_SUMMARY ===', content, re.DOTALL)
                    if m:
                        summary_text = m.group(1).strip()
                except Exception as e:
                    logger.warning(f"Ошибка чтения {sum_text_file}: {e}")

            # --- Обогащаем метриками из Excel (приоритет) ---
            if test_id in excel_data:
                ex = excel_data[test_id]
                for k, v in ex.items():
                    # Не перезаписываем duration и prompt из Excel, если они уже установлены из файлов
                    if k == 'correction_time' and duration == 0.0:
                        duration = v
                        continue
                    if k == 'total_time' and duration == 0.0:
                        duration = v
                        continue
                    if k == 'prompt_num' and not prompt_correction:
                        try:
                            pnum = int(v)
                            prompt_correction = f"№{pnum} (базовый)" if pnum == 1 else f"№{pnum}"
                        except (ValueError, TypeError):
                            prompt_correction = str(v)
                        continue
                    metrics[k] = v

            # --- Вычисляем CorScore, если отсутствует ---
            if 'CorScore' not in metrics:
                delta_wer = metrics.get('delta_WER', 0.0) or 0.0
                delta_lev = metrics.get('delta_LEV', 0.0) or 0.0
                perplexity = metrics.get('perplexity', 1.0) or 1.0
                metrics['CorScore'] = calculate_cor_score(delta_wer, delta_lev, perplexity)

            # ========== ФИЛЬТРАЦИЯ ПУСТЫХ ТЕСТОВ ==========
            # Список ключевых метрик, которые должны быть ненулевыми
            key_metrics = ['delta_WER', 'delta_LEV', 'LevRating', 'CorScore', 'G-Eval', 'SumScore', 'METEOR', 'LLM_Judge', 'BertScore']
            non_zero_count = 0
            for km in key_metrics:
                val = metrics.get(km)
                if val is not None and val != 0 and val != 'N/A' and val != '—':
                    non_zero_count += 1
            # Если меньше двух значащих метрик — пропускаем тест
            if non_zero_count < 2:
                logger.debug(f"Тест {test_id} пропущен (мало значащих метрик: {non_zero_count})")
                continue
            # =============================================

            # Нормализация промптов (если не найдены, ставим "№1 (базовый)")
            if not prompt_correction:
                prompt_correction = "№1 (базовый)"
            if not prompt_summary:
                prompt_summary = "№1 (базовый)"

            # --- Добавляем примеры ---
            if corrected_text or summary_text:
                examples_list.append({
                    "test_id": test_id,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "status": "completed",
                    "corrected_text": (corrected_text[:500] + "...") if corrected_text and len(corrected_text) > 500 else corrected_text,
                    "summary_text": (summary_text[:500] + "...") if summary_text and len(summary_text) > 500 else summary_text
                })

            # --- Формируем запись теста ---
            tests_dict[test_id] = {
                "id": test_id,
                "status": "completed",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "prompt_correction_text": prompt_correction,
                "prompt_summary_text": prompt_summary,
                "corrected_text": (corrected_text[:500] + "...") if corrected_text and len(corrected_text) > 500 else corrected_text,
                "summary_text": (summary_text[:500] + "...") if summary_text and len(summary_text) > 500 else summary_text,
            }

        # Сохраняем в глобальное состояние
        with monitor_lock:
            monitor_state["tests"] = tests_dict
            monitor_state["completed_tests"] = len(tests_dict)
            monitor_state["total_tests"] = len(tests_dict)
            monitor_state["failed_tests"] = 0
            monitor_state["examples"]["outputs"] = examples_list
            self._recalculate_aggregated_metrics()
            monitor_state["last_update"] = datetime.now().isoformat()

        logger.info(f"[WebMonitor] Загружено {len(tests_dict)} тестов (отфильтровано пустых)")

    def _recalculate_aggregated_metrics(self):
        tests = monitor_state["tests"]
        if not tests:
            return
        agg = {
            "avg_wer_improvement": 0.0,
            "avg_lev_improvement": 0.0,
            "avg_llm_judge": 0.0,
            "avg_meteor": 0.0,
            "avg_bertscore": 0.0,
            "avg_sumscore": 0.0,
            "avg_levrating": 0.0,
            "avg_geval": 0.0,
            "avg_corscore": 0.0,
        }
        count = len(tests)
        for t in tests.values():
            m = t.get("metrics", {})
            agg["avg_wer_improvement"] += m.get("delta_WER", 0.0)
            agg["avg_lev_improvement"] += m.get("delta_LEV", 0.0)
            agg["avg_llm_judge"] += m.get("LLM_Judge", 0.0)
            agg["avg_meteor"] += m.get("METEOR", 0.0)
            agg["avg_bertscore"] += m.get("BertScore", 0.0)
            agg["avg_sumscore"] += m.get("SumScore", 0.0)
            agg["avg_levrating"] += m.get("LevRating", 0.0)
            agg["avg_geval"] += m.get("G-Eval", 0.0)
            agg["avg_corscore"] += m.get("CorScore", 0.0)
        for key in agg:
            agg[key] = round(agg[key] / count, 6)
        monitor_state["metrics"] = agg

    # -------------------------------------------------------------------------
    # Оригинальные методы (Flask routes, update, start, stop) – без изменений
    # -------------------------------------------------------------------------
    def _setup_routes(self):
        from flask import jsonify, render_template_string

        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE, host=self.host, port=self.port)

        @self.app.route('/health')
        def health():
            return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

        @self.app.route('/api/status')
        def get_status():
            with monitor_lock:
                elapsed = 0
                if monitor_state["start_time"]:
                    elapsed = (datetime.now() - monitor_state["start_time"]).total_seconds()
                state_copy = monitor_state.copy()
                state_copy["tests"] = list(monitor_state["tests"].values())
                state_copy["program_elapsed"] = elapsed
                return jsonify(state_copy)

        @self.app.route('/api/tests')
        def get_tests():
            with monitor_lock:
                return jsonify({"tests": list(monitor_state["tests"].values())})

        @self.app.route('/api/metrics')
        def get_metrics():
            with monitor_lock:
                return jsonify({"metrics": monitor_state["metrics"]})

        @self.app.route('/api/progress')
        def get_progress():
            with monitor_lock:
                total = monitor_state["total_tests"]
                completed = monitor_state["completed_tests"]
                progress = (completed / total * 100) if total > 0 else 0
                return jsonify({
                    "total": total,
                    "completed": completed,
                    "failed": monitor_state["failed_tests"],
                    "progress": progress,
                    "current_test": monitor_state["current_test"],
                    "status": monitor_state["status"]
                })

        @self.app.route('/api/examples')
        def get_examples():
            with monitor_lock:
                return jsonify(monitor_state["examples"])

        @self.app.route('/api/reset', methods=['POST'])
        def reset_state():
            global monitor_state
            with monitor_lock:
                monitor_state = {
                    "status": "idle",
                    "total_tests": 0,
                    "completed_tests": 0,
                    "failed_tests": 0,
                    "current_test": "",
                    "start_time": None,
                    "end_time": None,
                    "tests": {},
                    "metrics": {
                        "avg_wer_improvement": 0,
                        "avg_lev_improvement": 0,
                        "avg_llm_judge": 0,
                        "avg_meteor": 0,
                        "avg_bertscore": 0,
                        "avg_sumscore": 0,
                        "avg_levrating": 0,
                        "avg_geval": 0,
                        "avg_corscore": 0
                    },
                    "last_update": None,
                    "examples": {"outputs": []}
                }
            self._load_existing_results()
            return jsonify({"status": "reset"})

    def update_test_status(self, test_id: str, status: str, metrics: Dict[str, Any] = None,
                           prompt_correction: str = None, prompt_summary: str = None,
                           corrected_text: str = None, summary_text: str = None,
                           duration: float = None,
                           top_temps_cor: str = None, top_temps_sum: str = None):
        if not self.running:
            return

        # Нормализация ключей
        if metrics:
            if 'G_Eval' in metrics:
                metrics['G-Eval'] = metrics.pop('G_Eval')
            if 'geval' in metrics:
                metrics['G-Eval'] = metrics.pop('geval')
            if 'Lev_Rating' in metrics:
                metrics['LevRating'] = metrics.pop('Lev_Rating')
            if 'levrating' in metrics:
                metrics['LevRating'] = metrics.pop('levrating')
            if 'meteor' in metrics and 'METEOR' not in metrics:
                metrics['METEOR'] = metrics['meteor']
            if 'best_temperature' in metrics and 'best_temp_cor' not in metrics:
                metrics['best_temp_cor'] = metrics['best_temperature']
            elif 'best_temp_cor' not in metrics:
                metrics['best_temp_cor'] = 'N/A'
            if 'best_temperature_summary' not in metrics:
                metrics['best_temperature_summary'] = 'N/A'
            if 'delta_wer' in metrics:
                metrics['delta_WER'] = metrics.pop('delta_wer')
            if 'delta_lev' in metrics:
                metrics['delta_LEV'] = metrics.pop('delta_lev')
            if 'cor_score' in metrics:
                metrics['CorScore'] = metrics.pop('cor_score')
            if 'corscore' in metrics:
                metrics['CorScore'] = metrics.pop('corscore')

        with monitor_lock:
            if status == "running":
                if monitor_state["start_time"] is None:
                    monitor_state["start_time"] = datetime.now()
                monitor_state["current_test"] = test_id
                monitor_state["status"] = "running"

            elif status == "completed":
                test_data = {
                    "id": test_id,
                    "status": "completed",
                    "metrics": metrics or {},
                    "timestamp": datetime.now().isoformat(),
                    "duration": duration if duration is not None else 0.0,
                    "prompt_correction_text": prompt_correction or "№1 (базовый)",
                    "prompt_summary_text": prompt_summary or "№1 (базовый)",
                    "corrected_text": (corrected_text[:500] + "...") if corrected_text and len(corrected_text) > 500 else corrected_text,
                    "summary_text": (summary_text[:500] + "...") if summary_text and len(summary_text) > 500 else summary_text,
                    "top_temps_cor": top_temps_cor or "",
                    "top_temps_sum": top_temps_sum or "",
                }
                monitor_state["tests"][test_id] = test_data
                monitor_state["completed_tests"] = len(
                    [t for t in monitor_state["tests"].values() if t["status"] == "completed"])
                self._update_aggregated_metrics(metrics)

                if corrected_text or summary_text:
                    example = {
                        "test_id": test_id,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "status": "completed",
                        "corrected_text": corrected_text[:500] + "..." if corrected_text and len(
                            corrected_text) > 500 else corrected_text,
                        "summary_text": summary_text[:500] + "..." if summary_text and len(
                            summary_text) > 500 else summary_text
                    }
                    monitor_state["examples"]["outputs"].insert(0, example)
                    monitor_state["examples"]["outputs"] = monitor_state["examples"]["outputs"][:10]

            elif status == "failed":
                test_data = {
                    "id": test_id,
                    "status": "failed",
                    "metrics": metrics or {},
                    "timestamp": datetime.now().isoformat(),
                    "duration": None,
                    "prompt_correction_text": None,
                    "prompt_summary_text": None
                }
                monitor_state["tests"][test_id] = test_data
                monitor_state["failed_tests"] = len(
                    [t for t in monitor_state["tests"].values() if t["status"] == "failed"])

            elif status == "system":
                if test_id == "system":
                    monitor_state["status"] = "completed"
                    monitor_state["end_time"] = datetime.now().isoformat()

            monitor_state["last_update"] = datetime.now().isoformat()

        logger.debug(f"[WebMonitor] Обновлён тест {test_id}: {status}")

    def set_total_tests(self, total: int):
        if not self.running:
            return
        with monitor_lock:
            monitor_state["total_tests"] = total
            monitor_state["last_update"] = datetime.now().isoformat()
            logger.info(f"[WebMonitor] Установлено общее количество тестов: {total}")

    def _update_aggregated_metrics(self, new_metrics: Dict[str, Any]):
        if not new_metrics:
            return

        completed = monitor_state["completed_tests"]
        metrics = monitor_state["metrics"]

        def update(key, val_key=None):
            if val_key is None:
                val_key = key
            if val_key in new_metrics and new_metrics[val_key] != 'N/A':
                try:
                    v = new_metrics[val_key]
                    if isinstance(v, str) and ' ' in v:
                        v = v.split()[0]
                    val = float(v)
                    old = metrics.get(key, 0)
                    metrics[key] = old + (val - old) / completed
                except (ValueError, TypeError):
                    pass

        update("avg_wer_improvement", "delta_WER")
        update("avg_lev_improvement", "delta_LEV")
        update("avg_llm_judge", "LLM_Judge")
        update("avg_meteor", "METEOR")
        update("avg_bertscore", "BertScore")
        update("avg_sumscore", "SumScore")
        update("avg_levrating", "LevRating")
        update("avg_geval", "G-Eval")
        update("avg_corscore", "CorScore")

    def start(self):
        if self.app is None:
            logger.warning("[WebMonitor] Flask не установлен, не могу запустить сервер")
            return
        if self.running:
            return
        try:
            try:
                from werkzeug.serving import make_server
                use_werkzeug = True
            except ImportError:
                use_werkzeug = False
            self.running = True
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            logging.getLogger('flask').setLevel(logging.ERROR)
            if use_werkzeug:
                self.server = make_server(self.host, self.port, self.app, threaded=True)
                self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.server_thread.start()
                logger.info(f"[WebMonitor] Сервер запущен (werkzeug): http://{self.host}:{self.port}")
                print("✅ WebMonitor запущен")
            else:
                self.server_thread = threading.Thread(
                    target=self.app.run,
                    kwargs={'host': self.host, 'port': self.port, 'debug': False, 'use_reloader': False,
                            'threaded': True},
                    daemon=True
                )
                self.server_thread.start()
                logger.info(f"[WebMonitor] Сервер запущен (Flask): http://{self.host}:{self.port}")
                print("✅ WebMonitor запущен")
            time.sleep(1)
            try:
                import urllib.request
                response = urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=2)
                if response.getcode() == 200:
                    logger.info("[WebMonitor] Сервер доступен")
                    print("✅ WebMonitor: сервер доступен")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[WebMonitor] Ошибка запуска: {e}")
            self.running = False
            raise

    def stop(self):
        if not self.running:
            return
        try:
            if hasattr(self, 'server') and self.server:
                self.server.shutdown()
            self.running = False
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=2)
            logger.info("[WebMonitor] Сервер остановлен")
            print("✅ WebMonitor остановлен")
        except Exception as e:
            logger.error(f"[WebMonitor] Ошибка остановки: {e}")

    def is_running(self) -> bool:
        return self.running


# =============================================================================
# HTML-шаблон (полностью из версии 5.7.15, без изменений)
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>CorSumAgentsAI - Мониторинг</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.sheetjs.com/xlsx-0.20.2/package/dist/xlsx.full.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1600px; margin: 0 auto; }
        .header { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
        .header-left { flex: 1; }
        .header-left h1 { color: #667eea; margin-bottom: 10px; }
        .header-left p { color: #666; }
        .export-btn { background: #28a745; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; margin-left: 20px; transition: background 0.2s; }
        .export-btn:hover { background: #218838; }
        .status-bar { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
        .status-item { background: white; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-item h3 { color: #666; font-size: 14px; }
        .status-item p { color: #333; font-size: 24px; font-weight: bold; }
        .progress-bar { width: 100%; height: 30px; background: #e9ecef; border-radius: 15px; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 15px; display: flex; align-items: center; justify-content: center; color: white; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin: 20px 0; }
        .chart-card { background: white; border-radius: 10px; padding: 20px; }
        .chart-card h3 { color: #666; margin-bottom: 15px; }
        .examples-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin: 20px 0; }
        .example-card { background: white; border-radius: 10px; padding: 20px; }
        .example-item { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .example-item.failed { background: #f8d7da; border-left: 4px solid #dc3545; }
        .example-item pre { white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow: auto; font-size: 11px; }
        .table-container { overflow-x: auto; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #e9ecef; font-size: 13px; }
        th { background: #f8f9fa; color: #666; position: sticky; top: 0; }
        tr.failed-row { background-color: #f8d7da; }
        .below-threshold { background-color: #f8d7da; color: #721c24; font-weight: bold; }
        .status-badge { padding: 5px 10px; border-radius: 20px; font-size: 12px; display: inline-block; }
        .status-completed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .status-running { background: #fff3cd; color: #856404; }
        .footer { text-align: center; color: white; margin-top: 20px; font-size: 12px; }
        .average-row { background-color: #e3f2fd; font-weight: bold; border-top: 2px solid #667eea; }
        .average-row td { background-color: #e3f2fd; }
        @media (max-width: 768px) { .charts-grid, .examples-grid { grid-template-columns: 1fr; } .header { flex-direction: column; align-items: flex-start; } .export-btn { margin-left: 0; margin-top: 15px; } }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="header-left">
            <h1>🚀 CorSumAgentsAI - Мониторинг</h1>
            <p>Адрес: http://127.0.0.1:{{port}} | Обновление каждые 30 секунд</p>
        </div>
        <button id="exportExcelBtn" class="export-btn">📊 Экспорт в Excel</button>
    </div>
    <div class="status-bar">
        <div class="status-item"><h3>Статус</h3><p id="status">-</p></div>
        <div class="status-item"><h3>Всего</h3><p id="total">0</p></div>
        <div class="status-item"><h3>Завершено</h3><p id="completed">0</p></div>
        <div class="status-item"><h3>Ошибки</h3><p id="failed">0</p></div>
        <div class="status-item"><h3>⏱️ Время работы</h3><p id="program-elapsed">0 сек</p></div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progress" style="width:0%">0%</div></div>

    <div class="charts-grid">
        <div class="chart-card"><h3>📈 delta_WER</h3><canvas id="werChart"></canvas><div>Порог: >0</div></div>
        <div class="chart-card"><h3>📊 LevRating (0-1)</h3><canvas id="levChart"></canvas><div>Порог: >0.6</div></div>
        <div class="chart-card"><h3>⭐ G-Eval (0-1)</h3><canvas id="gevalChart"></canvas><div>Порог: >0.5</div></div>
        <div class="chart-card"><h3>📝 METEOR (0-1)</h3><canvas id="meteorChart"></canvas><div>Порог: >0.2</div></div>
        <div class="chart-card"><h3>🏆 CorScore</h3><canvas id="corscoreChart"></canvas><div>Порог: >0.3</div></div>
        <div class="chart-card"><h3>🏆 SumScore (0-1)</h3><canvas id="sumscoreChart"></canvas><div>Порог: >0.5</div></div>
        <div class="chart-card"><h3>📉 BertScore (0-1)</h3><canvas id="bertscoreChart"></canvas><div>Порог: >0.7</div></div>
        <div class="chart-card"><h3>🤖 LLM-Judge (1-10)</h3><canvas id="llmjudgeChart"></canvas><div>Порог: >5</div></div>
    </div>

    <div class="examples-grid">
        <div class="example-card"><h3>✏️ Исправленные Тексты</h3><div id="corrected-texts">Загрузка...</div></div>
        <div class="example-card"><h3>📄 Резюме</h3><div id="summary-texts">Загрузка...</div></div>
    </div>

    <div class="table-container">
        <table id="metrics-table">
            <thead>
                <tr><th>ID</th><th>Статус</th><th>ΔWER</th><th>ΔLev</th><th>LevRating</th><th>Perplexity</th><th>CorScore</th>
                    <th>Prompt_cor</th><th>Temp</th><th>G-Eval</th><th>BertScore</th><th>METEOR</th><th>LLM-Judge</th>
                    <th>SumScore</th><th>Prompt_sum</th><th>Temp_sum</th><th>Длительность</th>
                </tr>
            </thead>
            <tbody id="tests-body">
                <tr><td colspan="17">Загрузка...<\/td><\/tr>
            <\/tbody>
        <\/table>
    <\/div>
    <div class="footer">Автообновление 30 сек | Красный фон — значение ниже порога | Длительность — время выполнения теста (сек)<\/div>
<\/div>

<script>
    let charts = {};
    const UPDATE_INTERVAL = 30000;

    const THRESHOLDS = {
        delta_WER: 0, delta_LEV: 0.01, LevRating: 0.6, Perplexity: 1.0,
        CorScore: 0.3, 'G-Eval': 0.5, BertScore: 0.7, METEOR: 0.2,
        LLM_Judge: 5, SumScore: 0.5
    };

    function getMetricValue(metrics, key, def=0) {
        if (!metrics) return def;
        let v = metrics[key];
        if (v === undefined || v === null || v === 'N/A') return def;
        if (typeof v === 'string') v = v.split(' ')[0];
        const n = parseFloat(v);
        return isNaN(n) ? def : n;
    }

    function formatDuration(seconds) {
        if (seconds === undefined || seconds === null || seconds === 0) return '—';
        return seconds.toFixed(2) + ' сек';
    }

    function formatElapsed(seconds) {
        if (seconds === undefined || seconds === null) return '0 сек';
        if (seconds < 60) return seconds.toFixed(0) + ' сек';
        let mins = Math.floor(seconds / 60);
        let secs = Math.floor(seconds % 60);
        if (mins < 60) return mins + ' мин ' + secs + ' сек';
        let hours = Math.floor(mins / 60);
        mins = mins % 60;
        return hours + ' ч ' + mins + ' мин ' + secs + ' сек';
    }

    function getPromptNumber(promptText) {
        if (!promptText) return null;
        const text = promptText.toLowerCase();
        if (text.includes('self-consistency') || text.includes('sc')) return 5;
        if (text.includes('cot') || text.includes('chain of thought') || text.includes('шаг')) return 4;
        if (text.includes('few-shot') || text.includes('few_shot') || text.includes('пример') || text.includes('примеры')) return 3;
        if (text.includes('saved') || text.includes('из памяти') || text.includes('лучший промпт')) return 2;
        return 1;
    }

    function getPromptDescription(promptText) {
        const num = getPromptNumber(promptText);
        const desc = {
            1: 'базовый',
            2: 'из памяти',
            3: 'few-shot',
            4: 'CoT',
            5: 'SC'
        };
        return desc[num] || 'N/A';
    }

    function formatPromptCell(promptText) {
        const num = getPromptNumber(promptText);
        if (num === null) return 'N/A';
        const desc = getPromptDescription(promptText);
        return `№${num} (${desc})`;
    }

    async function refreshData() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            document.getElementById('status').innerText = data.status || '-';
            document.getElementById('total').innerText = data.total_tests || 0;
            document.getElementById('completed').innerText = data.completed_tests || 0;
            document.getElementById('failed').innerText = data.failed_tests || 0;
            const elapsed = data.program_elapsed || 0;
            document.getElementById('program-elapsed').innerText = formatElapsed(elapsed);
            const total = data.total_tests || 1;
            const progress = ((data.completed_tests || 0) / total * 100).toFixed(1);
            document.getElementById('progress').style.width = progress + '%';
            document.getElementById('progress').innerText = progress + '%';
            updateCharts(data.tests);
            updateExamples(data.examples, data.tests);
            updateTable(data.tests);
        } catch(e) { console.error(e); }
    }

    function updateCharts(tests) {
        if (!tests || tests.length === 0) return;
        const last = tests.slice(-30);
        const labels = last.map(t => t.id);
        const deltaWer = last.map(t => getMetricValue(t.metrics, 'delta_WER', 0));
        const levRating = last.map(t => getMetricValue(t.metrics, 'LevRating', 0));
        const gEval = last.map(t => getMetricValue(t.metrics, 'G-Eval', 0));
        const meteor = last.map(t => getMetricValue(t.metrics, 'METEOR', 0));
        const corScore = last.map(t => getMetricValue(t.metrics, 'CorScore', 0));
        const sumScore = last.map(t => getMetricValue(t.metrics, 'SumScore', 0));
        const bertScore = last.map(t => getMetricValue(t.metrics, 'BertScore', 0));
        const llmJudge = last.map(t => getMetricValue(t.metrics, 'LLM_Judge', 0));

        const thr = { wer:0, lev:0.6, geval:0.5, meteor:0.2, corscore:0.3, sumscore:0.5, bertscore:0.7, llmjudge:5 };
        function create(id, data, label, color, yMin, yMax, threshold, isBar) {
            const ctx = document.getElementById(id);
            if (!ctx) return;
            if (charts[id]) charts[id].destroy();
            const ds = [{ label, data, borderColor: color, backgroundColor: isBar ? color : 'transparent', borderWidth: 2, fill: false, type: isBar ? 'bar' : 'line' }];
            if (threshold !== undefined) ds.push({ label: 'Порог', data: Array(data.length).fill(threshold), borderColor: 'green', borderWidth: 2, borderDash: [5,5], fill: false, type: 'line' });
            charts[id] = new Chart(ctx, { data: { labels, datasets: ds }, options: { responsive: true, scales: { y: { min: yMin, max: yMax } } } });
        }
        create('werChart', deltaWer, 'ΔWER', '#28a745', -0.5, 1, thr.wer, true);
        create('levChart', levRating, 'LevRating', '#ffc107', 0, 1, thr.lev, false);
        create('gevalChart', gEval, 'G-Eval', '#17a2b8', 0, 1, thr.geval, false);
        create('meteorChart', meteor, 'METEOR', '#6f42c1', 0, 1, thr.meteor, false);
        create('corscoreChart', corScore, 'CorScore', '#fd7e14', -0.5, 1, thr.corscore, true);
        create('sumscoreChart', sumScore, 'SumScore', '#667eea', 0, 1, thr.sumscore, false);
        create('bertscoreChart', bertScore, 'BertScore', '#20c997', 0, 1, thr.bertscore, false);
        create('llmjudgeChart', llmJudge, 'LLM-Judge', '#ff8c00', 0, 10, thr.llmjudge, false);
    }

    function updateExamples(examples, tests) {
        const corrDiv = document.getElementById('corrected-texts');
        const sumDiv = document.getElementById('summary-texts');
        const allTests = tests || [];

        // Последние 5 тестов с corrected_text
        const withCorr = allTests.filter(t => t.corrected_text && t.corrected_text.length > 0);
        const last5Corr = withCorr.slice(-5);
        // Последние 5 тестов с summary_text
        const withSum = allTests.filter(t => t.summary_text && t.summary_text.length > 0);
        const last5Sum = withSum.slice(-5);

        // Исправленные тексты
        if (last5Corr.length === 0) {
            const fallbackCorr = (examples && examples.outputs) ? examples.outputs.filter(e => e.corrected_text).slice(-5) : [];
            if (fallbackCorr.length === 0) {
                corrDiv.innerHTML = '<p>Нет данных</p>';
            } else {
                corrDiv.innerHTML = fallbackCorr.map(ex => `<div class="example-item ${ex.status==='failed'?'failed':''}"><h4>${escapeHtml(ex.test_id)} ${ex.status==='failed'?'❌ ОШИБКА':'✅'}</h4><pre>${escapeHtml(ex.corrected_text||'Нет данных')}</pre></div>`).join('');
            }
        } else {
            corrDiv.innerHTML = last5Corr.map(t => `<div class="example-item ${t.status==='failed'?'failed':''}"><h4>${escapeHtml(t.id)} ${t.status==='failed'?'❌ ОШИБКА':'✅'}</h4><pre>${escapeHtml(t.corrected_text||'Нет данных')}</pre></div>`).join('');
        }

        // Резюме
        if (last5Sum.length === 0) {
            const fallbackSum = (examples && examples.outputs) ? examples.outputs.filter(e => e.summary_text).slice(-5) : [];
            if (fallbackSum.length === 0) {
                sumDiv.innerHTML = '<p>Нет данных</p>';
            } else {
                sumDiv.innerHTML = fallbackSum.map(ex => `<div class="example-item ${ex.status==='failed'?'failed':''}"><h4>${escapeHtml(ex.test_id)} ${ex.status==='failed'?'❌ ОШИБКА':'✅'}</h4><pre>${escapeHtml(ex.summary_text||'Нет данных')}</pre></div>`).join('');
            }
        } else {
            sumDiv.innerHTML = last5Sum.map(t => `<div class="example-item ${t.status==='failed'?'failed':''}"><h4>${escapeHtml(t.id)} ${t.status==='failed'?'❌ ОШИБКА':'✅'}</h4><pre>${escapeHtml(t.summary_text||'Нет данных')}</pre></div>`).join('');
        }
    }

    function isBelow(metricName, val) {
        const thr = THRESHOLDS[metricName];
        if (thr === undefined) return false;
        if (metricName === 'Perplexity') return val > thr;
        return val < thr;
    }

    function formatMetric(value, metricName) {
        if (value === undefined || value === null || value === 'N/A') return 'N/A';
        let num = typeof value === 'number' ? value : parseFloat(String(value).split(' ')[0]);
        if (isNaN(num)) return String(value);
        let fmt = num.toFixed(3);
        if (metricName && isBelow(metricName, num)) return `<span class="below-threshold">${fmt}</span>`;
        return fmt;
    }

    function formatTemp(value) {
        if (value === undefined || value === null || value === 'N/A') return 'N/A';
        let n = parseFloat(String(value));
        return isNaN(n) ? String(value) : n.toFixed(2);
    }

    function formatLLMJudge(value) {
        if (value === undefined || value === null || value === 'N/A') return 'N/A';
        let num = (typeof value === 'string' && value.includes('из')) ? parseFloat(value.split(' ')[0]) : parseFloat(String(value).split(' ')[0]);
        if (isNaN(num)) return String(value);
        let fmt = num.toFixed(1);
        if (isBelow('LLM_Judge', num)) return `<span class="below-threshold">${fmt}</span>`;
        return fmt;
    }

    function formatPromptStats(statsMap) {
        if (!statsMap || Object.keys(statsMap).length === 0) return '—';
        const entries = Object.entries(statsMap).map(([num, count]) => ({ num: parseInt(num), count }));
        entries.sort((a,b) => b.count - a.count);
        return entries.map(e => `${e.num} - №${e.num} : ${e.count} раз`).join('\\n');
    }
    
    function formatTempStats(statsMap) {
        if (!statsMap || Object.keys(statsMap).length === 0) return '—';
        const entries = Object.entries(statsMap).map(([temp, count]) => ({ temp, count }));
        entries.sort((a,b) => b.count - a.count);
        return entries.slice(0, 3).map(e => `t=${e.temp} — ${e.count} раз`).join('\\n');
    }

    function getFormattedDate() {
        const now = new Date();
        const day = String(now.getDate()).padStart(2, '0');
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const year = String(now.getFullYear()).slice(2);
        return `${day}${month}${year}`;
    }

    function exportToExcel() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                const tests = data.tests || [];
                const completedTests = tests.filter(t => t.status === 'completed');
                const failedTests = tests.filter(t => t.status === 'failed');
                const majorityStatus = (completedTests.length > failedTests.length) ? 'completed' : (failedTests.length > completedTests.length ? 'failed' : 'mixed');

                const headers = [
                    'ID', 'Статус', 'ΔWER', 'ΔLev', 'LevRating', 'Perplexity', 'CorScore',
                    'Prompt_cor', 'Temp_cor', 'G-Eval', 'BertScore', 'METEOR', 'LLM-Judge', 'SumScore',
                    'Prompt_sum', 'Temp_sum', 'Длительность (сек)'
                ];

                const rows = [headers];

                for (let test of tests) {
                    const m = test.metrics || {};
                    const perplexityVal = getMetricValue(m, 'perplexity', 0);
                    const promptCor = formatPromptCell(test.prompt_correction_text);
                    const promptSum = formatPromptCell(test.prompt_summary_text);
                    const tempCor = formatTemp(m.best_temp_cor);
                    const tempSum = formatTemp(m.best_temperature_summary);
                    const duration = test.duration && test.duration > 0 ? test.duration.toFixed(2) : '—';

                    rows.push([
                        test.id,
                        test.status,
                        getMetricValue(m, 'delta_WER', 0).toFixed(4),
                        getMetricValue(m, 'delta_LEV', 0).toFixed(4),
                        getMetricValue(m, 'LevRating', 0).toFixed(4),
                        perplexityVal.toFixed(4),
                        getMetricValue(m, 'CorScore', 0).toFixed(4),
                        promptCor,
                        tempCor,
                        getMetricValue(m, 'G-Eval', 0).toFixed(4),
                        getMetricValue(m, 'BertScore', 0).toFixed(4),
                        getMetricValue(m, 'METEOR', 0).toFixed(4),
                        getMetricValue(m, 'LLM_Judge', 0).toFixed(1),
                        getMetricValue(m, 'SumScore', 0).toFixed(4),
                        promptSum,
                        tempSum,
                        duration
                    ]);
                }

                if (completedTests.length > 0 || failedTests.length > 0) {
                    let avg = {
                        delta_WER: 0, delta_LEV: 0, LevRating: 0, Perplexity: 0,
                        CorScore: 0, G_Eval: 0, BertScore: 0, METEOR: 0,
                        LLM_Judge: 0, SumScore: 0
                    };
                    let totalDuration = 0;
                    let durationCount = 0;
                    for (let test of completedTests) {
                        const m = test.metrics || {};
                        avg.delta_WER += getMetricValue(m, 'delta_WER', 0);
                        avg.delta_LEV += getMetricValue(m, 'delta_LEV', 0);
                        avg.LevRating += getMetricValue(m, 'LevRating', 0);
                        avg.Perplexity += getMetricValue(m, 'perplexity', 0);
                        avg.CorScore += getMetricValue(m, 'CorScore', 0);
                        avg.G_Eval += getMetricValue(m, 'G-Eval', 0);
                        avg.BertScore += getMetricValue(m, 'BertScore', 0);
                        avg.METEOR += getMetricValue(m, 'METEOR', 0);
                        avg.LLM_Judge += getMetricValue(m, 'LLM_Judge', 0);
                        avg.SumScore += getMetricValue(m, 'SumScore', 0);
                        if (test.duration && test.duration > 0) {
                            totalDuration += test.duration;
                            durationCount++;
                        }
                    }
                    const cnt = completedTests.length;
                    if (cnt > 0) {
                        for (let key in avg) {
                            avg[key] = (avg[key] / cnt).toFixed(4);
                        }
                    }
                    const avgDuration = (durationCount > 0) ? (totalDuration / durationCount).toFixed(2) : '—';

                    // Подсчёт статистики температур для Excel
                    let tempCorStatsExcel = {};
                    let tempSumStatsExcel = {};
                    for (let test of completedTests) {
                        const m = test.metrics || {};
                        const tc = m.best_temp_cor;
                        if (tc !== undefined && tc !== null && tc !== 'N/A' && tc !== '') {
                            let tv = parseFloat(String(tc));
                            if (!isNaN(tv)) {
                                const key = tv.toFixed(2);
                                tempCorStatsExcel[key] = (tempCorStatsExcel[key] || 0) + 1;
                            }
                        }
                        const ts = m.best_temperature_summary;
                        if (ts !== undefined && ts !== null && ts !== 'N/A' && ts !== '') {
                            let tv = parseFloat(String(ts));
                            if (!isNaN(tv)) {
                                const key = tv.toFixed(2);
                                tempSumStatsExcel[key] = (tempSumStatsExcel[key] || 0) + 1;
                            }
                        }
                    }

                    rows.push([
                        'СРЕДНИЕ', majorityStatus,
                        avg.delta_WER, avg.delta_LEV, avg.LevRating, avg.Perplexity, avg.CorScore,
                        '—', formatTempStats(tempCorStatsExcel).split(String.fromCharCode(10)).join('; '), avg.G_Eval, avg.BertScore, avg.METEOR, avg.LLM_Judge, avg.SumScore,
                        '—', formatTempStats(tempSumStatsExcel).split(String.fromCharCode(10)).join('; '), avgDuration
                    ]);
                }

                const ws = XLSX.utils.aoa_to_sheet(rows);
                const wb = XLSX.utils.book_new();
                XLSX.utils.book_append_sheet(wb, ws, 'Метрики');
                const fileName = `Таблица метрик_${getFormattedDate()}.xlsx`;
                XLSX.writeFile(wb, fileName);
            })
            .catch(e => console.error('Ошибка экспорта:', e));
    }

    function updateTable(tests) {
        const tbody = document.getElementById('tests-body');
        if (!tests || tests.length === 0) { tbody.innerHTML = '<tr><td colspan="17">Нет данных</td></tr>'; return; }

        const completedTests = tests.filter(t => t.status === 'completed');
        const failedTests = tests.filter(t => t.status === 'failed');
        const majorityStatus = (completedTests.length > failedTests.length) ? 'completed' : (failedTests.length > completedTests.length ? 'failed' : 'mixed');

        let avg = {
            delta_WER: 0, delta_LEV: 0, LevRating: 0, Perplexity: 0,
            CorScore: 0, G_Eval: 0, BertScore: 0, METEOR: 0,
            LLM_Judge: 0, SumScore: 0
        };
        let promptCorStats = {};
        let promptSumStats = {};
        let tempCorStats = {};
        let tempSumStats = {};

        let totalDuration = 0;
        let durationCount = 0;

        if (completedTests.length > 0) {
            for (let test of completedTests) {
                const m = test.metrics || {};
                avg.delta_WER += getMetricValue(m, 'delta_WER', 0);
                avg.delta_LEV += getMetricValue(m, 'delta_LEV', 0);
                avg.LevRating += getMetricValue(m, 'LevRating', 0);
                avg.Perplexity += getMetricValue(m, 'perplexity', 0);
                avg.CorScore += getMetricValue(m, 'CorScore', 0);
                avg.G_Eval += getMetricValue(m, 'G-Eval', 0);
                avg.BertScore += getMetricValue(m, 'BertScore', 0);
                avg.METEOR += getMetricValue(m, 'METEOR', 0);
                avg.LLM_Judge += getMetricValue(m, 'LLM_Judge', 0);
                avg.SumScore += getMetricValue(m, 'SumScore', 0);

                if (test.duration && test.duration > 0) {
                    totalDuration += test.duration;
                    durationCount++;
                }

                const promptCorText = test.prompt_correction_text;
                if (promptCorText) {
                    let num = getPromptNumber(promptCorText);
                    if (num) promptCorStats[num] = (promptCorStats[num] || 0) + 1;
                }
                const promptSumText = test.prompt_summary_text;
                if (promptSumText) {
                    let num = getPromptNumber(promptSumText);
                    if (num) promptSumStats[num] = (promptSumStats[num] || 0) + 1;
                }

                // Подсчёт статистики температур
                const tc = m.best_temp_cor;
                if (tc !== undefined && tc !== null && tc !== 'N/A' && tc !== '') {
                    let tv = parseFloat(String(tc));
                    if (!isNaN(tv)) {
                        const key = tv.toFixed(2);
                        tempCorStats[key] = (tempCorStats[key] || 0) + 1;
                    }
                }
                const ts = m.best_temperature_summary;
                if (ts !== undefined && ts !== null && ts !== 'N/A' && ts !== '') {
                    let tv = parseFloat(String(ts));
                    if (!isNaN(tv)) {
                        const key = tv.toFixed(2);
                        tempSumStats[key] = (tempSumStats[key] || 0) + 1;
                    }
                }
            }
            const cnt = completedTests.length;
            for (let key in avg) {
                avg[key] = (avg[key] / cnt).toFixed(4);
            }
        }
        const avgDuration = (durationCount > 0) ? (totalDuration / durationCount).toFixed(2) : '—';
        const majorityStatusDisplay = majorityStatus === 'completed' ? '✅ completed' : (majorityStatus === 'failed' ? '❌ failed' : '⚖️ mixed');

        // ✅ Natural sort по ID (1_, 2_, 10_, 20_, 100_ вместо 1_, 10_, 100_, 2_, 20_)
        function naturalSortKey(s) {
            const parts = s.split(/(\d+)/);
            return parts.map(p => { const n = parseInt(p); return isNaN(n) ? p.toLowerCase() : n; });
        }
        const sorted = [...tests].sort((a,b) => {
            const keyA = naturalSortKey(a.id);
            const keyB = naturalSortKey(b.id);
            for (let i = 0; i < Math.min(keyA.length, keyB.length); i++) {
                if (keyA[i] < keyB[i]) return -1;
                if (keyA[i] > keyB[i]) return 1;
            }
            return keyA.length - keyB.length;
        });

        let rows = '';
        for (let test of sorted) {
            const m = test.metrics || {};
            const statusClass = `status-${test.status}`;
            const rowClass = test.status === 'failed' ? 'failed-row' : '';
            const duration = test.duration;
            const perplexityVal = getMetricValue(m, 'perplexity', 0);
            const promptCorDisplay = formatPromptCell(test.prompt_correction_text);
            const promptSumDisplay = formatPromptCell(test.prompt_summary_text);
            const tempCor = formatTemp(m.best_temp_cor);
            const tempSum = formatTemp(m.best_temperature_summary);
            rows += `<tr class="${rowClass}">
                <td>${escapeHtml(test.id)}</td>
                <td><span class="status-badge ${statusClass}">${test.status}</span></td>
                <td>${formatMetric(m.delta_WER, 'delta_WER')}</td>
                <td>${formatMetric(m.delta_LEV, 'delta_LEV')}</td>
                <td>${formatMetric(m.LevRating, 'LevRating')}</td>
                <td>${formatMetric(perplexityVal, 'Perplexity')}</td>
                <td>${formatMetric(m.CorScore, 'CorScore')}</td>
                <td>${promptCorDisplay}</td>
                <td>${tempCor}</td>
                <td>${formatMetric(m['G-Eval'], 'G-Eval')}</td>
                <td>${formatMetric(m.BertScore, 'BertScore')}</td>
                <td>${formatMetric(m.METEOR, 'METEOR')}</td>
                <td>${formatLLMJudge(m.LLM_Judge)}</td>
                <td>${formatMetric(m.SumScore, 'SumScore')}</td>
                <td>${promptSumDisplay}</td>
                <td>${tempSum}</td>
                <td>${formatDuration(duration)}</td>
              </tr>`;
        }

        if (completedTests.length > 0) {
            const durationInfo = (durationCount > 0) ? `${avgDuration} сек (по ${durationCount} тестам)` : '—';
            rows += `<tr class="average-row">
                <td><strong>📊 СРЕДНИЕ</strong><br>(${completedTests.length} тестов)</td>
                <td><strong>${majorityStatusDisplay}</strong></td>
                <td><strong>${avg.delta_WER}</strong></td>
                <td><strong>${avg.delta_LEV}</strong></td>
                <td><strong>${avg.LevRating}</strong></td>
                <td><strong>${avg.Perplexity}</strong></td>
                <td><strong>${avg.CorScore}</strong></td>
                <td style="white-space: pre-line;"><strong>${formatPromptStats(promptCorStats)}</strong></td>
                <td style="white-space: pre-line;"><strong>${formatTempStats(tempCorStats)}</strong></td>
                <td><strong>${avg.G_Eval}</strong></td>
                <td><strong>${avg.BertScore}</strong></td>
                <td><strong>${avg.METEOR}</strong></td>
                <td><strong>${avg.LLM_Judge}</strong></td>
                <td><strong>${avg.SumScore}</strong></td>
                <td style="white-space: pre-line;"><strong>${formatPromptStats(promptSumStats)}</strong></td>
                <td style="white-space: pre-line;"><strong>${formatTempStats(tempSumStats)}</strong></td>
                <td><strong>${durationInfo}</strong></td>
              </tr>`;
        }

        tbody.innerHTML = rows;
    }

    function escapeHtml(str) { if (!str) return ''; return str.replace(/[&<>]/g, m => m==='&'?'&amp;':m==='<'?'&lt;':m==='>'?'&gt;':m); }

    document.getElementById('exportExcelBtn').addEventListener('click', exportToExcel);
    setInterval(refreshData, UPDATE_INTERVAL);
    refreshData();
</script>
</body>
</html>
"""

web_monitor = None


def get_web_monitor():
    return web_monitor


def init_web_monitor(host: str = "127.0.0.1", port: int = 5000) -> WebMonitor:
    global web_monitor
    web_monitor = WebMonitor(host, port)
    return web_monitor