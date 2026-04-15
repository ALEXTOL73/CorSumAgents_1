#!/usr/bin/env python3
"""
Ансамбль для суммаризации текста
Версия 1.13 - Добавлена пост-обработка результатов
"""
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory
from metrics.meteor_calculator import METEORCalculator
from utils.text_postprocessor import TextPostprocessor  # ✅ добавлен пост-процессор
from config import (
    MODEL_NAME, SUMMARY_TEMPERATURE_RANGE, SUMMARY_MAX_RETRY_ATTEMPTS,
    SUMMARY_RETRY_TEMPS, SUMMARY_USE_SAVED_PROMPTS, SUMMARY_USE_FEW_SHOT,
    SUMMARY_USE_CHAIN_OF_THOUGHT, SUMMARY_SELF_CONSISTENCY_ENABLED,
    SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT, SUMMARY_DYNAMIC_TEMPERATURES,
    BERTSCORE_ENABLED, SUMSCORE_WEIGHTS
)
import json
from pathlib import Path
import time


class SummarizerEnsemble(BaseAgent):
    BASE_PROMPT = """Ты профессиональный суммаризатор. Создай краткое резюме текста.

ТРЕБОВАНИЯ:
- Сохрани ключевые факты и основную мысль
- Объём: 1-4 предложения
- Не добавляй новую информацию
- Используй тот же язык, что и текст
- ✅ УЧИТЫВАЙ ЦЕЛЕВУЮ АУДИТОРИЮ: предполагается, что резюме будет читать человек, которому нужна суть без лишних деталей
- Верни только резюме

ТЕКСТ ДЛЯ СУММАРИЗАЦИИ:
{text}

РЕЗЮМЕ:"""

    FEW_SHOT_PROMPT = """Ты профессиональный суммаризатор. Создай краткое резюме текста, учитывая целевую аудиторию.

ПРИМЕРЫ:
Вход: "Иван Иванов родился в 1990 году в Москве. Он окончил МГУ и работает программистом. Увлекается спортом."
Выход: "Иван Иванов (1990 г.р., Москва) окончил МГУ, работает программистом, увлекается спортом."

Вход: "Сегодня на заседании правительства обсуждались вопросы экономического развития. Министр финансов представил новый бюджет."
Выход: "Правительство обсудило экономическое развитие, министр финансов представил новый бюджет."

ТЕПЕРЬ СДЕЛАЙ РЕЗЮМЕ ЭТОГО ТЕКСТА, ПРЕДНАЗНАЧЕННОЕ ДЛЯ ШИРОКОЙ АУДИТОРИИ:
{text}

РЕЗЮМЕ:"""

    CHAIN_OF_THOUGHT_PROMPT = """Ты профессиональный суммаризатор. Создай краткое резюме текста, учитывая целевую аудиторию.

ШАГ 1: Определи основную тему текста
ШАГ 2: Выдели ключевые факты, важные для широкой аудитории
ШАГ 3: Сформулируй резюме из 1-4 предложений, избегая узкоспециализированных терминов
ШАГ 4: Проверь, что не добавлено лишнего и резюме понятно неспециалисту

ТЕКСТ:
{text}

РЕЗЮМЕ:"""

    def __init__(self, client: LMStudioClient, memory: Optional[AgentMemory] = None):
        super().__init__(client, "SummarizerEnsemble")
        self.model_name = MODEL_NAME
        self.memory = memory
        self.meteor_calc = METEORCalculator()
        self.saved_prompts = self._load_saved_prompts()
        if BERTSCORE_ENABLED:
            from metrics.bertscore_calculator import BertScoreCalculator
            self.bert_calc = BertScoreCalculator()
        else:
            self.bert_calc = None
        self.logger.info(f"[SummarizerEnsemble] Инициализирован v1.13 (BertScore: {'вкл' if BERTSCORE_ENABLED else 'выкл'}, пост-обработка)")
        self.logger.info(f"[SummarizerEnsemble] Self-consistency: {SUMMARY_SELF_CONSISTENCY_ENABLED}")
        self.logger.info(f"[SummarizerEnsemble] Динамические температуры: {SUMMARY_DYNAMIC_TEMPERATURES}")

    def _load_saved_prompts(self) -> List[Dict[str, Any]]:
        if not self.memory or not SUMMARY_USE_SAVED_PROMPTS:
            return []
        try:
            prompts_file = Path("data/memory/best_summary_prompts.json")
            if prompts_file.exists():
                with open(prompts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompts = []
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict):
                    for domain, domain_prompts in data.items():
                        if isinstance(domain_prompts, list):
                            prompts.extend(domain_prompts)
                prompts.sort(key=lambda x: x.get("sumscore", 0), reverse=True)
                self.logger.info(f"[SummarizerEnsemble] Загружено {len(prompts)} сохранённых промптов")
                return prompts
        except Exception as e:
            self.logger.warning(f"[SummarizerEnsemble] Ошибка загрузки промптов: {e}")
        return []

    def _ensure_text_placeholder(self, prompt: str) -> str:
        if "{text}" not in prompt:
            prompt += "\n\nТЕКСТ:\n{text}\n\nРЕЗЮМЕ:"
        return prompt

    def _call_judge_for_metrics(self, original: str, summary: str, reference: str) -> Dict[str, float]:
        """Вызов судьи через execute для получения полных метрик (включая BertScore)"""
        try:
            from agents.summarization_judge import SummarizationJudge
            judge = SummarizationJudge(self.client)
            state = {
                "corrected_text": original,
                "summary_text": summary,
                "reference_summary": reference,
                "input_text": original,
                "retry_count_summary": 0
            }
            result = judge.execute(state)
            metrics = result.get("metrics_summary", {})
            g_eval = metrics.get("g_eval_overall", 0.5)
            llm_judge = metrics.get("llm_score", 5.0)
            meteor = metrics.get("meteor", 0.5)
            bertscore = metrics.get("bertscore", 0.0)
            sumscore = metrics.get("sumscore", 0.0)
        except Exception as e:
            self.logger.warning(f"[SummarizerEnsemble] Ошибка вызова судьи: {e}, используем значения по умолчанию")
            g_eval = 0.5
            llm_judge = 5.0
            meteor = 0.5
            bertscore = 0.0
            sumscore = 0.0

        return {
            "G_Eval": g_eval,
            "LLM_Judge": llm_judge,
            "METEOR": meteor,
            "BertScore": bertscore,
            "SumScore": sumscore
        }

    def _compute_dynamic_temperatures(self, input_text: str) -> List[float]:
        if not SUMMARY_DYNAMIC_TEMPERATURES:
            return [
                SUMMARY_TEMPERATURE_RANGE[0],
                (SUMMARY_TEMPERATURE_RANGE[0] + SUMMARY_TEMPERATURE_RANGE[1]) / 2,
                SUMMARY_TEMPERATURE_RANGE[1]
            ]
        length = len(input_text)
        if length > 2000:
            return [0.5, 0.7, 0.9]
        elif length > 1000:
            return [0.4, 0.6, 0.8]
        else:
            return [0.3, 0.5, 0.7]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        self.log_execution("Запуск ансамбля суммаризации")

        prompt_template = state.get("prompt_summary", "") or self.BASE_PROMPT
        prompt_template = self._ensure_text_placeholder(prompt_template)

        input_text = state.get("corrected_text", "") or state.get("input_text", "")
        reference_summary = state.get("reference_summary", "")

        if not input_text:
            self.logger.error("[SummarizerEnsemble] Пустой входной текст")
            return self._create_empty_result()

        temperatures = self._compute_dynamic_temperatures(input_text)
        self.logger.info(f"[SummarizerEnsemble] Динамические температуры: {temperatures}")

        variants, temps, prompts_list = self._generate_variants(prompt_template, input_text, temperatures)

        if SUMMARY_SELF_CONSISTENCY_ENABLED and len(variants) >= 2:
            best_temp = temps[0] if temps else 0.5
            for _ in range(SUMMARY_SELF_CONSISTENCY_EXTRA_COUNT):
                try:
                    full_prompt = prompt_template.format(text=input_text)
                    response = self.client.generate(
                        prompt=full_prompt,
                        temperature=best_temp,
                        system_prompt="Ты суммаризатор. Верни только резюме из 1-4 предложений, учитывая целевую аудиторию."
                    )
                    if response and len(response.strip()) >= 10:
                        # ✅ пост-обработка
                        cleaned = TextPostprocessor.clean_text(response.strip())
                        variants.append(cleaned)
                        temps.append(best_temp)
                        prompts_list.append(full_prompt)
                except Exception as e:
                    self.logger.warning(f"[SummarizerEnsemble] Self-consistency ошибка: {e}")

        while len(variants) < 3:
            variants.append("")
            temps.append(0.5)
            prompts_list.append(prompt_template)

        state["summary_outputs"] = variants.copy()
        state["summary_metrics_list"] = []

        best_result = None
        best_sumscore = -1.0
        best_idx = 0
        best_metrics = None

        print("\n" + "-" * 80)
        print("  📊 ОЦЕНКА ВАРИАНТОВ СУММАРИЗАЦИИ (РЕАЛЬНЫЕ МЕТРИКИ)")
        print("-" * 80)

        for i, variant in enumerate(variants):
            if not variant:
                state["summary_metrics_list"].append({"SumScore": 0})
                print(f"  Вариант #{i+1}: пустой, пропускаем")
                continue

            metrics = self._call_judge_for_metrics(input_text, variant, reference_summary)
            state["summary_metrics_list"].append(metrics)
            sumscore = metrics.get("SumScore", 0)
            g_eval = metrics.get("G_Eval", 0)
            meteor = metrics.get("METEOR", 0)
            llm_judge = metrics.get("LLM_Judge", 0)
            bertscore = metrics.get("BertScore", 0)
            temp = temps[i] if i < len(temps) else 0.5

            print(f"  Вариант #{i+1} (temp={temp:.2f}):")
            print(f"     └─ SumScore: {sumscore:.4f}, G-Eval: {g_eval:.4f}")
            print(f"     └─ METEOR: {meteor:.4f}, LLM-Judge: {llm_judge:.1f}")
            if BERTSCORE_ENABLED:
                print(f"     └─ BertScore: {bertscore:.4f}")

            if sumscore > best_sumscore:
                best_sumscore = sumscore
                best_variant = variant
                best_idx = i
                best_temp = temp
                best_prompt = prompts_list[i]
                best_metrics = metrics

        print("-" * 80)
        if best_sumscore >= 0:
            print(f"  ✅ Лучший вариант #{best_idx+1} (temp={best_temp:.2f}, SumScore={best_sumscore:.4f})")
            best_result = self._create_result(best_variant, best_temp, best_prompt, reference_summary, input_text, best_metrics)
        else:
            print(f"  ⚠️ Нет валидных вариантов, используется пустое резюме")
            best_result = self._create_empty_result()
        print("-" * 80 + "\n")

        if best_result.get("metrics_summary", {}).get("SumScore", 0) < 0.5:
            self.logger.warning(f"[SummarizerEnsemble] SumScore низкий, запуск адаптивных попыток")
            best_result = self._adaptive_retry(best_result, prompt_template, input_text, reference_summary)

        if self.memory and reference_summary:
            try:
                self.memory.learn_from_summarization(
                    original=input_text,
                    summary=best_result["summary_text"],
                    reference=reference_summary,
                    prompt_used=prompt_template,
                    model_used=self.model_name,
                    metrics=best_result.get("metrics_summary", {})
                )
            except Exception as e:
                self.logger.error(f"[SummarizerEnsemble] Ошибка сохранения в память: {e}")

        best_result["summary_outputs"] = variants
        elapsed = time.time() - start_time
        self.logger.info(f"[SummarizerEnsemble] Завершено за {elapsed:.2f} сек")
        return best_result

    def _generate_variants(self, prompt_template: str, input_text: str,
                          temperatures: List[float]) -> Tuple[List[str], List[float], List[str]]:
        variants, temps_used, prompts_used = [], [], []
        for temp in temperatures:
            try:
                full_prompt = prompt_template.format(text=input_text)
                response = self.client.generate(
                    prompt=full_prompt,
                    temperature=temp,
                    system_prompt="Ты суммаризатор. Верни только резюме из 1-4 предложений, учитывая целевую аудиторию."
                )
                if response and len(response.strip()) >= 10:
                    # ✅ пост-обработка
                    cleaned = TextPostprocessor.clean_text(response.strip())
                    variants.append(cleaned)
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
                else:
                    variants.append("")
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
            except Exception as e:
                self.logger.warning(f"[SummarizerEnsemble] Ошибка генерации при temp={temp}: {e}")
                variants.append("")
                temps_used.append(temp)
                prompts_used.append(prompt_template)
        return variants, temps_used, prompts_used

    def _adaptive_retry(self, current_best: Dict[str, Any], base_prompt: str,
                        input_text: str, reference: str) -> Dict[str, Any]:
        best_result = current_best
        best_sumscore = current_best.get("metrics_summary", {}).get("SumScore", 0)
        attempts = 0
        base_prompt = self._ensure_text_placeholder(base_prompt)

        strategies = []
        for temp in SUMMARY_RETRY_TEMPS:
            strategies.append(("temp_retry", base_prompt, temp))

        if SUMMARY_USE_SAVED_PROMPTS and self.saved_prompts:
            for i, saved in enumerate(self.saved_prompts[:3]):
                strategies.append((f"saved_{i}", self._ensure_text_placeholder(saved.get("prompt", base_prompt)), 0.5))

        if SUMMARY_USE_FEW_SHOT:
            strategies.append(("few_shot", self._ensure_text_placeholder(self.FEW_SHOT_PROMPT), 0.5))

        if SUMMARY_USE_CHAIN_OF_THOUGHT:
            strategies.append(("cot", self._ensure_text_placeholder(self.CHAIN_OF_THOUGHT_PROMPT), 0.3))

        print("\n" + "=" * 80)
        print("  🔄 АДАПТИВНАЯ СУММАРИЗАЦИЯ (SumScore < 0.5)")
        print("=" * 80)

        for name, prompt, temp in strategies:
            if attempts >= SUMMARY_MAX_RETRY_ATTEMPTS:
                break
            print(f"\n  Попытка {attempts+1}: стратегия '{name}' (temp={temp})")
            try:
                full_prompt = prompt.format(text=input_text)
                response = self.client.generate(
                    prompt=full_prompt,
                    temperature=temp,
                    system_prompt="Ты суммаризатор. Верни только резюме из 1-4 предложений, учитывая целевую аудиторию."
                )
                if not response or len(response.strip()) < 10:
                    attempts += 1
                    continue
                # ✅ пост-обработка
                cleaned = TextPostprocessor.clean_text(response.strip())
                metrics = self._call_judge_for_metrics(input_text, cleaned, reference)
                sumscore = metrics.get("SumScore", 0)
                print(f"     └─ SumScore={sumscore:.4f}")
                if sumscore > 0.7:
                    print(f"     └─ ✅ Успех! SumScore > 0.7")
                    return self._create_result(cleaned, temp, full_prompt, reference, input_text, metrics)
                if sumscore > best_sumscore:
                    print(f"     └─ Новый лучший результат (SumScore={sumscore:.4f})")
                    best_sumscore = sumscore
                    best_result = self._create_result(cleaned, temp, full_prompt, reference, input_text, metrics)
            except Exception as e:
                self.logger.warning(f"[SummarizerEnsemble] Ошибка в стратегии {name}: {e}")
            attempts += 1

        print("=" * 80 + "\n")
        return best_result

    def _create_result(self, summary_text: str, temperature: float, prompt: str,
                       reference: str, original: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        # ✅ пост-обработка финального резюме
        cleaned = TextPostprocessor.clean_text(summary_text)
        return {
            "summary_text": cleaned,
            "summary_outputs": [],
            "prompt_summary": prompt,
            "best_temperature_summary": str(temperature),
            "best_prompt_summary_type": "",
            "best_model": self.model_name,
            "metrics_summary": metrics
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        return {
            "summary_text": "",
            "summary_outputs": [],
            "prompt_summary": "",
            "best_temperature_summary": "N/A",
            "best_prompt_summary_type": "",
            "best_model": self.model_name,
            "metrics_summary": {}
        }

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] - {message}")