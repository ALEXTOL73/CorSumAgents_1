#!/usr/bin/env python3
"""
Ансамбль для суммаризации текста
Версия 1.17 - Усилено влияние few-shot и CoT, добавлены в основной ансамбль
"""
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory
from metrics.meteor_calculator import METEORCalculator
from utils.text_postprocessor import TextPostprocessor
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
from collections import Counter


class SummarizerEnsemble(BaseAgent):
    BASE_PROMPT = """Ты профессиональный суммаризатор. Создай краткое резюме текста.

ТРЕБОВАНИЯ:
- Сохрани ключевые факты и основную мысль
- ОБЯЗАТЕЛЬНО сохрани: числа, даты, имена собственные, названия организаций
- Не используй обобщающих фраз («в тексте говорится», «автор отмечает»)
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
        self.logger.info(f"[SummarizerEnsemble] Инициализирован v1.17 (усилен few-shot/CoT)")

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

    def _majority_vote(self, variants: List[str]) -> str:
        if len(variants) < 2:
            return variants[0] if variants else ""
        all_tokens = []
        tokenized = []
        for v in variants:
            tokens = v.lower().split()
            tokenized.append(tokens)
            all_tokens.extend(tokens)
        freq = Counter(all_tokens)
        best_idx = 0
        best_score = -1
        for i, tokens in enumerate(tokenized):
            unique_tokens = set(tokens)
            score = sum(freq[t] for t in unique_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
        return variants[best_idx]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        self.log_execution("Запуск ансамбля суммаризации")

        prompt_template = state.get("prompt_summary", "") or self.BASE_PROMPT
        prompt_template = self._ensure_text_placeholder(prompt_template)

        input_text = state.get("corrected_text", "") or state.get("input_text", "")
        reference_summary = state.get("reference_summary", "")
        domain = state.get("summary_domain", "general")

        if not input_text:
            self.logger.error("[SummarizerEnsemble] Пустой входной текст")
            return self._create_empty_result()

        temperatures = self._compute_dynamic_temperatures(input_text)
        self.logger.info(f"[SummarizerEnsemble] Динамические температуры: {temperatures}")

        # ✅ Генерируем варианты с разными промптами
        variants, temps, prompts_list = self._generate_variants_with_prompts(
            prompt_template, input_text, temperatures, reference_summary, domain
        )

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

        # Majority vote
        majority_variant = self._majority_vote([v for v in variants if v])
        majority_metrics = None
        if majority_variant:
            majority_metrics = self._call_judge_for_metrics(input_text, majority_variant, reference_summary)
            print(f"  Majority vote: SumScore={majority_metrics.get('SumScore', 0):.4f}")

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
            prompt_type = "базовый"
            if "ПРИМЕРЫ" in prompts_list[i]:
                prompt_type = "few-shot"
            elif "ШАГ" in prompts_list[i]:
                prompt_type = "CoT"
            elif "сохранённый" in prompts_list[i].lower():
                prompt_type = "saved"

            print(f"  Вариант #{i+1} (temp={temp:.2f}, {prompt_type}):")
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

        if majority_metrics and majority_metrics.get("SumScore", 0) > best_sumscore:
            print(f"  ✅ Majority vote лучше (SumScore={majority_metrics.get('SumScore', 0):.4f} > {best_sumscore:.4f})")
            best_result = self._create_result(majority_variant, 0.5, "majority_vote", reference_summary, input_text, majority_metrics)
        elif best_sumscore >= 0:
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

    def _generate_variants_with_prompts(self, base_prompt: str, input_text: str,
                                        temperatures: List[float], reference_summary: str,
                                        domain: str) -> Tuple[List[str], List[float], List[str]]:
        variants, temps_used, prompts_used = [], [], []

        # 1. Базовый промпт с разными температурами
        for temp in temperatures:
            try:
                full_prompt = base_prompt.format(text=input_text)
                response = self._generate_with_prompt(full_prompt, temp)
                if response:
                    variants.append(response)
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
                else:
                    variants.append("")
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
            except Exception as e:
                self.logger.warning(f"[SummarizerEnsemble] Ошибка базового промпта (temp={temp}): {e}")
                variants.append("")
                temps_used.append(temp)
                prompts_used.append(base_prompt)

        # 2. Few-shot промпт (низкая температура)
        if SUMMARY_USE_FEW_SHOT and self.memory:
            try:
                examples = self.memory.get_summary_few_shot_examples(input_text, domain, max_examples=2, length_ratio=0.2)
                few_shot_prompt = self._build_few_shot_prompt(examples, input_text)
                response = self._generate_with_prompt(few_shot_prompt, 0.3)
                if response:
                    variants.append(response)
                    temps_used.append(0.3)
                    prompts_used.append(few_shot_prompt)
                    self.logger.info("[SummarizerEnsemble] Few-shot вариант добавлен")
            except Exception as e:
                self.logger.warning(f"[SummarizerEnsemble] Ошибка few-shot: {e}")

        # 3. CoT промпт
        if SUMMARY_USE_CHAIN_OF_THOUGHT:
            try:
                cot_prompt = self.CHAIN_OF_THOUGHT_PROMPT.format(text=input_text)
                response = self._generate_with_prompt(cot_prompt, 0.5)
                if response:
                    variants.append(response)
                    temps_used.append(0.5)
                    prompts_used.append(cot_prompt)
                    self.logger.info("[SummarizerEnsemble] CoT вариант добавлен")
            except Exception as e:
                self.logger.warning(f"[SummarizerEnsemble] Ошибка CoT: {e}")

        # 4. Сохранённые промпты
        if SUMMARY_USE_SAVED_PROMPTS and self.saved_prompts:
            for i, saved in enumerate(self.saved_prompts[:2]):
                try:
                    saved_prompt = self._ensure_text_placeholder(saved.get("prompt", base_prompt))
                    full_prompt = saved_prompt.format(text=input_text)
                    response = self._generate_with_prompt(full_prompt, 0.4)
                    if response:
                        variants.append(response)
                        temps_used.append(0.4)
                        prompts_used.append(full_prompt)
                        self.logger.info(f"[SummarizerEnsemble] Сохранённый промпт #{i+1} добавлен")
                except Exception as e:
                    self.logger.warning(f"[SummarizerEnsemble] Ошибка saved промпта {i}: {e}")

        return variants, temps_used, prompts_used

    def _generate_with_prompt(self, full_prompt: str, temperature: float) -> Optional[str]:
        try:
            response = self.client.generate(
                prompt=full_prompt,
                temperature=temperature,
                system_prompt="Ты суммаризатор. Верни только резюме из 1-4 предложений, учитывая целевую аудиторию."
            )
            if response and len(response.strip()) >= 10:
                cleaned = TextPostprocessor.clean_text(response.strip())
                return cleaned
        except Exception as e:
            self.logger.warning(f"[SummarizerEnsemble] Ошибка генерации: {e}")
        return None

    def _build_few_shot_prompt(self, examples: List[Dict], input_text: str) -> str:
        example_text = "ПРИМЕРЫ ХОРОШИХ РЕЗЮМЕ:\n"
        for i, ex in enumerate(examples, 1):
            example_text += f"Пример {i}:\nВход: {ex['input']}\nВыход: {ex['output']}\n\n"
        prompt = f"""{example_text}
ТЕПЕРЬ СДЕЛАЙ РЕЗЮМЕ ЭТОГО ТЕКСТА:
{input_text}

РЕЗЮМЕ:"""
        return prompt

    def _adaptive_retry(self, current_best: Dict[str, Any], base_prompt: str,
                        input_text: str, reference: str) -> Dict[str, Any]:
        best_result = current_best
        best_sumscore = current_best.get("metrics_summary", {}).get("SumScore", 0)
        attempts = 0
        base_prompt = self._ensure_text_placeholder(base_prompt)
        strategies = []

        # Изменён порядок: сначала few-shot и CoT
        if SUMMARY_USE_FEW_SHOT:
            strategies.append(("few_shot", self.FEW_SHOT_PROMPT, 0.3))
        if SUMMARY_USE_CHAIN_OF_THOUGHT:
            strategies.append(("cot", self.CHAIN_OF_THOUGHT_PROMPT, 0.5))
        for temp in SUMMARY_RETRY_TEMPS:
            strategies.append(("temp_retry", base_prompt, temp))
        if SUMMARY_USE_SAVED_PROMPTS and self.saved_prompts:
            for i, saved in enumerate(self.saved_prompts[:3]):
                strategies.append((f"saved_{i}", self._ensure_text_placeholder(saved.get("prompt", base_prompt)), 0.5))

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