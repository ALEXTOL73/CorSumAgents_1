#!/usr/bin/env python3
"""
Агент суммаризации на базе ансамбля LLM
Версия 5.6.1 - Исправлен вызов судьи для получения BertScore
"""
import json
from typing import Dict, Any, List, Optional, Tuple

from agents.base_agent import BaseAgent
from config import (
    MODEL_NAME, MAX_LEV_RETRY_ATTEMPTS,
    USE_SAVED_PROMPTS, USE_FEW_SHOT_PROMPT, USE_CHAIN_OF_THOUGHT_PROMPT,
    BERTSCORE_ENABLED, LANGUAGE
)
from metrics.meteor_calculator import METEORCalculator
from utils.agent_memory import AgentMemory
from utils.lmstudio_client import LMStudioClient


class Summarizer(BaseAgent):
    """Ансамбль для суммаризации текста с корректным BertScore и SumScore"""

    BASE_PROMPT_RU = """Ты профессиональный суммаризатор. Создай краткое резюме текста.

ТРЕБОВАНИЯ:
- Сохрани ключевые факты и основную мысль
- Объём: 1-4 предложения
- Не добавляй новую информацию
- Используй тот же язык, что и текст
- Верни только резюме

ТЕКСТ ДЛЯ СУММАРИЗАЦИИ:
{text}

РЕЗЮМЕ:"""

    BASE_PROMPT_EN = """You are a professional summarizer. Create a brief summary of the text.

REQUIREMENTS:
- Preserve key facts and main idea
- Length: 1-4 sentences
- Do not add new information
- Use the same language as the text
- Return only the summary

TEXT FOR SUMMARIZATION:
{text}

SUMMARY:"""

    FEW_SHOT_PROMPT_RU = """Ты профессиональный суммаризатор. Создай краткое резюме текста.

ПРИМЕРЫ:
Вход: "Иван Иванов родился в 1990 году в Москве. Он окончил МГУ и работает программистом. Увлекается спортом."
Выход: "Иван Иванов (1990 г.р., Москва) окончил МГУ, работает программистом, увлекается спортом."

Вход: "Сегодня на заседании правительства обсуждались вопросы экономического развития. Министр финансов представил новый бюджет."
Выход: "Правительство обсудило экономическое развитие, министр финансов представил новый бюджет."

ТЕПЕРЬ СДЕЛАЙ РЕЗЮМЕ ЭТОГО ТЕКСТА:
{text}

РЕЗЮМЕ:"""

    FEW_SHOT_PROMPT_EN = """You are a professional summarizer. Create a brief summary of the text.

EXAMPLES:
Input: "John Smith was born in 1985 in New York. He graduated from MIT and works as a software engineer."
Output: "John Smith (b. 1985, New York) graduated from MIT and works as a software engineer."

Input: "The government meeting discussed economic development. The finance minister presented the new budget."
Output: "The government discussed economic development and presented the new budget."

NOW SUMMARIZE THIS TEXT:
{text}

SUMMARY:"""

    CHAIN_OF_THOUGHT_PROMPT_RU = """Ты профессиональный суммаризатор. Создай краткое резюме текста.

ШАГ 1: Определи основную тему текста
ШАГ 2: Выдели ключевые факты
ШАГ 3: Сформулируй резюме из 1-4 предложений
ШАГ 4: Проверь, что не добавлено лишнего

ТЕКСТ:
{text}

РЕЗЮМЕ:"""

    CHAIN_OF_THOUGHT_PROMPT_EN = """You are a professional summarizer. Create a brief summary of the text.

STEP 1: Identify the main topic
STEP 2: Extract key facts
STEP 3: Formulate a 1-4 sentence summary
STEP 4: Verify nothing extra was added

TEXT:
{text}

SUMMARY:"""

    @property
    def BASE_PROMPT(self):
        return self.BASE_PROMPT_EN if LANGUAGE.lower() == 'en' else self.BASE_PROMPT_RU

    @property
    def FEW_SHOT_PROMPT(self):
        return self.FEW_SHOT_PROMPT_EN if LANGUAGE.lower() == 'en' else self.FEW_SHOT_PROMPT_RU

    @property
    def CHAIN_OF_THOUGHT_PROMPT(self):
        return self.CHAIN_OF_THOUGHT_PROMPT_EN if LANGUAGE.lower() == 'en' else self.CHAIN_OF_THOUGHT_PROMPT_RU

    def __init__(self, client: LMStudioClient, memory: Optional[AgentMemory] = None):
        super().__init__(client, "Summarizer")
        self.model_name = MODEL_NAME
        self.memory = memory
        self.meteor_calc = METEORCalculator()
        self.saved_prompts = self._load_saved_prompts()
        self.logger.info(f"[Summarizer] Инициализирован v5.6.1 (BertScore: {'вкл' if BERTSCORE_ENABLED else 'выкл'})")

    def _load_saved_prompts(self) -> List[Dict[str, Any]]:
        if not self.memory or not USE_SAVED_PROMPTS:
            return []
        try:
            # ✅ Используем DATA_LANG_DIR для пути к памяти
            prompts_file = self.memory.memory_dir / "best_summary_prompts.json"
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
                prompts.sort(key=lambda x: x.get("improvement", 0), reverse=True)
                self.logger.info(f"[Summarizer] Загружено {len(prompts)} сохранённых промптов")
                return prompts
        except Exception as e:
            self.logger.warning(f"[Summarizer] Ошибка загрузки промптов: {e}")
        return []

    def _ensure_text_placeholder(self, prompt: str) -> str:
        if "{text}" not in prompt:
            if LANGUAGE.lower() == 'en':
                prompt += "\n\nTEXT:\n{text}\n\nSUMMARY:"
            else:
                prompt += "\n\nТЕКСТ:\n{text}\n\nРЕЗЮМЕ:"
        return prompt

    def _compute_summary_metrics(self, summary: str, reference: str, original: str) -> Dict[str, float]:
        """Вычисление метрик через вызов судьи (execute, чтобы получить BertScore)"""
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
            self.logger.warning(f"[Summarizer] Ошибка вызова судьи: {e}, используем fallback")
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

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Запуск суммаризации")
        prompt_template = state.get("prompt_summary", "") or self.BASE_PROMPT
        prompt_template = self._ensure_text_placeholder(prompt_template)

        input_text = state.get("corrected_text", "") or state.get("input_text", "")
        reference_summary = state.get("reference_summary", "")

        if not input_text:
            self.logger.error("[Summarizer] Пустой входной текст")
            return self._create_empty_result()

        temperatures = [0.3, 0.6, 0.9]
        variants, temps, prompts_list = self._generate_variants(prompt_template, input_text, temperatures)
        while len(variants) < 3:
            variants.append("")
            temps.append(0.5)
            prompts_list.append(prompt_template)

        state["summary_outputs"] = variants.copy()
        best_result = self._select_best_by_score(variants, reference_summary, input_text, temps, prompts_list)

        if not best_result.get("summary_text"):
            best_result["summary_text"] = variants[0] if variants[0] else "Невозможно создать резюме"

        sumscore = best_result.get("metrics_summary", {}).get("SumScore", 0)
        if sumscore < 0.5:
            self.logger.warning(f"[Summarizer] SumScore={sumscore:.3f} < 0.5, запуск адаптивных попыток")
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
                self.logger.error(f"[Summarizer] Ошибка записи в память: {e}")

        best_result["summary_outputs"] = variants
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
                    system_prompt="You are a professional summarizer. Return only a 1-4 sentence summary." if LANGUAGE.lower() == 'en' else "Ты профессиональный суммаризатор. Возвращай только резюме из 1-4 предложений.",
                )
                if response and len(response.strip()) >= 10:
                    variants.append(response.strip())
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
                else:
                    variants.append("")
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
            except Exception as e:
                self.logger.warning(f"[Summarizer] Ошибка генерации (temp={temp}): {e}")
                variants.append("")
                temps_used.append(temp)
                prompts_used.append(prompt_template)
        return variants, temps_used, prompts_used

    def _select_best_by_score(self, variants: List[str], reference: str, original: str,
                              temperatures: List[float], prompts: List[str]) -> Dict[str, Any]:
        best_variant = ""
        best_sumscore = -1.0
        best_idx = 0
        best_temp = temperatures[0] if temperatures else 0.5
        best_prompt = prompts[0] if prompts else ""
        best_metrics = None

        print("\n" + "-" * 80)
        print("  📊 ОЦЕНКА ВАРИАНТОВ СУММАРИЗАЦИИ (с BertScore)")
        print("-" * 80)

        for i, variant in enumerate(variants):
            if not variant:
                continue
            metrics = self._compute_summary_metrics(variant, reference, original)
            sumscore = metrics.get("SumScore", 0)
            g_eval = metrics.get("G_Eval", 0)
            meteor = metrics.get("METEOR", 0)
            llm_judge = metrics.get("LLM_Judge", 0)
            bertscore = metrics.get("BertScore", 0)
            temp = temperatures[i] if i < len(temperatures) else 0.5

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
                best_prompt = prompts[i]
                best_metrics = metrics

        print("-" * 80)
        if best_variant:
            print(f"  ✅ Лучший вариант #{best_idx+1} (temp={best_temp:.2f}, SumScore={best_sumscore:.4f})")
        else:
            print(f"  ⚠️ Нет валидных вариантов")
        print("-" * 80 + "\n")

        return self._create_result(best_variant, best_temp, best_prompt, reference, original, best_metrics)

    def _adaptive_retry(self, current_best: Dict[str, Any], base_prompt: str,
                        input_text: str, reference: str) -> Dict[str, Any]:
        best_result = current_best
        best_sumscore = current_best.get("metrics_summary", {}).get("SumScore", 0)
        attempts = 0
        base_prompt = self._ensure_text_placeholder(base_prompt)
        strategies = []

        for temp in [0.2, 0.5, 0.8]:
            strategies.append(("temp_retry", base_prompt, temp))

        if USE_SAVED_PROMPTS and self.saved_prompts:
            for i, saved in enumerate(self.saved_prompts[:3]):
                prompt = self._ensure_text_placeholder(saved.get("prompt", base_prompt))
                strategies.append((f"saved_{i}", prompt, 0.5))

        if USE_FEW_SHOT_PROMPT:
            strategies.append(("few_shot", self._ensure_text_placeholder(self.FEW_SHOT_PROMPT), 0.5))

        if USE_CHAIN_OF_THOUGHT_PROMPT:
            strategies.append(("cot", self._ensure_text_placeholder(self.CHAIN_OF_THOUGHT_PROMPT), 0.3))

        print("\n" + "=" * 80)
        print("  🔄 АДАПТИВНАЯ СУММАРИЗАЦИЯ (SumScore < 0.5)")
        print("=" * 80)

        for name, prompt, temp in strategies:
            if attempts >= MAX_LEV_RETRY_ATTEMPTS:
                break
            print(f"\n  Попытка {attempts+1}: стратегия '{name}' (temp={temp})")
            try:
                full_prompt = prompt.format(text=input_text)
                response = self.client.generate(
                    prompt=full_prompt,
                    temperature=temp,
                    system_prompt="You are a professional summarizer. Return only a 1-4 sentence summary." if LANGUAGE.lower() == 'en' else "Ты профессиональный суммаризатор. Возвращай только резюме из 1-4 предложений.",
                )
                if not response or len(response.strip()) < 10:
                    attempts += 1
                    continue
                metrics = self._compute_summary_metrics(response, reference, input_text)
                sumscore = metrics.get("SumScore", 0)
                print(f"     └─ SumScore={sumscore:.4f}")
                if sumscore > 0.7:
                    print(f"     └─ ✅ Успех! SumScore > 0.7")
                    return self._create_result(response, temp, full_prompt, reference, input_text, metrics)
                if sumscore > best_sumscore:
                    print(f"     └─ Новый лучший результат (SumScore={sumscore:.4f})")
                    best_sumscore = sumscore
                    best_result = self._create_result(response, temp, full_prompt, reference, input_text, metrics)
            except Exception as e:
                self.logger.warning(f"[Summarizer] Ошибка в {name}: {e}")
            attempts += 1

        print("=" * 80 + "\n")
        return best_result

    def _create_result(self, summary_text: str, temperature: float, prompt: str,
                       reference: str, original: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        if metrics is None:
            metrics = self._compute_summary_metrics(summary_text, reference, original)
        return {
            "summary_text": summary_text.strip(),
            "summary_outputs": [],
            "prompt_summary": prompt,
            "best_temperature_summary": str(temperature),
            "best_prompt_type": "",
            "best_model": self.model_name,
            "metrics_summary": metrics
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        return {
            "summary_text": "",
            "summary_outputs": [],
            "prompt_summary": "",
            "best_temperature_summary": "N/A",
            "best_prompt_type": "",
            "best_model": self.model_name,
            "metrics_summary": {}
        }

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] - {message}")