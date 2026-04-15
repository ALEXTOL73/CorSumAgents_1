#!/usr/bin/env python3
"""
Агент коррекции на базе ансамбля LLM
Версия 5.8.0 - Пост-обработка результатов + унификация весов
"""
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory
from metrics.wer_calculator import WERCalculator
from metrics.levenstein_calculator import LevenshteinCalculator
from metrics.perplexity_calculator import PerplexityCalculator
from utils.text_postprocessor import TextPostprocessor  # ✅ добавлен пост-процессор
from config import (
    ENSEMBLE_SIZE, TEMPERATURE_RANGE, MODEL_NAME,
    ADAPTIVE_LEV_RETRY_ENABLED, MAX_LEV_RETRY_ATTEMPTS, DELTA_LEV_THRESHOLD,
    LEV_RETRY_TEMPS, USE_SAVED_PROMPTS, USE_FEW_SHOT_PROMPT, USE_CHAIN_OF_THOUGHT_PROMPT,
    DYNAMIC_TEMPERATURES_ENABLED, SELF_CONSISTENCY_ENABLED, SELF_CONSISTENCY_EXTRA_COUNT,
    ERROR_PROFILE_ENABLED, LEV_WEIGHT, PERPLEXITY_WEIGHT
)
import json
from pathlib import Path


class CorrectorEnsemble(BaseAgent):
    BASE_PROMPT = """Ты профессиональный редактор с 20-летним стажем.

ЗАДАЧА: Исправь все ошибки в тексте, сохраняя исходный смысл.

ТИПЫ ОШИБОК ДЛЯ ИСПРАВЛЕНИЯ:
1. Орфографические ошибки
2. Пунктуационные ошибки
3. Грамматические ошибки
4. Стилистические недостатки

ВАЖНО:
- Не меняй смысл текста
- Не добавляй новую информацию
- Не удаляй ключевые детали
- Сохраняй форматирование
- ✅ СОХРАНИ СТИЛЬ АВТОРА (тональность, манеру изложения, лексику)

ТЕКСТ ДЛЯ КОРРЕКЦИИ:
{text}

ИСПРАВЛЕННЫЙ ТЕКСТ:"""

    FEW_SHOT_PROMPT = """Ты профессиональный редактор. Исправь ошибки в тексте, сохраняя стиль автора.

ПРИМЕР 1:
Вход: "Как отметила палата в по которых он учаях центры занял об ru 
 го глав antirь с заявлениями получат свой пособий по безрабоIn по что устраиваться на работу в и сразу по "
Выход: "Как отметила палата, в некоторых случаях центры занятости соглашались с заявлениями получателей пособий по \
безработице, что устраиваться на работу «неразумно»."

ПРИМЕР 2:
Вход: " Примерно 6 3 млн у бати за границу она 3 7 мин человек пересолили с Г в другой регион Украины "
Выход: "Примерно 6,3 млн уехали за границу, еще 3,7 млн человек переселились в другой регион Украины." 

ПРИМЕР 3:
Вход: "в варе этого года верховный комиссар ООН по делам боже 10 с в 10 или 11 по гранди сообщил и о около 10 млн 
 украинцев были вынужде 10 я покину 11 дома из за боевых действий "
Выход: "В январе этого года верховный комиссар ООН по делам беженцев Филиппо Гранди сообщил, что около 10 млн \
украинцев были вынуждены покинуть дома из-за боевых действий."

ТЕПЕРЬ ИСПРАВЬ ЭТОТ ТЕКСТ, СОХРАНЯЯ СТИЛЬ:
{text}

ИСПРАВЛЕННЫЙ ТЕКСТ:"""

    CHAIN_OF_THOUGHT_PROMPT = """Ты профессиональный редактор. Исправь ошибки в тексте, сохраняя стиль автора.

ШАГ 1: Прочитай текст внимательно
ШАГ 2: Найди все орфографические ошибки
ШАГ 3: Найди все пунктуационные ошибки
ШАГ 4: Найди все грамматические ошибки
ШАГ 5: Исправь каждую ошибку по порядку
ШАГ 6: Проверь что смысл и стиль не изменились
ШАГ 7: Верни только исправленный текст

ТЕКСТ ДЛЯ КОРРЕКЦИИ:
{text}

ИСПРАВЛЕННЫЙ ТЕКСТ:"""

    def __init__(self, client: LMStudioClient, ensemble_size: int = ENSEMBLE_SIZE, memory: AgentMemory = None):
        super().__init__(client, "CorrectorEnsemble")
        self.model_name = MODEL_NAME
        self.default_ensemble_size = ensemble_size
        self.memory = memory
        self.current_ensemble_size = ensemble_size
        self.wer_calc = WERCalculator()
        self.lev_calc = LevenshteinCalculator()
        self.perplexity_calc = PerplexityCalculator(language="ru")
        self.saved_prompts = self._load_saved_prompts()
        self.error_profile = {}
        self.logger.info(f"[CorrectorEnsemble] Инициализирован v5.8.0 (пост-обработка)")

    def _load_saved_prompts(self) -> List[Dict[str, Any]]:
        if not self.memory or not USE_SAVED_PROMPTS:
            return []
        try:
            prompts_file = Path("data/memory/best_prompts.json")
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
                if prompts and isinstance(prompts[0], dict):
                    prompts.sort(key=lambda x: x.get("improvement", 0), reverse=True)
                self.logger.info(f"[CorrectorEnsemble] Загружено {len(prompts)} сохранённых промптов")
                return prompts
        except Exception as e:
            self.logger.warning(f"[CorrectorEnsemble] Ошибка загрузки промптов: {e}")
        return []

    def _ensure_text_placeholder(self, prompt: str) -> str:
        if "{text}" not in prompt:
            prompt += "\n\nТЕКСТ ДЛЯ КОРРЕКЦИИ:\n{text}\n\nИСПРАВЛЕННЫЙ ТЕКСТ:"
        return prompt

    def _compute_dynamic_temperatures(self, input_text: str, reference_text: str = None) -> List[float]:
        if not DYNAMIC_TEMPERATURES_ENABLED:
            return [TEMPERATURE_RANGE[0], (TEMPERATURE_RANGE[0] + TEMPERATURE_RANGE[1]) / 2, TEMPERATURE_RANGE[1]]
        length = len(input_text)
        if reference_text:
            wer = self.wer_calc.calculate(reference_text, input_text)
        else:
            wer = 0.5
        if wer > 0.6 or length > 2000:
            return [0.4, 0.7, 1.0]
        elif wer > 0.3:
            return [0.2, 0.5, 0.8]
        else:
            return [0.1, 0.3, 0.6]

    def _get_error_profile_prompt(self) -> str:
        if not ERROR_PROFILE_ENABLED or not self.error_profile:
            return ""
        error_types = []
        if self.error_profile.get("spelling", 0) > 2:
            error_types.append("обрати особое внимание на орфографию")
        if self.error_profile.get("grammar", 0) > 2:
            error_types.append("проверь грамматику")
        if self.error_profile.get("punctuation", 0) > 2:
            error_types.append("исправь пунктуацию")
        if error_types:
            return f"\nВАЖНО: {', '.join(error_types)}!"
        return ""

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Запуск ансамбля")
        prompt_template = state.get("prompt_correction_template", "") or state.get("prompt_correction", "")
        if not prompt_template:
            prompt_template = self.BASE_PROMPT
        prompt_template = self._ensure_text_placeholder(prompt_template)

        input_text = state.get("input_text_for_correction", "") or state.get("input_text", "")
        reference_text = state.get("reference_text", "")

        if not input_text:
            self.logger.error("[Ensemble] Пустой входной текст")
            return self._create_empty_result()

        wer_before = self.wer_calc.calculate(reference_text, input_text) if reference_text else 0.5
        lev_before = self.lev_calc.calculate(reference_text, input_text) if reference_text else 0.0
        self.logger.info(f"[Ensemble] WER до: {wer_before:.4f}, Lev до: {lev_before:.4f}")

        domain = state.get("detected_language", "general")
        if self.memory and ERROR_PROFILE_ENABLED:
            self.error_profile = self.memory.get_error_profile(domain) or {}

        error_instruction = self._get_error_profile_prompt()
        if error_instruction:
            prompt_template = error_instruction + "\n" + prompt_template
            self.logger.info(f"[Ensemble] Добавлена инструкция: {error_instruction}")

        temperatures = self._compute_dynamic_temperatures(input_text, reference_text)
        self.logger.info(f"[Ensemble] Динамические температуры: {temperatures}")

        variants, temps, prompts_list = self._generate_variants(prompt_template, input_text, reference_text, temperatures)

        if SELF_CONSISTENCY_ENABLED and len(variants) >= 2:
            best_temp = temps[0] if temps else 0.5
            variants, temps, prompts_list = self._add_self_consistency(
                variants, temps, prompts_list, input_text, reference_text, best_temp
            )

        while len(variants) < 3:
            variants.append(input_text)
            temps.append(0.5)
            prompts_list.append(prompt_template)

        state["ensemble_outputs"] = variants.copy()
        best_result = self._select_best_by_score(variants, reference_text, input_text, temps, prompts_list)

        corrected = best_result.get("corrected_text", "")
        if not corrected or len(corrected.strip()) < len(input_text.strip()) * 0.3:
            self.logger.warning("[Ensemble] Результат коррекции пуст, возвращаем оригинал")
            best_result = self._create_result(input_text, 0.5, prompt_template, reference_text, input_text, wer_before, lev_before)
            best_result["ensemble_outputs"] = [input_text, input_text, input_text]

        m_cor = best_result.get("metrics_correction", {}) or {}
        delta_lev = float(m_cor.get("delta_LEV", 0) or 0)
        if ADAPTIVE_LEV_RETRY_ENABLED and delta_lev < DELTA_LEV_THRESHOLD:
            self.logger.warning(f"[Ensemble] delta_Lev={delta_lev:.4f} < {DELTA_LEV_THRESHOLD}, адаптивные попытки")
            best_result = self._adaptive_retry(best_result, prompt_template, input_text, reference_text, wer_before, lev_before)

        if self.memory and reference_text:
            try:
                wer_after = self.wer_calc.calculate(reference_text, best_result["corrected_text"])
                self.memory.learn_from_correction(
                    original=input_text,
                    corrected=best_result["corrected_text"],
                    reference=reference_text,
                    wer_before=wer_before,
                    wer_after=wer_after,
                    domain=domain,
                    prompt_used=prompt_template,
                    model_used=best_result["best_model"],
                    error_profile=self.error_profile
                )
            except Exception as e:
                self.logger.error(f"[Ensemble] Ошибка записи в память: {e}")

        best_result["ensemble_outputs"] = variants
        return best_result

    def _generate_variants(self, prompt_template: str, input_text: str, reference_text: str,
                          temperatures: List[float]) -> Tuple[List[str], List[float], List[str]]:
        variants, temps_used, prompts_used = [], [], []
        for temp in temperatures:
            try:
                full_prompt = prompt_template.format(text=input_text)
                self.logger.debug(f"[Ensemble] Промпт (temp={temp}): {full_prompt[:200]}")
                response = self.client.generate(
                    prompt=full_prompt,
                    temperature=temp,
                    system_prompt="Ты профессиональный редактор. Возвращай только исправленный текст, сохраняя стиль автора."
                )
                self.logger.debug(f"[Ensemble] Ответ (temp={temp}): {response[:100] if response else 'None'}")
                if response and len(response.strip()) >= len(input_text.strip()) * 0.3:
                    # ✅ пост-обработка
                    cleaned = TextPostprocessor.clean_text(response.strip())
                    variants.append(cleaned)
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
                else:
                    variants.append(input_text)
                    temps_used.append(temp)
                    prompts_used.append(full_prompt)
            except Exception as e:
                self.logger.warning(f"[Ensemble] Ошибка генерации (temp={temp}): {e}")
                variants.append(input_text)
                temps_used.append(temp)
                prompts_used.append(prompt_template)
        return variants, temps_used, prompts_used

    def _add_self_consistency(self, variants: List[str], temps: List[float], prompts: List[str],
                              input_text: str, reference_text: str, best_temp: float) -> Tuple[List[str], List[float], List[str]]:
        self.logger.info(f"[Ensemble] Self-consistency: добавление {SELF_CONSISTENCY_EXTRA_COUNT} вариантов с temp={best_temp}")
        for _ in range(SELF_CONSISTENCY_EXTRA_COUNT):
            try:
                full_prompt = prompts[0].format(text=input_text)
                response = self.client.generate(
                    prompt=full_prompt,
                    temperature=best_temp,
                    system_prompt="Ты профессиональный редактор. Возвращай только исправленный текст, сохраняя стиль автора."
                )
                if response and len(response.strip()) >= len(input_text.strip()) * 0.3:
                    # ✅ пост-обработка
                    cleaned = TextPostprocessor.clean_text(response.strip())
                    variants.append(cleaned)
                    temps.append(best_temp)
                    prompts.append(full_prompt)
            except Exception as e:
                self.logger.warning(f"[Ensemble] Self-consistency ошибка: {e}")
        return variants, temps, prompts

    def _select_best_by_score(self, variants: List[str], reference: str, original: str,
                             temperatures: List[float], prompts: List[str]) -> Dict[str, Any]:
        wer_original = self.wer_calc.calculate(reference, original) if reference else 0.5
        lev_original = self.lev_calc.calculate(reference, original) if reference else 0.0

        print("\n" + "-" * 80)
        print("  📊 ОЦЕНКА ВАРИАНТОВ КОРРЕКЦИИ")
        print("-" * 80)

        best_variant, best_score, best_idx, best_temp, best_prompt = original, -float('inf'), 0, temperatures[0] if temperatures else 0.5, prompts[0] if prompts else ""

        for i, variant in enumerate(variants):
            if not variant:
                continue
            wer_variant = self.wer_calc.calculate(reference, variant) if reference else 0.5
            lev_variant = self.lev_calc.calculate(reference, variant) if reference else 0.0
            perplexity_result = self.perplexity_calc.calculate(variant, reference) if reference else {}
            perplexity = perplexity_result.get("perplexity", 1.0)
            delta_wer = wer_original - wer_variant
            delta_lev = lev_variant - lev_original
            # ✅ используем константы из config
            perplexity_term = (1.0 - perplexity / 100.0) * PERPLEXITY_WEIGHT
            score = delta_wer + LEV_WEIGHT * delta_lev + perplexity_term
            temp = temperatures[i] if i < len(temperatures) else 0.5

            print(f"  Вариант #{i+1} (temp={temp:.2f}):")
            print(f"     └─ WER: {wer_variant:.4f} (Δ={delta_wer:+.4f})")
            print(f"     └─ LevRating: {lev_variant:.4f} (Δ={delta_lev:+.4f})")
            print(f"     └─ Perplexity: {perplexity:.4f}")
            print(f"     └─ Score: {score:.6f}")

            self.logger.info(f"[Ensemble] Вариант #{i+1} (temp={temp:.2f}): WER={wer_variant:.4f} (Δ={delta_wer:+.4f}), Lev={lev_variant:.4f} (Δ={delta_lev:+.4f}), PPL={perplexity:.4f}, Score={score:.6f}")

            if score > best_score:
                best_score, best_variant, best_idx, best_temp, best_prompt = score, variant, i, temp, prompts[i]

        print("-" * 80)
        if best_score < 0:
            print(f"  ⚠️ Лучший score = {best_score:.4f} < 0, возвращаем оригинал")
            self.logger.warning(f"[Ensemble] Лучший score <0, возвращаем оригинал")
            return self._create_result(original, best_temp, best_prompt, reference, original, wer_original, lev_original)
        else:
            print(f"  ✅ Лучший вариант #{best_idx+1} (temp={best_temp:.2f}, score={best_score:.6f})")
            print("-" * 80 + "\n")
            return self._create_result(best_variant, best_temp, best_prompt, reference, original, wer_original, lev_original)

    def _adaptive_retry(self, current_best: Dict[str, Any], base_prompt: str, input_text: str,
                       reference_text: str, wer_before: float, lev_before: float) -> Dict[str, Any]:
        best_result, best_score = current_best, self._calc_cor_score(current_best, wer_before, lev_before)
        attempts = 0
        base_prompt = self._ensure_text_placeholder(base_prompt)
        strategies = []

        for temp in LEV_RETRY_TEMPS:
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
        print("  🔄 АДАПТИВНАЯ КОРРЕКЦИЯ (delta_Lev < 0)")
        print("=" * 80)

        for name, prompt, temp in strategies:
            if attempts >= MAX_LEV_RETRY_ATTEMPTS:
                break
            print(f"\n  Попытка {attempts+1}: стратегия '{name}' (temp={temp})")
            self.logger.info(f"[Ensemble] Адаптивная попытка {attempts+1}: {name} (temp={temp})")
            try:
                full_prompt = prompt.format(text=input_text)
                response = self.client.generate(prompt=full_prompt, temperature=temp, system_prompt="Ты профессиональный редактор. Возвращай только исправленный текст, сохраняя стиль автора.")
                if not response or len(response.strip()) < len(input_text.strip()) * 0.3:
                    print(f"     └─ Ответ слишком короткий, пропускаем")
                    attempts += 1
                    continue
                # ✅ пост-обработка
                cleaned = TextPostprocessor.clean_text(response.strip())
                wer_after = self.wer_calc.calculate(reference_text, cleaned) if reference_text else 0.5
                lev_after = self.lev_calc.calculate(reference_text, cleaned) if reference_text else 0.0
                delta_lev, delta_wer = lev_after - lev_before, wer_before - wer_after
                perplexity = self.perplexity_calc.calculate(cleaned, reference_text).get("perplexity", 1.0) if reference_text else 1.0
                perplexity_term = (1.0 - perplexity / 100.0) * PERPLEXITY_WEIGHT
                cor_score = delta_wer + LEV_WEIGHT * delta_lev + perplexity_term
                print(f"     └─ delta_WER={delta_wer:+.4f}, delta_Lev={delta_lev:+.4f}, Perplexity={perplexity:.4f}, CorScore={cor_score:.6f}")
                if delta_lev > DELTA_LEV_THRESHOLD:
                    print(f"     └─ ✅ Успех! delta_Lev > 0")
                    return self._create_result(cleaned, temp, full_prompt, reference_text, input_text, wer_before, lev_before)
                if cor_score > best_score:
                    print(f"     └─ Новый лучший результат (CorScore={cor_score:.6f})")
                    best_score, best_result = cor_score, self._create_result(cleaned, temp, full_prompt, reference_text, input_text, wer_before, lev_before)
            except Exception as e:
                self.logger.warning(f"[Ensemble] Ошибка в {name}: {e}")
                print(f"     └─ Ошибка: {e}")
            attempts += 1

        print("=" * 80 + "\n")
        return best_result

    def _create_result(self, corrected_text: str, temperature: float, prompt: str,
                      reference: str, original: str, wer_original: float, lev_original: float) -> Dict[str, Any]:
        # ✅ пост-обработка финального текста
        cleaned = TextPostprocessor.clean_text(corrected_text)
        if not cleaned or len(cleaned.strip()) < len(original.strip()) * 0.3:
            cleaned = original
        wer_after = self.wer_calc.calculate(reference, cleaned) if reference else 0.5
        lev_after = self.lev_calc.calculate(reference, cleaned) if reference else 0.0
        delta_wer, delta_lev = wer_original - wer_after, lev_after - lev_original
        perplexity_result = self.perplexity_calc.calculate(cleaned, reference) if reference else {}
        perplexity = perplexity_result.get("perplexity", 1.0)
        perplexity_term = (1.0 - perplexity / 100.0) * PERPLEXITY_WEIGHT
        cor_score = delta_wer + LEV_WEIGHT * delta_lev + perplexity_term
        return {
            "corrected_text": cleaned,
            "ensemble_outputs": [],
            "prompt_correction": prompt,
            "adaptive_config": {"temperature": temperature},
            "perplexity": perplexity_result,
            "best_model": self.model_name,
            "best_temperature": str(temperature),
            "metrics_correction": {
                "WER_0": wer_original, "WER": wer_after, "delta_WER": delta_wer,
                "LevRating_0": lev_original, "LevRating": lev_after, "delta_LEV": delta_lev,
                "CorScore": cor_score
            }
        }

    def _calc_cor_score(self, result: Dict[str, Any], wer_before: float, lev_before: float) -> float:
        m = result.get("metrics_correction", {})
        delta_wer = m.get("delta_WER", 0)
        delta_lev = m.get("delta_LEV", 0)
        perplexity = result.get("perplexity", {}).get("perplexity", 1.0)
        perplexity_term = (1.0 - perplexity / 100.0) * PERPLEXITY_WEIGHT
        return delta_wer + LEV_WEIGHT * delta_lev + perplexity_term

    def _create_empty_result(self) -> Dict[str, Any]:
        return {"corrected_text": "", "ensemble_outputs": [], "prompt_correction": "", "adaptive_config": {}, "perplexity": {}, "best_model": self.model_name, "best_temperature": "N/A", "metrics_correction": {}}

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] - {message}")