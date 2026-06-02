#!/usr/bin/env python3
"""
Агент-агрегатор для объединения выходов ансамбля корректоров
Версия 5.0.9 - Получение типов промптов и температур из state
"""
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from config import MODEL_NAME, LEV_WEIGHT, PERPLEXITY_WEIGHT, LANGUAGE
from metrics.levenstein_calculator import LevenshteinCalculator
from metrics.perplexity_calculator import PerplexityCalculator
from metrics.wer_calculator import WERCalculator
from utils.lmstudio_client import LMStudioClient


class CorrectorAggregator(BaseAgent):
    AGGREGATION_PROMPT = """Ты - профессиональный редактор-эксперт.

ЗАДАЧА: Улучши базовую коррекцию текста, используя идеи из альтернативных вариантов.

ВАЖНО:
- Сохрани ИСХОДНЫЙ СМЫСЛ текста
- Сохрани ПРИМЕРНЫЙ РАЗМЕР (не более ±5%)
- Не добавляй новую информацию
- Не удаляй ключевые детали

БАЗОВАЯ КОРРЕКЦИЯ (лучшая по метрикам):
{base_correction}

АЛЬТЕРНАТИВНЫЙ ВАРИАНТ 1:
{alt_1}

АЛЬТЕРНАТИВНЫЙ ВАРИАНТ 2:
{alt_2}

УЛУЧШЕННЫЙ ТЕКСТ:"""

    def __init__(self, client: LMStudioClient):
        super().__init__(client, "CorrectorAggregator")
        self.wer_calc = WERCalculator()
        self.lev_calc = LevenshteinCalculator()
        self.perplexity_calc = PerplexityCalculator(language=LANGUAGE.lower())

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("  🧩 АГРЕГАТОР КОРРЕКЦИЙ - ОЦЕНКА ВАРИАНТОВ")
        print("=" * 80)

        ensemble_outputs = state.get("ensemble_outputs", [])
        reference_text = state.get("reference_text", "")
        input_text = state.get("input_text", "")

        # Получаем типы промптов и температуры из state (теперь они передаются ансамблем)
        ensemble_prompts = state.get("ensemble_prompts", [])
        ensemble_temps = state.get("ensemble_temperatures", [])

        # Дополняем недостающие значения
        if len(ensemble_prompts) < len(ensemble_outputs):
            ensemble_prompts.extend(["N/A"] * (len(ensemble_outputs) - len(ensemble_prompts)))
        if len(ensemble_temps) < len(ensemble_outputs):
            ensemble_temps.extend(["N/A"] * (len(ensemble_outputs) - len(ensemble_temps)))

        print(f"  📊 Получено вариантов: {len(ensemble_outputs)}")
        print()

        if not ensemble_outputs:
            print("  ⚠️ Нет вариантов, возвращаем corrected_text")
            return {"corrected_text": state.get("corrected_text", ""), "aggregator_used": False}
        if len(ensemble_outputs) == 1:
            print("  ℹ️ Только один вариант, агрегация не требуется")
            print(f"  ✅ Итоговый текст: {ensemble_outputs[0][:200]}...")
            return {"corrected_text": ensemble_outputs[0], "aggregator_used": False, "aggregation_reason": "Один вариант"}

        # Вычисляем метрики для каждого варианта
        wer_original = self.wer_calc.calculate(reference_text, input_text) if reference_text else 0.5
        lev_original = self.lev_calc.calculate(reference_text, input_text) if reference_text else 0.0

        print("  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
        print("  │                              ОЦЕНКА ВАРИАНТОВ КОРРЕКЦИИ                                     │")
        print("  ├─────────────────────────────────────────────────────────────────────────────────────────────┤")
        print("  │ № │     WER   │ LevRating │ Perplexity │   ΔWER   │  ΔLev   │  Score │   Промпт    │ Temp   │")
        print("  ├─────────────────────────────────────────────────────────────────────────────────────────────┤")

        scores = []
        for i, variant in enumerate(ensemble_outputs):
            if not variant:
                scores.append(-1e9)
                prompt_type = ensemble_prompts[i] if i < len(ensemble_prompts) else "N/A"
                # ✅ Определяем номер ТИПА промпта (1-5), не индекс варианта
                if "saved" in prompt_type.lower():
                    prompt_short = "#2 (Mem)"
                elif "few_shot" in prompt_type.lower():
                    prompt_short = "#3 (FS)"
                elif "cot" in prompt_type.lower() or "chain" in prompt_type.lower():
                    prompt_short = "#4 (CoT)"
                elif "self-consistency" in prompt_type.lower():
                    prompt_short = "#5 (SC)"
                elif "base" in prompt_type.lower() or prompt_type == "N/A":
                    prompt_short = "#1 (base)"
                else:
                    prompt_short = f"#{i+1}"
                temp_val = ensemble_temps[i] if i < len(ensemble_temps) else "N/A"

                print(f"  │ {i+1} │   пустой       │                 │             │          │         │         │ {prompt_short:10} │{temp_val:4} │")
                continue
            wer_v = self.wer_calc.calculate(reference_text, variant) if reference_text else 0.5
            lev_v = self.lev_calc.calculate(reference_text, variant) if reference_text else 0.0
            ppl = self.perplexity_calc.calculate(variant, reference_text).get("perplexity", 1.0) if reference_text else 1.0
            delta_wer = wer_original - wer_v
            delta_lev = lev_v - lev_original
            score = delta_wer + LEV_WEIGHT * delta_lev + (1.0 - ppl) * PERPLEXITY_WEIGHT
            scores.append(score)
            # ✅ Определяем номер ТИПА промпта (1-5), не индекс варианта
            prompt_type = ensemble_prompts[i] if i < len(ensemble_prompts) else "N/A"
            if "saved" in prompt_type.lower():
                prompt_short = "#2 (Mem)"
            elif "few_shot" in prompt_type.lower():
                prompt_short = "#3 (FS)"
            elif "cot" in prompt_type.lower() or "chain" in prompt_type.lower():
                prompt_short = "#4 (CoT)"
            elif "self-consistency" in prompt_type.lower():
                prompt_short = "#5 (SC)"
            elif "base" in prompt_type.lower() or prompt_type == "N/A":
                prompt_short = "#1 (base)"
            else:
                prompt_short = f"#{i+1}"
            temp_val = ensemble_temps[i] if i < len(ensemble_temps) else "N/A"
            print(f"  │ {i+1} │ {wer_v:8.4f}  │ {lev_v:8.4f}  │ {ppl:8.4f}   │ {delta_wer:+6.4f}  │ {delta_lev:+6.4f} │{score:7.3f} │ {prompt_short:10}  │{temp_val:4}    │")

        print("  └──────────────────────────────────────────────────────────────────────────────────────────────┘")
        print()

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_correction = ensemble_outputs[best_idx].strip()
        print(f"  ✅ Лучший вариант: #{best_idx+1} (Score = {scores[best_idx]:.4f})")
        print(f"  📝 Текст: {best_correction[:200]}...")

        alt_variants = [v for i, v in enumerate(ensemble_outputs) if i != best_idx]
        alt_1 = alt_variants[0] if len(alt_variants) > 0 else ""
        alt_2 = alt_variants[1] if len(alt_variants) > 1 else ""

        aggregated = None
        if alt_1 and alt_2:
            print("\n  🤖 Запуск LLM-агрегации...")
            aggregated = self._aggregate_via_llm(best_correction, alt_1, alt_2)
            if aggregated:
                print(f"  📝 Агрегированный текст: {aggregated[:200]}...")
            else:
                print("  ⚠️ Агрегация не удалась")
        elif alt_1:
            aggregated = best_correction

        if aggregated and aggregated.strip():
            final_correction, reason = self._select_best_by_metrics(best_correction, aggregated, reference_text, input_text)
        else:
            final_correction, reason = best_correction, "Агрегация не удалась"

        # Проверка смысла
        similarity = self.lev_calc.calculate(best_correction, final_correction)
        print(f"\n  🔍 Сходство с базовым вариантом: {similarity:.4f}")
        if similarity < 0.6:
            final_correction = best_correction
            reason = "Смысл изменился (сходство < 0.6)"
            print(f"  ⚠️ {reason}")

        if not final_correction or len(final_correction.strip()) < len(input_text.strip()) * 0.3:
            final_correction = best_correction
            reason = "Результат невалиден (слишком короткий)"
            print(f"  ⚠️ {reason}")

        print(f"\n  ✅ Итоговый текст: {final_correction[:200]}...")
        print(f"  📌 Причина выбора: {reason}")
        print("=" * 80 + "\n")

        return {
            "corrected_text": final_correction,
            "aggregator_used": True,
            "base_variant_idx": best_idx,
            "aggregation_similarity": similarity,
            "aggregation_reason": reason
        }

    def _select_best_base(self, variants: List[str], reference: str, original: str) -> tuple:
        if not variants:
            return "", 0
        if len(variants) == 1:
            return variants[0], 0
        wer_original = self.wer_calc.calculate(reference, original) if reference else 0.5
        lev_original = self.lev_calc.calculate(reference, original) if reference else 0.0
        best_score, best_idx, best_variant = -float('inf'), 0, variants[0]
        for i, v in enumerate(variants):
            if not v:
                continue
            wer_v = self.wer_calc.calculate(reference, v) if reference else 0.5
            lev_v = self.lev_calc.calculate(reference, v) if reference else 0.0
            score = (wer_original - wer_v) + LEV_WEIGHT *(lev_v - lev_original)
            if score > best_score:
                best_score, best_idx, best_variant = score, i, v
        return best_variant.strip(), best_idx

    def _select_best_by_metrics(self, base: str, aggregated: str, reference: str, original: str) -> tuple:
        wer_orig = self.wer_calc.calculate(reference, original) if reference else 0.5
        lev_orig = self.lev_calc.calculate(reference, original) if reference else 0.0
        # Базовый
        wer_base = self.wer_calc.calculate(reference, base) if reference else 0.5
        lev_base = self.lev_calc.calculate(reference, base) if reference else 0.0
        ppl_base = self.perplexity_calc.calculate(base, reference).get("perplexity", 1.0) if reference else 1.0
        score_base = (wer_orig - wer_base) + LEV_WEIGHT * (lev_base - lev_orig) + (1.0 - ppl_base)*PERPLEXITY_WEIGHT
        # Агрегированный
        wer_agg = self.wer_calc.calculate(reference, aggregated) if reference else 0.5
        lev_agg = self.lev_calc.calculate(reference, aggregated) if reference else 0.0
        ppl_agg = self.perplexity_calc.calculate(aggregated, reference).get("perplexity", 1.0) if reference else 1.0
        score_agg = (wer_orig - wer_agg) + LEV_WEIGHT * (lev_agg - lev_orig) + (1.0 - ppl_agg)*PERPLEXITY_WEIGHT

        print(f"\n  📊 Сравнение метрик:")
        print(f"     Базовый:     WER={wer_base:.4f}, Lev={lev_base:.4f}, PPL={ppl_base:.4f}, Score={score_base:.4f}")
        print(f"     Агрегированный: WER={wer_agg:.4f}, Lev={lev_agg:.4f}, PPL={ppl_agg:.4f}, Score={score_agg:.4f}")

        if score_base >= score_agg:
            return base, f"Базовый лучше (score={score_base:.4f})"
        else:
            return aggregated, f"Агрегированный лучше (score={score_agg:.4f})"

    def _aggregate_via_llm(self, base: str, alt_1: str, alt_2: str) -> str:
        prompt = self.AGGREGATION_PROMPT.format(base_correction=base[:2000], alt_1=alt_1[:1000], alt_2=alt_2[:1000])
        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt="Ты профессиональный редактор. Сохраняй смысл и размер текста.",
            max_tokens=2048,
            model=MODEL_NAME
        )
        return response.strip() if response else None

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] {message}")

    def log_warning(self, message: str):
        self.logger.warning(f"[{self.name}] ⚠️ {message}")

    def log_info(self, message: str):
        self.logger.info(f"[{self.name}] {message}")