#!/usr/bin/env python3
"""
Агрегатор для суммаризации
Версия 1.3 - Логика, аналогичная CorrectorAggregator:
   1. Выбирает лучший вариант из summary_outputs (по метрикам, если есть, иначе по длине)
   2. Пытается улучшить его через LLM, используя остальные варианты
   3. Сравнивает метрики базового и улучшенного вариантов (через вызов судьи)
   4. Возвращает лучший
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from metrics.wer_calculator import WERCalculator
from metrics.levenstein_calculator import LevenshteinCalculator
from metrics.perplexity_calculator import PerplexityCalculator
from config import MODEL_NAME


class SummarizerAggregator(BaseAgent):
    AGGREGATION_PROMPT = """Ты - эксперт по суммаризации.

ЗАДАЧА: Улучши базовое резюме, используя идеи из альтернативных вариантов.

ВАЖНО:
- Сохрани ключевые факты
- Объём: 1-4 предложения
- Не добавляй новую информацию

БАЗОВОЕ РЕЗЮМЕ (лучшее по метрикам):
{base_summary}

АЛЬТЕРНАТИВНЫЙ ВАРИАНТ 1:
{alt_1}

АЛЬТЕРНАТИВНЫЙ ВАРИАНТ 2:
{alt_2}

УЛУЧШЕННОЕ РЕЗЮМЕ:"""

    def __init__(self, client: LMStudioClient):
        super().__init__(client, "SummarizerAggregator")
        self.wer_calc = WERCalculator()
        self.lev_calc = LevenshteinCalculator()
        self.perplexity_calc = PerplexityCalculator(language="ru")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("  🧩 АГРЕГАТОР СУММАРИЗАЦИИ")
        print("=" * 80)

        summary_outputs = state.get("summary_outputs", [])
        reference_summary = state.get("reference_summary", "")
        input_text = state.get("corrected_text", "") or state.get("input_text", "")

        print(f"  Получено вариантов: {len(summary_outputs)}")

        if not summary_outputs:
            print("  ⚠️ Нет вариантов, возвращаем текущее резюме")
            return {"summary_text": state.get("summary_text", ""), "aggregator_used": False}
        if len(summary_outputs) == 1:
            print("  ℹ️ Только один вариант, агрегация не требуется")
            return {"summary_text": summary_outputs[0], "aggregator_used": False, "aggregation_reason": "Один вариант"}

        # 1. Выбираем лучший вариант как базу
        best_summary, best_idx = self._select_best_base(summary_outputs, reference_summary, input_text)
        print(f"  Лучшая база: вариант #{best_idx+1}")

        # 2. Альтернативные варианты (все, кроме лучшего)
        alt_variants = [v for i, v in enumerate(summary_outputs) if i != best_idx]
        alt_1 = alt_variants[0] if len(alt_variants) > 0 else ""
        alt_2 = alt_variants[1] if len(alt_variants) > 1 else ""

        # 3. Пытаемся улучшить через LLM
        aggregated = None
        if alt_1 and alt_2:
            print("  🤖 LLM агрегация...")
            aggregated = self._aggregate_via_llm(best_summary, alt_1, alt_2)

        # 4. Сравниваем метрики базового и агрегированного вариантов
        if aggregated and aggregated.strip():
            final_summary, reason = self._select_best_by_metrics(
                base_summary=best_summary,
                aggregated_summary=aggregated,
                reference=reference_summary,
                original=input_text
            )
            print(f"  Результат сравнения: {reason}")
        else:
            final_summary = best_summary
            reason = "Агрегация не удалась или не проводилась"
            print(f"  ⚠️ {reason}")

        # 5. Проверка смысла (чтобы не слишком изменилось)
        similarity = self.lev_calc.calculate(best_summary, final_summary)
        if similarity < 0.6:
            print(f"  ⚠️ Смысл изменился (схожесть={similarity:.3f}), возвращаем базу")
            final_summary = best_summary
            reason = "Смысл изменился"

        # 6. Защита от пустого результата
        if not final_summary or len(final_summary.strip()) < 10:
            print("  ⚠️ Результат невалиден, возвращаем базу")
            final_summary = best_summary
            reason = "Результат невалиден"

        print(f"  Результат: {reason}")
        print("=" * 80 + "\n")

        return {
            "summary_text": final_summary,
            "aggregator_used": True,
            "base_variant_idx": best_idx,
            "aggregation_similarity": similarity,
            "aggregation_reason": reason
        }

    def _select_best_base(self, variants: List[str], reference: str, original: str) -> tuple:
        """Выбор лучшего варианта в качестве базы (по метрикам, если есть, иначе по длине)"""
        if not variants:
            return "", 0
        if len(variants) == 1:
            return variants[0], 0

        # Пытаемся оценить каждый вариант по метрикам (вызываем судью)
        best_score = -float('inf')
        best_idx = 0
        best_variant = variants[0]

        for i, variant in enumerate(variants):
            if not variant:
                continue
            metrics = self._compute_summary_metrics(variant, reference, original)
            sumscore = metrics.get("SumScore", 0)
            print(f"  Вариант #{i+1}: SumScore={sumscore:.4f}")
            if sumscore > best_score:
                best_score = sumscore
                best_idx = i
                best_variant = variant

        return best_variant.strip(), best_idx

    def _select_best_by_metrics(self, base_summary: str, aggregated_summary: str,
                                reference: str, original: str) -> tuple:
        """Сравнение метрик базового и агрегированного вариантов"""
        metrics_base = self._compute_summary_metrics(base_summary, reference, original)
        metrics_agg = self._compute_summary_metrics(aggregated_summary, reference, original)

        sumscore_base = metrics_base.get("SumScore", 0)
        sumscore_agg = metrics_agg.get("SumScore", 0)

        print(f"     └─ Базовый: SumScore={sumscore_base:.4f}")
        print(f"     └─ Агрегированный: SumScore={sumscore_agg:.4f}")

        if sumscore_base >= sumscore_agg:
            return base_summary, f"Базовый лучше (SumScore={sumscore_base:.4f})"
        else:
            return aggregated_summary, f"Агрегированный лучше (SumScore={sumscore_agg:.4f})"

    def _aggregate_via_llm(self, base: str, alt_1: str, alt_2: str) -> str:
        """Агрегация через LLM"""
        prompt = self.AGGREGATION_PROMPT.format(
            base_summary=base[:1000],
            alt_1=alt_1[:500],
            alt_2=alt_2[:500]
        )
        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt="Ты эксперт по суммаризации. Верни только улучшенное резюме.",
            max_tokens=512,
            model=MODEL_NAME
        )
        return response.strip() if response else None

    def _compute_summary_metrics(self, summary: str, reference: str, original: str) -> Dict[str, float]:
        """Вычисление метрик суммаризации через вызов судьи"""
        try:
            from agents.summarization_judge import SummarizationJudge
            judge = SummarizationJudge(self.client)
            if hasattr(judge, 'evaluate') and callable(judge.evaluate):
                result = judge.evaluate(original, summary, reference)
                g_eval = result.get("g_eval", 0.5)
                llm_judge = result.get("llm_judge", 5.0)
            else:
                state = {"corrected_text": original, "summary_text": summary, "reference_summary": reference}
                result = judge.execute(state)
                metrics = result.get("metrics_summary", {})
                g_eval = metrics.get("g_eval_overall", 0.5)
                llm_judge = metrics.get("llm_score", 5.0)
        except Exception as e:
            self.logger.warning(f"[SummarizerAggregator] Ошибка вызова судьи: {e}")
            g_eval = 0.5
            llm_judge = 5.0

        meteor = 0.5
        if reference:
            try:
                from metrics.meteor_calculator import METEORCalculator
                meteor_calc = METEORCalculator()
                meteor = meteor_calc.calculate(reference, summary)
            except Exception as e:
                self.logger.warning(f"[SummarizerAggregator] Ошибка METEOR: {e}")

        from config import calculate_sumscore
        sumscore = calculate_sumscore(g_eval, llm_judge, meteor)
        return {"G_Eval": g_eval, "LLM_Judge": llm_judge, "METEOR": meteor, "SumScore": sumscore}