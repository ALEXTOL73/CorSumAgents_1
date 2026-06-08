#!/usr/bin/env python3
"""
Агрегатор для суммаризации
Версия 1.3 - Логика, аналогичная CorrectorAggregator:
   1. Выбирает лучший вариант из summary_outputs (по метрикам, если есть, иначе по длине)
   2. Пытается улучшить его через LLM, используя остальные варианты
   3. Сравнивает метрики базового и улучшенного вариантов (через вызов судьи)
   4. Возвращает лучший
"""
import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from config import MODEL_NAME, LANGUAGE, BERTSCORE_ENABLED
from metrics.levenstein_calculator import LevenshteinCalculator
from metrics.perplexity_calculator import PerplexityCalculator
from metrics.wer_calculator import WERCalculator
from utils.lmstudio_client import LMStudioClient


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
        self.perplexity_calc = PerplexityCalculator(language=LANGUAGE.lower())

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("  🧩 АГРЕГАТОР СУММАРИЗАЦИИ - ОЦЕНКА ВАРИАНТОВ")
        print("=" * 80)

        summary_outputs = state.get("summary_outputs", [])
        reference_summary = state.get("reference_summary", "")
        input_text = state.get("corrected_text", "") or state.get("input_text", "")

        # Получаем типы промптов и температуры из state
        ensemble_prompts = state.get("summary_prompts", []) or state.get("ensemble_prompts_summary", [])
        ensemble_temps = state.get("summary_temperatures", []) or state.get("ensemble_temps_summary", [])

        # DEBUG: показываем что получили из state
        print(f"  📋 summary_prompts: {len(ensemble_prompts)} items | first: {str(ensemble_prompts[0])[:80] if ensemble_prompts else 'EMPTY'}")
        print(f"  🌡️  summary_temperatures: {len(ensemble_temps)} items | values: {ensemble_temps[:5] if ensemble_temps else 'EMPTY'}")

        # Дополняем недостающие значения
        if len(ensemble_prompts) < len(summary_outputs):
            ensemble_prompts.extend(["N/A"] * (len(summary_outputs) - len(ensemble_prompts)))
        if len(ensemble_temps) < len(summary_outputs):
            ensemble_temps.extend(["N/A"] * (len(summary_outputs) - len(ensemble_temps)))

        print(f"  📊 Получено вариантов: {len(summary_outputs)}")

        if not summary_outputs:
            print("  ⚠️ Нет вариантов, возвращаем текущее резюме")
            return {"summary_text": state.get("summary_text", ""), "aggregator_used": False}
        if len(summary_outputs) == 1:
            print("  ℹ️ Только один вариант, агрегация не требуется")
            return {"summary_text": summary_outputs[0], "aggregator_used": False, "aggregation_reason": "Один вариант"}

        # 1. Выбираем лучший вариант как базу (с выводом таблицы)
        best_summary, best_idx, top_temps_sum_str = self._select_best_base(
            summary_outputs, reference_summary, input_text,
            ensemble_prompts, ensemble_temps
        )
        print(f"  ✅ Лучшая база: вариант #{best_idx+1}")

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
            "aggregation_reason": reason,
            "top_temps_sum": top_temps_sum_str
        }

    @staticmethod
    def _get_prompt_short(prompt_type: str, idx: int) -> str:
        """Определяем короткий номер ТИПА промпта (1-5)"""
        pt_lower = (prompt_type or "").lower()
        if "saved" in pt_lower or "mem" in pt_lower or "сохран" in pt_lower:
            return "#2 (Mem)"
        elif "few_shot" in pt_lower or "few-shot" in pt_lower or "fs" in pt_lower or "ПРИМЕРЫ" in prompt_type or "EXAMPLES" in prompt_type:
            return "#3 (FS)"
        elif "cot" in pt_lower or "chain" in pt_lower or "сот" in pt_lower or "ШАГ" in prompt_type or "STEP" in prompt_type:
            return "#4 (CoT)"
        elif "self-consistency" in pt_lower or "sc" in pt_lower or "самосоглас" in pt_lower:
            return "#5 (SC)"
        elif "base" in pt_lower or "базов" in pt_lower or prompt_type == "N/A" or not prompt_type:
            return "#1 (base)"
        else:
            return f"#{idx+1}"

    def _select_best_base(self, variants: List[str], reference: str, original: str,
                          ensemble_prompts: List[str] = None, ensemble_temps: list = None) -> tuple:
        """Выбор лучшего варианта в качестве базы (по метрикам, если есть, иначе по длине)"""
        if not variants:
            return "", 0, "N/A"
        if len(variants) == 1:
            return variants[0], 0, "N/A"

        if ensemble_prompts is None:
            ensemble_prompts = ["N/A"] * len(variants)
        if ensemble_temps is None:
            ensemble_temps = ["N/A"] * len(variants)

        # ====== ФАЗА 1: Молча считаем все метрики (подавляем логи в консоли) ======
        all_metrics = []
        root_logger = logging.getLogger()
        old_level = root_logger.level
        # Временно ставим WARNING, чтобы INFO-логи судьи не ломали таблицу
        root_logger.setLevel(logging.WARNING)
        try:
            for i, variant in enumerate(variants):
                if not variant:
                    all_metrics.append(None)
                    continue
                metrics = self._compute_summary_metrics(variant, reference, original)
                all_metrics.append(metrics)
        finally:
            root_logger.setLevel(old_level)

        # ====== ФАЗА 2: Выводим таблицу ЦЕЛИКОМ ======
        print()
        print("  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐")
        print("  │                              ОЦЕНКА ВАРИАНТОВ СУММАРИЗАЦИИ                                      │")
        print("  ├──────────────────────────────────────────────────────────────────────────────────────────────────┤")
        print("  │ № │  SumScore │  G-Eval  │ LLM-Judge │  METEOR  │ BertScore │   Промпт    │  Temp  │")
        print("  ├──────────────────────────────────────────────────────────────────────────────────────────────────┤")

        best_score = -float('inf')
        best_idx = 0
        best_variant = variants[0]

        for i, variant in enumerate(variants):
            prompt_short = self._get_prompt_short(
                ensemble_prompts[i] if i < len(ensemble_prompts) else "N/A", i
            )
            temp_val = ensemble_temps[i] if i < len(ensemble_temps) else "N/A"
            temp_str = f"{temp_val:.2f}" if isinstance(temp_val, (int, float)) else str(temp_val)

            if not variant or all_metrics[i] is None:
                print(f"  │ {i+1} │   пустой   │          │           │          │           │ {prompt_short:10}  │ {temp_str:5}  │")
                continue

            metrics = all_metrics[i]
            sumscore = metrics.get("SumScore", 0)
            g_eval = metrics.get("G_Eval", 0)
            llm_judge = metrics.get("LLM_Judge", 0)
            meteor = metrics.get("METEOR", 0)
            bertscore = metrics.get("BertScore", 0)

            bert_str = f"{bertscore:.4f}" if BERTSCORE_ENABLED and bertscore else "  N/A  "
            print(f"  │ {i+1} │ {sumscore:8.4f}  │ {g_eval:7.4f} │ {llm_judge:8.1f}  │ {meteor:7.4f}  │ {bert_str:9} │ {prompt_short:10}  │ {temp_str:5}  │")

            if sumscore > best_score:
                best_score = sumscore
                best_idx = i
                best_variant = variant

        print("  └──────────────────────────────────────────────────────────────────────────────────────────────────┘")
        print()

        # Топ-3 температур по SumScore
        scored_temp_pairs = []
        for i, m in enumerate(all_metrics):
            if m is not None and i < len(ensemble_temps):
                sc = m.get("SumScore", 0)
                t = ensemble_temps[i]
                try:
                    t_float = float(t) if not isinstance(t, (int, float)) else t
                    scored_temp_pairs.append((t_float, sc))
                except (ValueError, TypeError):
                    pass
        scored_temp_pairs.sort(key=lambda x: x[1], reverse=True)
        top_temps_parts = []
        for t, sc in scored_temp_pairs[:3]:
            top_temps_parts.append(f"{t:.2f} ({sc:.3f})")
        top_temps_sum_str = "\n".join(top_temps_parts) if top_temps_parts else "N/A"

        return best_variant.strip(), best_idx, top_temps_sum_str

    def _select_best_by_metrics(self, base_summary: str, aggregated_summary: str,
                                reference: str, original: str) -> tuple:
        """Сравнение метрик базового и агрегированного вариантов"""
        # Молча считаем метрики (подавляем логи судьи)
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        try:
            metrics_base = self._compute_summary_metrics(base_summary, reference, original)
            metrics_agg = self._compute_summary_metrics(aggregated_summary, reference, original)
        finally:
            root_logger.setLevel(old_level)

        sumscore_base = metrics_base.get("SumScore", 0)
        sumscore_agg = metrics_agg.get("SumScore", 0)
        g_eval_base = metrics_base.get("G_Eval", 0)
        g_eval_agg = metrics_agg.get("G_Eval", 0)
        llm_base = metrics_base.get("LLM_Judge", 0)
        llm_agg = metrics_agg.get("LLM_Judge", 0)
        meteor_base = metrics_base.get("METEOR", 0)
        meteor_agg = metrics_agg.get("METEOR", 0)

        print(f"\n  📊 Сравнение метрик:")
        print(f"     ┌───────────────────────────────────────────────────────────┐")
        print(f"     │              │  SumScore │ G-Eval  │ LLM-Judge │ METEOR  │")
        print(f"     ├───────────────────────────────────────────────────────────┤")
        print(f"     │ Базовый      │ {sumscore_base:8.4f}  │ {g_eval_base:6.4f} │ {llm_base:8.1f}  │ {meteor_base:6.4f} │")
        print(f"     │ Агрегирован. │ {sumscore_agg:8.4f}  │ {g_eval_agg:6.4f} │ {llm_agg:8.1f}  │ {meteor_agg:6.4f} │")
        print(f"     └───────────────────────────────────────────────────────────┘")

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
        result_metrics = {"G_Eval": g_eval, "LLM_Judge": llm_judge, "METEOR": meteor, "SumScore": sumscore}

        # BertScore (если включён)
        if BERTSCORE_ENABLED and reference:
            try:
                from metrics.bertscore_calculator import BertScoreCalculator
                bert_calc = BertScoreCalculator()
                bert_result = bert_calc.calculate(reference, summary)
                bertscore = bert_result if isinstance(bert_result, (int, float)) else 0
                result_metrics["BertScore"] = bertscore
            except Exception as e:
                self.logger.warning(f"[SummarizerAggregator] Ошибка BertScore: {e}")

        return result_metrics