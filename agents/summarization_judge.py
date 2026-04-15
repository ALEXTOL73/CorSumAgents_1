"""
Агент-судья для оценки качества суммаризации
Версия 5.2.14 - Использование BertScoreCalculator (синглтон) вместо прямого вызова bert_score
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from metrics.meteor_calculator import METEORCalculator
from metrics.geval_calculator import GEvalCalculator
from metrics.LLM_judge_calculator import LLMJudgeCalculator
from metrics.bertscore_calculator import BertScoreCalculator  # ✅ добавили импорт синглтона
from config import BERTSCORE_ENABLED, SUMSCORE_ENABLED, SUMSCORE_WEIGHTS
from utils.logger import setup_logger

logger = setup_logger("SummarizationJudge")


class SummarizationJudge(BaseAgent):
    """Оценка качества суммаризации с использованием BertScoreCalculator (синглтон)"""

    def __init__(self, client: LMStudioClient):
        super().__init__(client, "SummarizationJudge")
        self.meteor_calc = METEORCalculator()
        self.g_eval = GEvalCalculator(client)
        self.llm_judge = LLMJudgeCalculator(client)
        # ✅ Используем глобальный синглтон BertScoreCalculator (модель загружается один раз)
        self.bert_calc = BertScoreCalculator() if BERTSCORE_ENABLED else None

        if BERTSCORE_ENABLED:
            logger.info(f"[SummarizationJudge] BertScore: ✅ ВКЛЮЧЁН (через синглтон BertScoreCalculator)")
        else:
            logger.info(f"[SummarizationJudge] BertScore: ❌ ОТКЛЮЧЁН")

        if SUMSCORE_ENABLED:
            logger.info(f"[SummarizationJudge] SumScore: ✅ ВКЛЮЧЁН")
            logger.info(f"[SummarizationJudge] Веса: G-Eval={SUMSCORE_WEIGHTS['g_eval']}, LLM-Judge={SUMSCORE_WEIGHTS['llm_judge']}, METEOR={SUMSCORE_WEIGHTS['meteor']}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("=" * 60)
        self.log_execution("НАЧАЛО ОЦЕНКИ СУММАРИЗАЦИИ")
        self.log_execution("=" * 60)

        original = state.get("input_text", "") or state.get("corrected_text", "")
        summary = state.get("summary_text", "")
        reference = state.get("reference_summary", "")

        self.logger.info(f"\n📊 Исходный текст: {len(original)} символов")
        self.logger.info(f"📊 Суммаризация: {len(summary)} символов")
        self.logger.info(f"📊 Эталонное резюме: {len(reference)} символов")

        compression_ratio = len(summary) / len(original) if len(original) > 0 else 0
        self.logger.info(f"📊 Коэффициент сжатия: {compression_ratio:.2%}")

        # METEOR
        self.log_execution("--- METEOR МЕТРИКА ---")
        meteor_metrics = self.meteor_calc.compute_all_metrics(reference, summary)
        meteor_score = meteor_metrics.get("meteor", 0.0)
        meteor_interpretation = meteor_metrics.get("meteor_interpretation", "N/A")
        self.logger.info(f"  METEOR Score: {meteor_score:.4f}")

        # G-EVAL
        self.log_execution("--- G-EVAL ОЦЕНКА ---")
        try:
            g_eval_raw = self.g_eval.evaluate(summary, original, reference)
            g_eval_result = self._ensure_object(g_eval_raw)
            self.logger.info(f"  G-Eval Overall: {g_eval_result.overall_score:.4f}")
        except Exception as e:
            self.logger.error(f"Ошибка G-Eval: {e}")
            g_eval_result = type('obj', (object,), {
                'overall_score': 0.5, 'coherence': 0.5, 'consistency': 0.5,
                'fluency': 0.5, 'relevance': 0.5, 'conciseness': 0.5,
                'explanation': f"Ошибка: {e}", 'processing_time': 0.0
            })()

        # BERTSCORE (через синглтон)
        self.log_execution("--- BERTSCORE МЕТРИКА ---")
        bertscore_score = None
        if BERTSCORE_ENABLED and self.bert_calc is not None:
            try:
                # ✅ Используем синглтон BertScoreCalculator (модель загружается один раз)
                bertscore_score = self.bert_calc.calculate(reference, summary)
                self.logger.info(f"  BertScore: {bertscore_score:.4f}")
            except Exception as e:
                self.logger.error(f"Ошибка BertScore: {e}")
                bertscore_score = 0.0
        else:
            self.logger.info(f"  BertScore: ⏭️ Пропущено (недоступно)")

        # LLM-JUDGE
        self.log_execution("--- LLM-JUDGE ОЦЕНКА ---")
        llm_judge_result = self.llm_judge.evaluate(original, summary, reference)
        llm_score = llm_judge_result.get("score", 5)
        llm_explanation = llm_judge_result.get("explanation", "Нет объяснения")
        self.logger.info(f"  LLM-Judge: {llm_score}/10")

        # SUMSCORE
        self.log_execution("--- SUMSCORE (ГЛАВНАЯ МЕТРИКА) ---")
        if SUMSCORE_ENABLED:
            g_eval_normalized = g_eval_result.overall_score
            llm_judge_normalized = llm_score / 10.0
            meteor_normalized = meteor_score

            sumscore = (
                g_eval_normalized * SUMSCORE_WEIGHTS["g_eval"] +
                llm_judge_normalized * SUMSCORE_WEIGHTS["llm_judge"] +
                meteor_normalized * SUMSCORE_WEIGHTS["meteor"]
            )
            sumscore = max(0.0, min(1.0, sumscore))
            sumscore_assessment = self._get_sumscore_assessment(sumscore)

            self.logger.info(f"  SumScore: {sumscore:.3f}/1.000 ({sumscore_assessment})")
        else:
            sumscore = None
            sumscore_assessment = "N/A"

        # Итоговая оценка
        detailed_explanation = self._generate_detailed_explanation(
            llm_score, g_eval_result, meteor_score, summary, reference,
            compression_ratio, bertscore_score
        )
        final_assessment = self._calculate_final_assessment(
            llm_score, g_eval_result.overall_score, meteor_score, bertscore_score
        )
        self.logger.info(f"  {final_assessment}")

        current_retry = state.get("retry_count_summary", 0)
        needs_retry = (sumscore is not None and sumscore < 0.55) and current_retry < 3

        metrics_data = {
            "llm_score": llm_score,
            "llm_explanation": llm_explanation,
            "g_eval_overall": round(g_eval_result.overall_score, 4),
            "g_eval_coherence": round(g_eval_result.coherence, 4),
            "g_eval_consistency": round(g_eval_result.consistency, 4),
            "g_eval_fluency": round(g_eval_result.fluency, 4),
            "g_eval_relevance": round(g_eval_result.relevance, 4),
            "g_eval_conciseness": round(g_eval_result.conciseness, 4),
            "g_eval_processing_time": round(g_eval_result.processing_time, 2),
            "g_eval_explanation": g_eval_result.explanation,
            "meteor": round(meteor_score, 4),
            "meteor_interpretation": meteor_interpretation,
            "detailed_explanation": detailed_explanation,
            "final_assessment": final_assessment,
            "compression_ratio": round(compression_ratio, 4),
        }

        if BERTSCORE_ENABLED and bertscore_score is not None:
            metrics_data["bertscore"] = round(bertscore_score, 4)
        else:
            metrics_data["bertscore"] = None

        if SUMSCORE_ENABLED and sumscore is not None:
            metrics_data["sumscore"] = round(sumscore, 3)
            metrics_data["sumscore_assessment"] = sumscore_assessment
        else:
            metrics_data["sumscore"] = None

        self.log_execution("=" * 60)
        self.log_execution("ОЦЕНКА СУММАРИЗАЦИИ ЗАВЕРШЕНА")
        self.log_execution("=" * 60)

        return {
            "metrics_summary": metrics_data,
            "needs_retry_summary": needs_retry,
            "sumscore": sumscore,
            "sumscore_assessment": sumscore_assessment
        }

    # ---------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ----------
    def _ensure_object(self, maybe_dict):
        if hasattr(maybe_dict, 'overall_score'):
            return maybe_dict
        if isinstance(maybe_dict, dict):
            obj = type('GEvalResult', (object,), {})
            obj.overall_score = maybe_dict.get('overall_score', 0.5)
            obj.coherence = maybe_dict.get('coherence', 0.5)
            obj.consistency = maybe_dict.get('consistency', 0.5)
            obj.fluency = maybe_dict.get('fluency', 0.5)
            obj.relevance = maybe_dict.get('relevance', 0.5)
            obj.conciseness = maybe_dict.get('conciseness', 0.5)
            obj.explanation = maybe_dict.get('explanation', '')
            obj.processing_time = maybe_dict.get('processing_time', 0.0)
            return obj
        return maybe_dict

    # ✅ Метод _compute_bertscore больше не нужен — используем self.bert_calc.calculate()
    # Оставляем заглушку для совместимости, если где-то вызывается (но в коде не используется)
    def _compute_bertscore(self, ref: str, cand: str) -> float:
        """Заглушка для обратной совместимости (используйте self.bert_calc.calculate)"""
        if self.bert_calc:
            return self.bert_calc.calculate(ref, cand)
        return 0.0

    def _get_sumscore_assessment(self, sumscore: float) -> str:
        if sumscore >= 0.8:
            return "⭐⭐⭐⭐⭐ ОТЛИЧНО"
        elif sumscore >= 0.65:
            return "⭐⭐⭐⭐ ХОРОШО"
        elif sumscore >= 0.55:
            return "⭐⭐⭐ УДОВЛЕТВОРИТЕЛЬНО"
        elif sumscore >= 0.45:
            return "⭐⭐ ТРЕБУЕТСЯ УЛУЧШЕНИЕ"
        else:
            return "⭐ НЕУДОВЛЕТВОРИТЕЛЬНО"

    def _generate_detailed_explanation(self, llm_score: int, g_eval_result,
                                       meteor_score: float, summary: str,
                                       reference: str, compression: float,
                                       bertscore_score: float = None) -> str:
        g_eval_score_10 = g_eval_result.overall_score * 10
        if SUMSCORE_ENABLED:
            avg_score = (
                g_eval_result.overall_score * SUMSCORE_WEIGHTS["g_eval"] +
                (llm_score / 10.0) * SUMSCORE_WEIGHTS["llm_judge"] +
                meteor_score * SUMSCORE_WEIGHTS["meteor"]
            ) * 10
        else:
            avg_score = (llm_score + g_eval_score_10) / 2

        if avg_score >= 9:
            base = "🏆 ВЫСОЧАЙШЕЕ КАЧЕСТВО"
        elif avg_score >= 7:
            base = "✅ ХОРОШЕЕ КАЧЕСТВО"
        elif avg_score >= 5:
            base = "⚠️  СРЕДНЕЕ КАЧЕСТВО"
        elif avg_score >= 3:
            base = "❌ НИЗКОЕ КАЧЕСТВО"
        else:
            base = "🚫 КРИТИЧЕСКОЕ КАЧЕСТВО"

        detailed = f"{base}:\n"
        detailed += f"  LLM-Judge: {llm_score}/10\n"
        detailed += f"  G-Eval: {g_eval_result.overall_score:.2f}\n"
        detailed += f"  METEOR: {meteor_score:.4f}\n"
        if BERTSCORE_ENABLED and bertscore_score is not None:
            detailed += f"  BertScore: {bertscore_score:.4f}\n"
        detailed += f"  Сжатие: {compression:.1%}"
        return detailed

    def _calculate_final_assessment(self, llm_score: int, g_eval_overall: float,
                                    meteor_score: float, bertscore_score: float = None) -> str:
        llm_normalized = llm_score / 10.0
        weights = {"llm": 0.30, "g_eval": 0.4, "meteor": 0.6}
        if BERTSCORE_ENABLED and bertscore_score is not None:
            weights["bertscore"] = 0.3
        else:
            weights["meteor"] = 0.30

        combined = (
            llm_normalized * weights["llm"] +
            g_eval_overall * weights["g_eval"] +
            meteor_score * weights["meteor"]
        )
        if BERTSCORE_ENABLED and bertscore_score is not None:
            combined += bertscore_score * weights["bertscore"]

        if combined >= 0.85:
            return f"🏆 ОТЛИЧНО (score: {combined:.2f})"
        elif combined >= 0.70:
            return f"✅ ХОРОШО (score: {combined:.2f})"
        elif combined >= 0.55:
            return f"⚠️  УДОВЛЕТВОРИТЕЛЬНО (score: {combined:.2f})"
        else:
            return f"❌ ТРЕБУЕТСЯ УЛУЧШЕНИЕ (score: {combined:.2f})"

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] {message}")