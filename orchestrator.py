#!/usr/bin/env python3
"""
Orchestrator - Оркестратор системы агентов на базе LangGraph
Версия 5.7.4 - Исправлен бесконечный цикл кросс-валидации
"""
from typing import TypedDict, Annotated, List, Any, Optional, Dict

from langgraph.graph import StateGraph, END, START

from config import (
    MAX_RETRIES, MAX_CROSS_VALIDATION_ITERATIONS, RETRY_THRESHOLDS,
    BERTSCORE_ENABLED, ADAPTIVE_CORRECTION_ENABLED, MAX_ADAPTIVE_ATTEMPTS,
    TEMP_RETRY_TEMPS, DELTA_LEV_THRESHOLD,
    CROSS_VALIDATION_EARLY_STOP_SUMSCORE,
    REFLECTION_ENABLED, REFLECTION_MIN_SCORE, REFLECTION_MAX_ATTEMPTS, REFLECTION_USE_IN_PROMPT,
    MAX_TOTAL_ITERATIONS, LANGUAGE
)
from metrics.levenstein_calculator import LevenshteinCalculator
from metrics.perplexity_calculator import PerplexityCalculator
from metrics.wer_calculator import WERCalculator
from utils.agent_memory import AgentMemory
from utils.lmstudio_client import LMStudioClient
from utils.logger import setup_logger

logger = setup_logger("Orchestrator")


def last_value(a: Any, b: Any) -> Any:
    return b


class AgentState(TypedDict):
    input_text: Annotated[str, last_value]
    reference_text: Annotated[str, last_value]
    reference_summary: Annotated[str, last_value]
    task_type: Annotated[str, last_value]
    test_id: Annotated[str, last_value]
    detected_language: Annotated[str, last_value]
    summary_language: Annotated[str, last_value]
    prompt_correction: Annotated[str, last_value]
    prompt_summary: Annotated[str, last_value]
    corrected_text: Annotated[str, last_value]
    ensemble_outputs: Annotated[List[str], last_value]
    ensemble_prompts: Annotated[List[str], last_value]
    ensemble_temperatures: Annotated[List[Any], last_value]
    summary_outputs: Annotated[List[str], last_value]
    summary_temperatures: Annotated[List[Any], last_value]
    summary_prompts: Annotated[List[str], last_value]
    top_temps_cor: Annotated[str, last_value]
    top_temps_sum: Annotated[str, last_value]
    summary_text: Annotated[str, last_value]
    metrics_correction: Annotated[Dict[str, Any], last_value]
    metrics_summary: Annotated[Dict[str, Any], last_value]
    perplexity: Annotated[Dict[str, Any], last_value]
    best_temperature: Annotated[str, last_value]
    best_prompt_type: Annotated[str, last_value]
    best_temperature_summary: Annotated[str, last_value]
    best_prompt_summary_type: Annotated[str, last_value]
    needs_retry_correction: Annotated[bool, last_value]
    needs_retry_summary: Annotated[bool, last_value]
    retry_count_correction: Annotated[int, last_value]
    retry_count_summary: Annotated[int, last_value]
    correction_iteration: Annotated[int, last_value]
    cross_validation_iteration: Annotated[int, last_value]
    meaning_preserved: Annotated[bool, last_value]
    aggregator_used: Annotated[bool, last_value]
    previous_corrected_text: Annotated[str, last_value]
    adaptive_config: Annotated[Dict[str, int], last_value]
    best_model: Annotated[str, last_value]
    best_prompt: Annotated[str, last_value]
    aggregation_similarity: Annotated[float, last_value]
    aggregation_reason: Annotated[str, last_value]
    total_iterations: Annotated[int, last_value]
    reflection_score: Annotated[int, last_value]
    reflection_suggestion: Annotated[str, last_value]
    reflection_count_correction: Annotated[int, last_value]
    reflection_count_summary: Annotated[int, last_value]
    reflection_type: Annotated[str, last_value]
    input_text_for_correction: Annotated[str, last_value]
    input_text_for_summary: Annotated[str, last_value]
    summary_metrics_list: Annotated[List[Dict[str, Any]], last_value]
    summary_variants: Annotated[List[str], last_value]


class Orchestrator:
    MAX_CORRECTION_ITERATIONS = 3
    MAX_CROSS_VALIDATION_ITERATIONS = MAX_CROSS_VALIDATION_ITERATIONS
    MAX_TOTAL_ITERATIONS = MAX_TOTAL_ITERATIONS
    MAX_REFLECTIONS = REFLECTION_MAX_ATTEMPTS
    RECURSION_LIMIT = 50

    def __init__(self, client: LMStudioClient, max_retries: int = None,
                 memory: Optional[AgentMemory] = None):
        self.client = client
        self.max_retries = max_retries if max_retries is not None else MAX_RETRIES
        self.memory = memory
        self.wer_calc = WERCalculator()
        self.lev_calc = LevenshteinCalculator()
        self.perplexity_calc = PerplexityCalculator(language=LANGUAGE.lower())
        self.optimization_history = []
        self.graph = self._build_graph()
        logger.info(f"[Orchestrator] Инициализирован v5.7.4 (рефлексия: {REFLECTION_ENABLED})")

    def _build_graph(self) -> StateGraph:
        from agents import (
            CorrectionPromptGenerator,
            CorrectorEnsemble,
            CorrectorAggregator,
            CorrectionJudge,
            SummarizationPromptGenerator,
            SummarizerEnsemble,
            SummarizerAggregator,
            SummarizationJudge,
            ReflectionAgent
        )
        logger.info("[Orchestrator] Построение графа v5.7.4 с рефлексией")

        workflow = StateGraph(AgentState)
        corr_prompt_gen = CorrectionPromptGenerator(self.client, memory=self.memory)
        corrector = CorrectorEnsemble(self.client, memory=self.memory)
        aggregator = CorrectorAggregator(self.client)
        corr_judge = CorrectionJudge()
        reflection_agent = ReflectionAgent(self.client)

        sum_prompt_gen = SummarizationPromptGenerator(self.client, memory=self.memory)
        summarizer = SummarizerEnsemble(self.client, memory=self.memory)
        sum_aggregator = SummarizerAggregator(self.client)
        sum_judge = SummarizationJudge(self.client)

        # Узлы
        workflow.add_node("gen_prompt_corr", lambda s: self._log_step("Генерация промпта коррекции", corr_prompt_gen.execute(s)))
        workflow.add_node("corrector_ensemble", lambda s: self._log_step("Ансамбль корректоров", corrector.execute(s)))
        workflow.add_node("aggregator", lambda s: self._log_step("Агрегатор коррекций", aggregator.execute(s)))
        workflow.add_node("judge_correction", lambda s: self._log_step("Судья коррекции", corr_judge.execute(s)))
        workflow.add_node("reflect_correction", lambda s: self._log_step("Рефлексия коррекции", self._run_reflection(s, "correction", reflection_agent)))
        workflow.add_node("gen_prompt_sum", lambda s: self._log_step("Генерация промпта суммаризации", sum_prompt_gen.execute(s)))
        workflow.add_node("summarizer_ensemble", lambda s: self._log_step("Ансамбль суммаризации", summarizer.execute(s)))
        workflow.add_node("sum_aggregator", lambda s: self._log_step("Агрегатор суммаризации", sum_aggregator.execute(s)))
        workflow.add_node("judge_summary", lambda s: self._log_step("Судья суммаризации", sum_judge.execute(s)))
        workflow.add_node("reflect_summary", lambda s: self._log_step("Рефлексия суммаризации", self._run_reflection(s, "summary", reflection_agent)))
        workflow.add_node("cross_validation_check", lambda s: self._cross_validation_check(s))
        workflow.add_node("increment_counter", lambda s: self._increment_counter(s))

        # Линейные связи
        workflow.add_edge(START, "increment_counter")
        workflow.add_edge("increment_counter", "gen_prompt_corr")
        workflow.add_edge("gen_prompt_corr", "corrector_ensemble")
        workflow.add_edge("corrector_ensemble", "aggregator")
        workflow.add_edge("aggregator", "judge_correction")
        workflow.add_edge("judge_correction", "reflect_correction")

        # Условный переход после рефлексии коррекции
        workflow.add_conditional_edges(
            "reflect_correction",
            self._should_retry_correction_with_reflection,
            {"retry": "corrector_ensemble", "continue": "gen_prompt_sum"}
        )

        workflow.add_edge("gen_prompt_sum", "summarizer_ensemble")
        workflow.add_edge("summarizer_ensemble", "sum_aggregator")
        workflow.add_edge("sum_aggregator", "judge_summary")
        workflow.add_edge("judge_summary", "reflect_summary")
        workflow.add_edge("reflect_summary", "cross_validation_check")

        # Условный переход после кросс-валидации (только условный, без лишнего ребра)
        workflow.add_conditional_edges(
            "cross_validation_check",
            self._should_cross_validate,
            {"retry": "increment_counter", "finish": END}
        )

        return workflow.compile()

    def _run_reflection(self, state: AgentState, reflection_type: str, reflection_agent) -> AgentState:
        if not REFLECTION_ENABLED:
            return state
        temp_state = state.copy()
        temp_state["reflection_type"] = reflection_type
        result = reflection_agent.execute(temp_state)
        state["reflection_score"] = result.get("reflection_score", 5)
        state["reflection_suggestion"] = result.get("reflection_suggestion", "")
        state["reflection_type"] = reflection_type
        if reflection_type == "correction":
            count = state.get("reflection_count_correction", 0) + 1
            state["reflection_count_correction"] = count
        else:
            count = state.get("reflection_count_summary", 0) + 1
            state["reflection_count_summary"] = count
        logger.info(f"[Orchestrator] Рефлексия {reflection_type}: оценка={state['reflection_score']}, попытка={count}")
        return state

    def _should_retry_correction_with_reflection(self, state: AgentState) -> str:
        if not REFLECTION_ENABLED:
            return "continue"
        score = state.get("reflection_score", 5)
        count = state.get("reflection_count_correction", 0)
        max_attempts = REFLECTION_MAX_ATTEMPTS
        if count >= max_attempts:
            logger.info(f"[Orchestrator] Достигнут лимит рефлексий коррекции ({max_attempts})")
            return "continue"
        if score >= REFLECTION_MIN_SCORE:
            logger.info(f"[Orchestrator] Рефлексия коррекции: оценка {score} >= {REFLECTION_MIN_SCORE}, качество достаточное")
            return "continue"
        else:
            logger.info(f"[Orchestrator] Рефлексия коррекции: оценка {score} < {REFLECTION_MIN_SCORE}, требуется повтор")
            suggestion = state.get("reflection_suggestion", "")
            if suggestion and REFLECTION_USE_IN_PROMPT:
                current_prompt = state.get("prompt_correction", "")
                enhanced_prompt = f"{current_prompt}\n\nУЧИТЫВАЙ ЭТО ЗАМЕЧАНИЕ: {suggestion}"
                state["prompt_correction"] = enhanced_prompt
                logger.debug(f"[Orchestrator] Промпт коррекции улучшен на основе рефлексии")
            return "retry"

    def _should_retry_summary_with_reflection(self, state: AgentState) -> str:
        if not REFLECTION_ENABLED:
            return "continue"
        score = state.get("reflection_score", 5)
        count = state.get("reflection_count_summary", 0)
        max_attempts = REFLECTION_MAX_ATTEMPTS
        if count >= max_attempts:
            logger.info(f"[Orchestrator] Достигнут лимит рефлексий суммаризации ({max_attempts})")
            return "continue"
        if score >= REFLECTION_MIN_SCORE:
            logger.info(f"[Orchestrator] Рефлексия суммаризации: оценка {score} >= {REFLECTION_MIN_SCORE}, качество достаточное")
            return "continue"
        else:
            logger.info(f"[Orchestrator] Рефлексия суммаризации: оценка {score} < {REFLECTION_MIN_SCORE}, требуется повтор")
            suggestion = state.get("reflection_suggestion", "")
            if suggestion and REFLECTION_USE_IN_PROMPT:
                current_prompt = state.get("prompt_summary", "")
                enhanced_prompt = f"{current_prompt}\n\nУЧИТЫВАЙ ЭТО ЗАМЕЧАНИЕ: {suggestion}"
                state["prompt_summary"] = enhanced_prompt
                logger.debug(f"[Orchestrator] Промпт суммаризации улучшен на основе рефлексии")
            return "retry"

    def _increment_counter(self, state: AgentState) -> AgentState:
        total = state.get("total_iterations", 0) + 1
        state["total_iterations"] = total
        if total >= self.MAX_TOTAL_ITERATIONS:
            logger.warning(f"[Orchestrator] Достигнут лимит итераций ({self.MAX_TOTAL_ITERATIONS}), принудительное завершение")
            state["meaning_preserved"] = True
            state["cross_validation_iteration"] = self.MAX_CROSS_VALIDATION_ITERATIONS
        return state

    def _should_cross_validate(self, state: AgentState) -> str:
        iteration = state.get("cross_validation_iteration", 0)
        meaning_preserved = state.get("meaning_preserved", True)
        total = state.get("total_iterations", 0)
        if total >= self.MAX_TOTAL_ITERATIONS:
            return "finish"
        if iteration >= self.MAX_CROSS_VALIDATION_ITERATIONS:
            logger.info(f"[Orchestrator] Достигнут лимит кросс-валидации ({self.MAX_CROSS_VALIDATION_ITERATIONS})")
            return "finish"
        if not meaning_preserved:
            state["cross_validation_iteration"] = iteration + 1
            logger.warning(f"[Orchestrator] Кросс-валидация не пройдена (итерация {iteration + 1}/{self.MAX_CROSS_VALIDATION_ITERATIONS}). "
                          f"Повтор не поможет — завершение теста.")
            # НЕ повторяем — повторный запуск того же конвейера даст тот же результат.
            # Механизмы качества (ансамбль, агрегатор, рефлексия) уже отработали.
            return "finish"
        logger.info(f"[Orchestrator] Кросс-валидация пройдена, завершение теста")
        return "finish"

    def _log_step(self, step_name: str, result: dict) -> dict:
        print("\n" + "=" * 80)
        print(f"  🔹 ШАГ: {step_name}")
        print("=" * 80)
        logger.info(f"[Orchestrator] Выполнен шаг: {step_name}")
        return result

    def _cross_validation_check(self, state: AgentState) -> Dict[str, Any]:
        iteration = state.get("cross_validation_iteration", 0)
        print("\n" + "-" * 80)
        print("  📊 КРОСС-ВАЛИДАЦИЯ")
        print("-" * 80)
        print(f"  📊 Итерация: {iteration + 1}/{self.MAX_CROSS_VALIDATION_ITERATIONS}")
        meaning_preserved = self._check_meaning_preserved(state)
        state["meaning_preserved"] = meaning_preserved
        sumscore = state.get("metrics_summary", {}).get("SumScore", 0)
        if sumscore >= CROSS_VALIDATION_EARLY_STOP_SUMSCORE:
            print(f"  ✅ Достигнут высокий SumScore ({sumscore:.3f} >= {CROSS_VALIDATION_EARLY_STOP_SUMSCORE}), досрочное завершение")
            state["meaning_preserved"] = True
            state["cross_validation_iteration"] = self.MAX_CROSS_VALIDATION_ITERATIONS
        if meaning_preserved:
            print(f"  ✅ Смысл сохранён")
        else:
            print(f"  ⚠️ Смысл потерян (BertScore ниже порога) — тест будет завершён без повтора")
        return state

    def _check_meaning_preserved(self, state: AgentState) -> bool:
        original = state.get("corrected_text", "") or state.get("input_text", "")
        summary = state.get("summary_text", "")
        if not original or not summary:
            return True
        if BERTSCORE_ENABLED:
            from metrics.bertscore_calculator import BertScoreCalculator
            similarity = BertScoreCalculator().calculate_p_umfd(original, summary)
        else:
            similarity = self.lev_calc.calculate(original, summary)
        threshold = RETRY_THRESHOLDS.get("min_semantic_similarity", 0.5)
        return similarity >= threshold

    def _adaptive_correction(self, state: AgentState, base_prompt: str) -> AgentState:
        if not ADAPTIVE_CORRECTION_ENABLED:
            return state
        input_text = state.get("input_text", "")
        reference_text = state.get("reference_text", "")
        corrected_text = state.get("corrected_text", "")
        if not input_text or not reference_text or not corrected_text:
            return state
        wer_before = self.wer_calc.calculate(reference_text, input_text)
        lev_before = self.lev_calc.calculate(reference_text, input_text)
        wer_after = self.wer_calc.calculate(reference_text, corrected_text)
        lev_after = self.lev_calc.calculate(reference_text, corrected_text)
        delta_lev = lev_after - lev_before
        if delta_lev >= DELTA_LEV_THRESHOLD:
            return state
        if "{text}" not in base_prompt:
            base_prompt += "\n\nТЕКСТ ДЛЯ КОРРЕКЦИИ:\n{text}\n\nИСПРАВЛЕННЫЙ ТЕКСТ:"
        best_state = state.copy()
        best_corscore = self._calculate_corscore(state, wer_before, lev_before)
        attempts = 0
        for temp in TEMP_RETRY_TEMPS:
            if attempts >= MAX_ADAPTIVE_ATTEMPTS:
                break
            try:
                full_prompt = base_prompt.format(text=input_text)
                response = self.client.generate(prompt=full_prompt, temperature=temp, system_prompt="Ты профессиональный редактор. Возвращай только исправленный текст.")
                if not response or len(response.strip()) < len(input_text.strip()) * 0.3:
                    attempts += 1
                    continue
                new_state = state.copy()
                new_state["corrected_text"] = response
                new_state["best_temperature"] = str(temp)
                new_lev_after = self.lev_calc.calculate(reference_text, response)
                new_wer_after = self.wer_calc.calculate(reference_text, response)
                new_delta_lev = new_lev_after - lev_before
                new_delta_wer = wer_before - new_wer_after
                perplexity_result = self.perplexity_calc.calculate(response, reference_text)
                perplexity = perplexity_result.get("perplexity", 1.0)
                new_state["perplexity"] = perplexity_result
                new_corscore = new_delta_wer + 5*new_delta_lev + (1.0 - perplexity)
                if new_delta_lev > DELTA_LEV_THRESHOLD:
                    new_state["metrics_correction"]["delta_LEV"] = new_delta_lev
                    new_state["metrics_correction"]["LevRating"] = new_lev_after
                    new_state["metrics_correction"]["WER"] = new_wer_after
                    new_state["metrics_correction"]["delta_WER"] = new_delta_wer
                    new_state["metrics_correction"]["CorScore"] = new_corscore
                    return new_state
                if new_corscore > best_corscore:
                    best_state = new_state
                    best_corscore = new_corscore
                attempts += 1
            except Exception as e:
                logger.warning(f"[Orchestrator] Ошибка при температуре {temp}: {e}")
        best_state["metrics_correction"]["CorScore"] = best_corscore
        return best_state

    def _calculate_corscore(self, state: AgentState, wer_before: float, lev_before: float) -> float:
        corrected_text = state.get("corrected_text", "")
        reference_text = state.get("reference_text", "")
        if not corrected_text or not reference_text:
            return 0.0
        wer_after = self.wer_calc.calculate(reference_text, corrected_text)
        lev_after = self.lev_calc.calculate(reference_text, corrected_text)
        delta_wer = wer_before - wer_after
        delta_lev = lev_after - lev_before
        perplexity = state.get("perplexity", {}).get("perplexity", 1.0)
        return delta_wer + 5*delta_lev + (1.0 - perplexity)

    def execute(self, initial_state: AgentState) -> AgentState:
        test_id = initial_state.get("test_id", "unknown")
        defaults = {
            "retry_count_correction": 0, "retry_count_summary": 0,
            "cross_validation_iteration": 0, "meaning_preserved": True,
            "adaptive_config": {}, "correction_iteration": 0,
            "perplexity": {}, "best_temperature": "N/A",
            "best_prompt_type": "базовый",
            "metrics_correction": {}, "previous_corrected_text": "",
            "needs_retry_correction": False, "needs_retry_summary": False,
            "aggregator_used": False, "aggregation_similarity": 0.0,
            "aggregation_reason": "", "best_model": self.client.model if hasattr(self.client, 'model') else "unknown",
            "best_prompt": "", "ensemble_outputs": [], "ensemble_prompts": [], "ensemble_temperatures": [], "summary_outputs": [],
            "summary_temperatures": [], "summary_prompts": [],
            "top_temps_cor": "", "top_temps_sum": "",
            "best_temperature_summary": "N/A", "best_prompt_summary_type": "",
            "detected_language": "ru", "summary_language": "ru",
            "reflection_suggestion": "", "total_iterations": 0,
            "reflection_count_correction": 0, "reflection_count_summary": 0,
            "reflection_score": 5, "reflection_type": "",
            "summary_metrics_list": [],
            "input_text_for_correction": initial_state.get("input_text", ""),
            "input_text_for_summary": initial_state.get("corrected_text", "") or initial_state.get("input_text", ""),
            "summary_variants": []
        }
        for key, default in defaults.items():
            if initial_state.get(key) is None:
                initial_state[key] = default

        print("\n" + "█"*80 + f"\n█  🚀 ЗАПУСК ТЕСТА: {test_id} (v5.7.4 с рефлексией)".center(80) + "\n" + "█"*80 + "\n")
        logger.info(f"[Orchestrator] ЗАПУСК ТЕСТА: {test_id} (v5.7.4)")

        try:
            final_state = self.graph.invoke(initial_state)
            base_prompt = final_state.get("prompt_correction", "")
            if base_prompt:
                final_state = self._adaptive_correction(final_state, base_prompt)

            print("\n" + "█"*80 + f"\n█  ✅ ТЕСТ {test_id} ЗАВЕРШЕН УСПЕШНО".center(80) + "\n" + "█"*80 + "\n")
            return final_state
        except Exception as e:
            print("\n" + "█"*80 + f"\n█  ❌ ОШИБКА ТЕСТА {test_id}".center(80) + f"\n█  {str(e)[:60]}".center(80) + "\n" + "█"*80 + "\n")
            logger.error(f"[Orchestrator] ОШИБКА ТЕСТА {test_id}: {e}")
            raise

    def close(self):
        self.client.close()
        if self.memory:
            self.memory._save_memory()
            logger.info("[Orchestrator] Память сохранена")