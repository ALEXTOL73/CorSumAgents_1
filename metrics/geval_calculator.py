"""
Калькулятор метрики G-Eval (расширенная версия с 5 критериями)
Версия 3.5 - Создан на основе g-eval.py

Особенности:
- 5 критериев оценки: Coherence, Consistency, Fluency, Relevance, Conciseness
- Взвешенный общий score
- Подробное объяснение для каждого критерия
- Защита от ошибок парсинга JSON
- Измерение времени обработки
"""
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from config import LANGUAGE
from utils.lmstudio_client import LMStudioClient
from utils.logger import setup_logger

logger = setup_logger("GEvalCalculator", "geval_calculator")


@dataclass
class GEvalResult:
    """Результат G-Eval оценки"""
    coherence: float  # Связность (0-1)
    consistency: float  # Согласованность (0-1)
    fluency: float  # Беглость (0-1)
    relevance: float  # Релевантность (0-1)
    conciseness: float  # Лаконичность (0-1)
    overall_score: float  # Общий score (0-1)
    explanation: str  # Объяснение
    processing_time: float  # Время обработки (сек)


class GEvalCalculator:
    """
    Расширенная G-Eval метрика с 5 критериями оценки

    Критерии:
    - COHERENCE (связность) - логичность изложения
    - CONSISTENCY (согласованность) - соответствие исходному тексту
    - FLUENCY (беглость) - грамматическая корректность
    - RELEVANCE (релевантность) - сохранение ключевой информации
    - CONCISENESS (лаконичность) - отсутствие избыточности

    Веса критериев:
    - Coherence: 0.25
    - Consistency: 0.30
    - Fluency: 0.15
    - Relevance: 0.20
    - Conciseness: 0.10
    """

    WEIGHTS = {
        "coherence": 0.25,
        "consistency": 0.30,
        "fluency": 0.15,
        "relevance": 0.20,
        "conciseness": 0.10,
    }

    def __init__(self, client: LMStudioClient, model_name: str = "local-model"):
        """
        Инициализация калькулятора

        Args:
            client: Клиент LM Studio
            model_name: Имя модели для оценки
        """
        self.client = client
        self.model_name = model_name

    def _clean_json_string(self, s: str) -> str:
        """
        Очистка JSON строки от управляющих символов

        Args:
            s: Исходная строка

        Returns:
            Очищенная строка
        """
        # Удаляем управляющие символы (коды 0-31), но оставляем перевод строки и табуляцию
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', s)
        # Заменяем переводы строк и табуляцию на пробелы
        cleaned = re.sub(r'[\n\r\t]', ' ', cleaned)
        # Сжимаем множественные пробелы
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

    def _extract_json(self, response: str) -> Dict:
        """
        Извлечение JSON из ответа LLM

        Args:
            response: Ответ от LLM

        Returns:
            Словарь с извлечёнными данными
        """
        # Ищем JSON-блок между фигурными скобками
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            logger.warning("[G-Eval] Не найден JSON блок в ответе")
            return {}

        json_str = match.group()

        # Пробуем очистить и распарсить
        cleaned = self._clean_json_string(json_str)
        try:
            parsed = json.loads(cleaned)
            # Ensure all numeric values are actually floats, not dicts or other types
            for key in ["coherence", "consistency", "fluency", "relevance", "conciseness"]:
                if key in parsed:
                    val = parsed[key]
                    if not isinstance(val, (int, float)):
                        logger.warning(f"[G-Eval] Non-numeric value for {key}: {type(val).__name__}, converting to 0.5")
                        parsed[key] = 0.5
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"[G-Eval] Ошибка парсинга JSON после очистки: {e}")

        # Fallback: извлекаем значения вручную с помощью регулярных выражений
        fields = {}

        # Числовые поля
        num_pattern = r'"(\w+)":\s*([0-9.]+)'
        for m in re.finditer(num_pattern, cleaned):
            key = m.group(1)
            try:
                fields[key] = float(m.group(2))
            except ValueError:
                pass

        # Строковые поля (обычные)
        str_pattern = r'"(\w+)":\s*"([^"]*)"'
        for m in re.finditer(str_pattern, cleaned):
            key = m.group(1)
            fields[key] = m.group(2)

        # Поле explanation может содержать кавычки, поэтому обрабатываем отдельно
        expl_pattern = r'"explanation":\s*"([^"]*)"'
        expl_match = re.search(expl_pattern, cleaned, re.DOTALL)
        if expl_match:
            fields["explanation"] = expl_match.group(1)

        # Если собрали все основные поля, возвращаем
        required = ["coherence", "consistency", "fluency", "relevance", "conciseness"]
        if all(k in fields for k in required):
            return fields

        logger.warning("[G-Eval] Не все обязательные поля найдены в JSON")
        return {}

    def evaluate(
            self,
            summary: str,
            original_text: str,
            reference: Optional[str] = None
    ) -> GEvalResult:
        """
        Оценка суммаризации по 5 критериям

        Args:
            summary: Сгенерированная суммаризация
            original_text: Исходный текст
            reference: Эталонная суммаризация (опционально)

        Returns:
            GEvalResult с оценками по всем критериям
        """
        start_time = time.time()

        # Проверка на пустые значения
        if not summary:
            logger.warning("[G-Eval] Пустая суммаризация")
            return GEvalResult(
                coherence=0.0, consistency=0.0, fluency=0.0,
                relevance=0.0, conciseness=0.0, overall_score=0.0,
                explanation="Пустая суммаризация",
                processing_time=time.time() - start_time
            )

        if not original_text:
            logger.warning("[G-Eval] Пустой исходный текст")
            return GEvalResult(
                coherence=0.5, consistency=0.5, fluency=0.5,
                relevance=0.5, conciseness=0.5, overall_score=0.5,
                explanation="Пустой исходный текст",
                processing_time=time.time() - start_time
            )

        # Формирование промпта с учетом языка
        # ✅ Используем LANGUAGE из config
        is_russian = LANGUAGE.lower() == 'ru'
        
        if is_russian:
            system_prompt = """Ты — эксперт по оценке качества суммаризации текстов.
Оцени суммаризацию по критериям от 0 до 1.
Выведи JSON с оценками."""

            reference_text = reference if reference else "Эталон не предоставлен"

            user_prompt = f"""Оцени суммаризацию по критериям:

Критерии:
- COHERENCE (связность) - логичность изложения, плавность перехода между идеями
- CONSISTENCY (согласованность) - соответствие исходному тексту, отсутствие противоречий
- FLUENCY (беглость) - грамматическая корректность, естественность языка
- RELEVANCE (релевантность) - сохранение ключевой информации, отсечение второстепенного
- CONCISENESS (лаконичность) - отсутствие избыточности, краткость изложения

ИСХОДНЫЙ ТЕКСТ:
{original_text[:4000]}

СУММАРИЗАЦИЯ:
{summary[:2000]}

ЭТАЛОННАЯ СУММАРИЗАЦИЯ:
{reference_text[:2000]}

Выведи JSON:
{{
    "coherence": 0.XX,
    "consistency": 0.XX,
    "fluency": 0.XX,
    "relevance": 0.XX,
    "conciseness": 0.XX,
    "explanation": "обоснование оценок"
}}
"""
        else:
            system_prompt = """You are an expert in text summarization quality assessment.
Evaluate the summary on criteria from 0 to 1.
Output JSON with scores."""

            reference_text = reference if reference else "Golden reference not provided"

            user_prompt = f"""Evaluate the summary based on criteria:

Criteria:
- COHERENCE - logical flow, smooth transitions between ideas
- CONSISTENCY - alignment with source text, no contradictions
- FLUENCY - grammatical correctness, natural language
- RELEVANCE - retention of key information, filtering of secondary content
- CONCISENESS - no redundancy, brevity of expression

ORIGINAL TEXT:
{original_text[:4000]}

SUMMARY:
{summary[:2000]}

GOLDEN SUMMARY:
{reference_text[:2000]}

Output JSON:
{{
    "coherence": 0.XX,
    "consistency": 0.XX,
    "fluency": 0.XX,
    "relevance": 0.XX,
    "conciseness": 0.XX,
    "explanation": "justification for scores"
}}
"""

        try:
            response = self.client.generate(
                prompt=user_prompt,
                temperature=0.1,
                system_prompt=system_prompt,
                max_tokens=1024
            )

            result = self._extract_json(response)

            if not result:
                logger.warning("[G-Eval] Не удалось извлечь JSON, используем значения по умолчанию")
                result = {}

            # Расчет общего score с весами
            overall = sum(
                result.get(c, 0.5) * self.WEIGHTS.get(c, 0) for c in self.WEIGHTS.keys()
            )

            processing_time = time.time() - start_time

            logger.info(f"[G-Eval] Оценка завершена за {processing_time:.2f}с, overall={overall:.4f}")
            logger.debug(f"[G-Eval] Детальные оценки: {result}")

            return GEvalResult(
                coherence=result.get("coherence", 0.5),
                consistency=result.get("consistency", 0.5),
                fluency=result.get("fluency", 0.5),
                relevance=result.get("relevance", 0.5),
                conciseness=result.get("conciseness", 0.5),
                overall_score=overall,
                explanation=result.get("explanation", "Нет объяснения"),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"[G-Eval] Ошибка оценки: {e}", exc_info=True)
            processing_time = time.time() - start_time

            return GEvalResult(
                coherence=0.5, consistency=0.5, fluency=0.5,
                relevance=0.5, conciseness=0.5, overall_score=0.5,
                explanation=f"Ошибка: {e}",
                processing_time=processing_time
            )

    def evaluate_with_golden(
            self,
            summary: str,
            golden: str
    ) -> GEvalResult:
        """
        Оценка суммаризации через сравнение с эталоном

        Args:
            summary: Сгенерированная суммаризация
            golden: Эталонная суммаризация

        Returns:
            GEvalResult с оценками по всем критериям
        """
        start_time = time.time()
        
        # ✅ Используем LANGUAGE из config
        is_russian = LANGUAGE.lower() == 'ru'

        if is_russian:
            system_prompt = "Ты — эксперт по сравнению суммаризаций."

            user_prompt = f"""Сравни суммаризацию с эталоном.

ЭТАЛОН:
{golden[:2000]}

СУММАРИЗАЦИЯ:
{summary[:2000]}

Оцени по критериям от 0 до 1. Выведи JSON:
{{
    "coherence": 0.XX,
    "consistency": 0.XX,
    "fluency": 0.XX,
    "relevance": 0.XX,
    "conciseness": 0.XX,
    "explanation": "обоснование оценок"
}}
"""
        else:
            system_prompt = "You are an expert in comparing summaries."

            user_prompt = f"""Compare the summary with the golden reference.

GOLDEN REFERENCE:
{golden[:2000]}

SUMMARY:
{summary[:2000]}

Evaluate on criteria from 0 to 1. Output JSON:
{{
    "coherence": 0.XX,
    "consistency": 0.XX,
    "fluency": 0.XX,
    "relevance": 0.XX,
    "conciseness": 0.XX,
    "explanation": "justification for scores"
}}
"""

        try:
            response = self.client.generate(
                prompt=user_prompt,
                temperature=0.1,
                system_prompt=system_prompt,
                max_tokens=1024
            )

            result = self._extract_json(response)

            if not result:
                result = {}

            overall = sum(
                result.get(c, 0.5) * self.WEIGHTS.get(c, 0) for c in self.WEIGHTS.keys()
            )

            processing_time = time.time() - start_time

            return GEvalResult(
                coherence=result.get("coherence", 0.5),
                consistency=result.get("consistency", 0.5),
                fluency=result.get("fluency", 0.5),
                relevance=result.get("relevance", 0.5),
                conciseness=result.get("conciseness", 0.5),
                overall_score=overall,
                explanation=result.get("explanation", "Нет объяснения"),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"[G-Eval] Ошибка оценки с эталоном: {e}", exc_info=True)
            processing_time = time.time() - start_time

            return GEvalResult(
                coherence=0.5, consistency=0.5, fluency=0.5,
                relevance=0.5, conciseness=0.5, overall_score=0.5,
                explanation=f"Ошибка: {e}",
                processing_time=processing_time
            )

    def to_dict(self, result: GEvalResult) -> Dict[str, Any]:
        """
        Конвертация результата в словарь

        Args:
            result: GEvalResult

        Returns:
            Словарь с данными
        """
        return {
            "coherence": result.coherence,
            "consistency": result.consistency,
            "fluency": result.fluency,
            "relevance": result.relevance,
            "conciseness": result.conciseness,
            "overall_score": result.overall_score,
            "explanation": result.explanation,
            "processing_time": result.processing_time,
        }

    def get_detailed_report(self, result: GEvalResult) -> str:
        """
        Генерация подробного отчёта об оценке

        Args:
            result: GEvalResult

        Returns:
            Строка с подробным отчётом
        """
        report = f"""
================================================================================
G-EVAL ДЕТАЛЬНЫЙ ОТЧЁТ
================================================================================

ОБЩИЙ SCORE: {result.overall_score:.4f} (из 1.0)
ВРЕМЯ ОБРАБОТКИ: {result.processing_time:.2f} сек

КРИТЕРИИ ОЦЕНКИ:
┌──────────────────────────────────────────────────────────────────────────────┐
│ Критерий          │ Оценка  │ Вес   │ Взвешенный вклад                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Coherence         │ {result.coherence:.4f}    │ 0.25  │ {result.coherence * 0.25:.4f}                          │
│ Consistency       │ {result.consistency:.4f}    │ 0.30  │ {result.consistency * 0.30:.4f}                          │
│ Fluency           │ {result.fluency:.4f}    │ 0.15  │ {result.fluency * 0.15:.4f}                          │
│ Relevance         │ {result.relevance:.4f}    │ 0.20  │ {result.relevance * 0.20:.4f}                          │
│ Conciseness       │ {result.conciseness:.4f}    │ 0.10  │ {result.conciseness * 0.10:.4f}                          │
└──────────────────────────────────────────────────────────────────────────────┘

ИНТЕРПРЕТАЦИЯ:
"""

        if result.overall_score >= 0.8:
            report += "  🏆 ОТЛИЧНО - Высококачественная суммаризация\n"
        elif result.overall_score >= 0.6:
            report += "  ✅ ХОРОШО - Качественная суммаризация с минорными недочётами\n"
        elif result.overall_score >= 0.4:
            report += "  ⚠️  УДОВЛЕТВОРИТЕЛЬНО - Требуются улучшения\n"
        else:
            report += "  ❌ НИЗКОЕ КАЧЕСТВО - Требуется существенная переработка\n"

        report += f"""
ОБЪЯСНЕНИЕ:
{result.explanation}

================================================================================
"""

        return report