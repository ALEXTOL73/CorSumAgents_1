"""
Калькулятор метрики G-Eval через LLM-судью (базовая версия)
Версия 3.6 - Исправлен парсинг JSON с fallback для обрезанных ответов
Особенности:
- Оценка качества суммаризации через LLM-судью
- JSON формат ответа с оценкой и объяснением
- ✅ Улучшенный парсинг JSON с восстановлением обрезанных ответов
- Защита от ошибок парсинга JSON
- Детальное логирование
"""
from typing import Dict, Any
from utils.logger import setup_logger
from utils.lmstudio_client import LMStudioClient

logger = setup_logger("LLMJudgeCalculator", "llm_judge_calculator")


class LLMJudgeCalculator:
    """
    Оценка качества суммаризации через LLM-судью (базовая версия)
    Использует LLM для оценки суммаризации по шкале 1-10
    с подробным объяснением оценки.
    """

    def __init__(self, client: LMStudioClient):
        """
        Инициализация калькулятора

        Args:
            client: Клиент LM Studio
        """
        self.client = client

    def evaluate(
            self,
            original: str,
            summary: str,
            reference: str
    ) -> Dict[str, Any]:
        """
        Оценка суммаризации через LLM

        Args:
            original: Исходный текст
            summary: Сгенерированная суммаризация
            reference: Эталонная суммаризация

        Returns:
            Словарь с оценкой и объяснением
        """
        logger.info("[LLM-Judge] Запуск оценки через LLM-судью")

        # Проверка на пустые значения
        if not original:
            logger.warning("[LLM-Judge] Пустой исходный текст")
            return {"score": 5, "explanation": "Пустой исходный текст"}

        if not summary:
            logger.warning("[LLM-Judge] Пустая суммаризация")
            return {"score": 1, "explanation": "Пустая суммаризация"}

        if not reference:
            logger.warning("[LLM-Judge] Пустой эталон, используем только оригинал")
            reference = "Эталон не предоставлен"

        # Обрезаем длинные тексты для экономии токенов
        original_trimmed = original[:500] + ('...' if len(original) > 500 else '')

        prompt = f"""
Ты - профессиональный судья качества суммаризации текстов.

ИСХОДНЫЙ ТЕКСТ:
{original_trimmed}

СУММАРИЗАЦИЯ:
{summary}

ЭТАЛОННАЯ СУММАРИЗАЦИЯ:
{reference}

Критерии оценки (1-10):
1-3: Бессвязный текст, потеря смысла
4-6: Частичная передача смысла, есть ошибки
7-8: Хорошая передача смысла, минорные недочеты
9-10: Идеальная суммаризация, полное соответствие эталону

Верни ответ ТОЛЬКО в формате JSON:
{{"score": <число 1-10>, "explanation": "<краткое объяснение 2-3 предложения>"}}
"""
        system_prompt = "Отвечай ТОЛЬКО в формате JSON без дополнительного текста"

        # ✅ ИСПРАВЛЕНИЕ v3.6: Увеличиваем max_tokens для полного ответа
        response = self.client.generate_json(
            prompt,
            temperature=0.2,
            system_prompt=system_prompt,
            max_tokens=1024  # ✅ УВЕЛИЧЕНО с 512 до 1024
        )

        # Валидация ответа
        if "error" in response:
            logger.warning("[LLM-Judge] Ошибка парсинга, используем оценку по умолчанию")
            return {
                "score": 5,
                "explanation": "Не удалось получить оценку от судьи",
                "raw_response": response.get("raw", "")
            }

        score = response.get("score", 5)
        explanation = response.get("explanation", "Нет объяснения")

        # Валидация диапазона
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            logger.warning(f"[LLM-Judge] Неверный диапазон оценки: {score}")
            score = 5

        logger.info(f"[LLM-Judge] Оценка: {score}/10")
        logger.debug(f"[LLM-Judge] Объяснение: {explanation}")

        return {
            "score": int(score),
            "explanation": explanation,
            "raw_response": response
        }