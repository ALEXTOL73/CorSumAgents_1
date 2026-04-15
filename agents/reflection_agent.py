#!/usr/bin/env python3
"""
Агент саморефлексии для оценки и улучшения результатов
Версия 1.0 - Анализ и предложения по улучшению
"""
from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.logger import setup_logger

logger = setup_logger("ReflectionAgent")


class ReflectionAgent(BaseAgent):
    """
    Агент саморефлексии: оценивает результат и предлагает улучшения
    """

    REFLECTION_PROMPT_CORRECTION = """Ты - эксперт по оценке качества исправления текста. Проанализируй результат.

ОРИГИНАЛЬНЫЙ ТЕКСТ (с ошибками):
{original}

ИСПРАВЛЕННЫЙ ТЕКСТ:
{corrected}

ЭТАЛОННЫЙ ТЕКСТ (идеальный вариант):
{reference}

ЗАДАНИЕ:
1. Оцени качество исправления по шкале от 1 до 10 (1 = ужасно, 10 = идеально).
2. Укажи, какие ошибки остались или были внесены новые.
3. Предложи конкретные улучшения для следующей попытки.

ФОРМАТ ОТВЕТА (строго соблюдай):
ОЦЕНКА: <число от 1 до 10>
ОШИБКИ: <краткий список>
ПРЕДЛОЖЕНИЕ: <конкретное указание, что изменить в промпте или подходе>"""

    REFLECTION_PROMPT_SUMMARY = """Ты - эксперт по оценке качества суммаризации. Проанализируй резюме.

ИСХОДНЫЙ ТЕКСТ:
{original}

СОЗДАННОЕ РЕЗЮМЕ:
{summary}

ЭТАЛОННОЕ РЕЗЮМЕ (идеал):
{reference}

ЗАДАНИЕ:
1. Оцени качество резюме по шкале от 1 до 10 (1 = неудовлетворительно, 10 = отлично).
2. Укажи, чего не хватает (ключевые факты, связность, сжатие) или что лишнее.
3. Предложи конкретные улучшения для следующей попытки.

ФОРМАТ ОТВЕТА (строго соблюдай):
ОЦЕНКА: <число от 1 до 10>
НЕДОСТАТКИ: <краткий список>
ПРЕДЛОЖЕНИЕ: <конкретное указание, что изменить в промпте или подходе>"""

    def __init__(self, client: LMStudioClient):
        super().__init__(client, "ReflectionAgent")
        self.logger.info("[ReflectionAgent] Инициализирован v1.0")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение саморефлексии для коррекции или суммаризации
        """
        task_type = state.get("task_type", "combined")
        reflection_type = state.get("reflection_type", "correction")  # correction или summary

        if reflection_type == "correction":
            return self._reflect_correction(state)
        else:
            return self._reflect_summary(state)

    def _reflect_correction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        original = state.get("input_text", "")
        corrected = state.get("corrected_text", "")
        reference = state.get("reference_text", "")

        if not original or not corrected:
            return self._empty_reflection()

        prompt = self.REFLECTION_PROMPT_CORRECTION.format(
            original=original[:1500],
            corrected=corrected[:1500],
            reference=reference[:1500] if reference else "Не предоставлен"
        )

        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=0.2,  # Низкая температура для стабильности
                system_prompt="Ты строгий эксперт. Анализируй объективно.",
                max_tokens=500
            )
            score, suggestion = self._parse_reflection(response)
            self.logger.info(f"[Reflection] Коррекция: оценка={score}, предложение={suggestion[:100]}...")
        except Exception as e:
            self.logger.error(f"[Reflection] Ошибка: {e}")
            score, suggestion = 5, "Ошибка анализа, повторить с базовым промптом"

        return {
            "reflection_score": score,
            "reflection_suggestion": suggestion,
            "reflection_type": "correction"
        }

    def _reflect_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        original = state.get("corrected_text", "") or state.get("input_text", "")
        summary = state.get("summary_text", "")
        reference = state.get("reference_summary", "")

        if not original or not summary:
            return self._empty_reflection()

        prompt = self.REFLECTION_PROMPT_SUMMARY.format(
            original=original[:1500],
            summary=summary[:500],
            reference=reference[:500] if reference else "Не предоставлен"
        )

        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=0.2,
                system_prompt="Ты строгий эксперт по суммаризации.",
                max_tokens=500
            )
            score, suggestion = self._parse_reflection(response)
            self.logger.info(f"[Reflection] Суммаризация: оценка={score}, предложение={suggestion[:100]}...")
        except Exception as e:
            self.logger.error(f"[Reflection] Ошибка: {e}")
            score, suggestion = 5, "Ошибка анализа, повторить с базовым промптом"

        return {
            "reflection_score": score,
            "reflection_suggestion": suggestion,
            "reflection_type": "summary"
        }

    def _parse_reflection(self, response: str) -> tuple:
        """
        Извлекает оценку (1-10) и предложение из ответа LLM.
        Возвращает (score, suggestion)
        """
        score = 5  # по умолчанию
        suggestion = "Повторить с базовым промптом"

        if not response:
            return score, suggestion

        lines = response.strip().split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('оценка:') or line_lower.startswith('score:'):
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        score = int(float(parts[1].strip()))
                        score = max(1, min(10, score))
                    except:
                        pass
            elif line_lower.startswith('предложение:') or line_lower.startswith('suggestion:'):
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    suggestion = parts[1].strip()
            elif line_lower.startswith('предложение') and ':' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    suggestion = parts[1].strip()

        # Если не нашли явно, ищем в тексте
        if score == 5:
            import re
            match = re.search(r'\b([1-9]|10)\b', response)
            if match:
                score = int(match.group(1))

        return score, suggestion

    def _empty_reflection(self) -> Dict[str, Any]:
        return {
            "reflection_score": 5,
            "reflection_suggestion": "Недостаточно данных для рефлексии",
            "reflection_type": "none"
        }

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] {message}")