"""
Пост-процессор для очистки и нормализации текстов, сгенерированных LLM
Версия 1.1 - Добавлены: словарные исправления, нормализация чисел, ужесточение пробелов
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import setup_logger

logger = setup_logger("TextPostprocessor", "text_postprocessor")


class TextPostprocessor:
    """Пост-обработка текстовых выходов LLM"""

    # Шаблоны для удаления префиксов/суффиксов (на разных языках)
    PREFIX_PATTERNS = [
        r'^Вот исправленный текст:\s*',
        r'^Исправленный текст:\s*',
        r'^Исправленный вариант:\s*',
        r'^Результат исправления:\s*',
        r'^Корректировка:\s*',
        r'^Исправление:\s*',
        r'^Исправлено:\s*',
        r'^Вот резюме:\s*',
        r'^Резюме:\s*',
        r'^Краткое изложение:\s*',
        r'^Summary:\s*',
        r'^Here is the corrected text:\s*',
        r'^Corrected text:\s*',
        r'^Correction:\s*',
        r'^Here is the summary:\s*',
        r'^Summary:\s*',
        r'^< | file_separator | >\s*',
        r'^< | end_header_id | >\s*'
    ]

    # Кэш для словаря исправлений
    _fixes_cache: Optional[Dict[str, str]] = None
    _fixes_file_path = Path("data/common_fixes.json")

    @classmethod
    def _load_fixes(cls) -> Dict[str, str]:
        """Загрузка словаря исправлений из JSON-файла (один раз)"""
        if cls._fixes_cache is not None:
            return cls._fixes_cache
        if not cls._fixes_file_path.exists():
            logger.warning(f"Файл словаря исправлений не найден: {cls._fixes_file_path}")
            cls._fixes_cache = {}
            return {}
        try:
            with open(cls._fixes_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cls._fixes_cache = data
                logger.info(f"Загружено {len(cls._fixes_cache)} словарных исправлений")
            else:
                logger.warning("Неверный формат common_fixes.json (ожидался словарь)")
                cls._fixes_cache = {}
        except Exception as e:
            logger.error(f"Ошибка загрузки словаря исправлений: {e}")
            cls._fixes_cache = {}
        return cls._fixes_cache

    @classmethod
    def _apply_dictionary_fixes(cls, text: str) -> str:
        """Применение исправлений из словаря (типичные ошибки)"""
        fixes = cls._load_fixes()
        if not fixes:
            return text
        # Сортируем по длине ключа (от длинных к коротким), чтобы избежать частичных замен
        for wrong, correct in sorted(fixes.items(), key=lambda x: len(x[0]), reverse=True):
            # Ищем как целое слово (границы слова) и заменяем
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def _normalize_numbers(cls, text: str) -> str:
        """
        Приведение чисел к единому формату:
        - "6 3 млн" → "6,3 млн"
        - "1 000 000" → "1000000" или "1 000 000" (оставляем пробелы для читаемости?)
        Здесь: убираем пробелы внутри цифр, заменяем на запятую, если между цифрами и буквой.
        """
        # Замена пробелов между цифрами на запятую, если после цифр идёт буква (млн, тыс и т.п.)
        # Пример: "6 3 млн" → "6,3 млн"
        text = re.sub(r'(\d+)\s+(\d+)\s*(млн|тыс|трлн|млрд|тысяч|миллионов|миллиардов|триллионов|million|thousand|billion)',
                      r'\1,\2 \3', text, flags=re.IGNORECASE)
        # Удаляем пробелы внутри больших чисел (например, "1 000 000" → "1000000")
        # Но оставляем пробелы, если они разделяют разные сущности? Лучше убрать все пробелы внутри цифр.
        text = re.sub(r'(\d+)\s+(?=\d)', r'\1', text)
        # Также заменяем точки в числах на запятые (для десятичных)
        # Но это сложно, пока оставим как есть.
        return text

    @classmethod
    def _remove_extra_spaces_before_punctuation(cls, text: str) -> str:
        """Удаление пробелов перед знаками препинания (уже частично есть, но усилим)"""
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Основная функция пост-обработки: удаление префиксов, словарные исправления,
        нормализация чисел, пробелов и пунктуации, капитализация первого символа.
        """
        if not text or not isinstance(text, str):
            return ""

        original_text = text
        text = text.strip()

        # 1. Удаление известных префиксов
        for pattern in cls.PREFIX_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 2. Словарные исправления (типичные ошибки)
        text = cls._apply_dictionary_fixes(text)

        # 3. Нормализация чисел (пробелы внутри чисел, формат "6 3 млн")
        text = cls._normalize_numbers(text)

        # 4. Удаление повторяющихся фраз (простейший случай: повтор предложения)
        text = cls._remove_duplicate_phrases(text)

        # 5. Нормализация пробелов (общая)
        text = cls._normalize_whitespace(text)

        # 6. Удаление пробелов перед знаками препинания (доп. проход)
        text = cls._remove_extra_spaces_before_punctuation(text)

        # 7. Нормализация пунктуации
        text = cls._normalize_punctuation(text)

        # 8. Капитализация первого символа
        text = cls._capitalize_first_char(text)

        # 9. Удаление пустых скобок, лишних кавычек
        text = cls._clean_extra_punctuation(text)

        return text

    @classmethod
    def _remove_duplicate_phrases(cls, text: str) -> str:
        """Удаление дублирующихся предложений или фраз"""
        if not text:
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return text
        unique_sentences = []
        prev = None
        for sent in sentences:
            if sent != prev:
                unique_sentences.append(sent)
                prev = sent
        return ' '.join(unique_sentences)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        """Нормализация пробелов: удаление лишних пробелов, табуляций, переводов строк"""
        text = re.sub(r'\s+', ' ', text)
        # Добавление пробела после знака препинания, если его нет (кроме точки в конце)
        text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)
        return text.strip()

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        """Нормализация пунктуации: замена многоточий, кавычек, тире"""
        text = re.sub(r'\.{2,}', '...', text)
        text = text.replace('—', '-')
        text = re.sub(r'\.{2,}$', '.', text)
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    @classmethod
    def _capitalize_first_char(cls, text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

    @classmethod
    def _clean_extra_punctuation(cls, text: str) -> str:
        """Удаление лишних скобок, кавычек, если они пустые или лишние"""
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        return text
