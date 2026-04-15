"""
Пост-процессор для очистки и нормализации текстов, сгенерированных LLM
Версия 1.0 - Удаление лишних комментариев, нормализация пробелов и пунктуации, капитализация
"""
import re
from typing import List, Optional


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
        r'^< | file_separator | >\s*'
    ]

    SUFFIX_PATTERNS = [
        r'\s*$',
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Основная функция пост-обработки: удаление префиксов, нормализация пробелов и пунктуации,
        капитализация первого символа.

        Args:
            text: Исходный текст от LLM

        Returns:
            Очищенный и нормализованный текст
        """
        if not text or not isinstance(text, str):
            return ""

        original_text = text
        text = text.strip()

        # 1. Удаление известных префиксов
        for pattern in cls.PREFIX_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 2. Удаление повторяющихся фраз (простейший случай: повтор предложения)
        text = cls._remove_duplicate_phrases(text)

        # 3. Нормализация пробелов
        text = cls._normalize_whitespace(text)

        # 4. Нормализация пунктуации
        text = cls._normalize_punctuation(text)

        # 5. Капитализация первого символа
        text = cls._capitalize_first_char(text)

        # 6. Удаление пустых скобок, лишних кавычек (опционально)
        text = cls._clean_extra_punctuation(text)

        return text

    @classmethod
    def _remove_duplicate_phrases(cls, text: str) -> str:
        """Удаление дублирующихся предложений или фраз"""
        if not text:
            return text

        # Разделение на предложения (по точкам, восклицательным, вопросительным знакам)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return text

        # Удаляем последовательные дубликаты
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
        # Замена всех пробельных символов на пробел
        text = re.sub(r'\s+', ' ', text)
        # Удаление пробелов перед знаками препинания
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Добавление пробела после знака препинания, если его нет (кроме точки в конце)
        text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)
        return text.strip()

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        """Нормализация пунктуации: замена многоточий, кавычек, тире"""
        # Замена многоточий на стандартное ...
        text = re.sub(r'\.{2,}', '...', text)
        # Замена длинных тире на короткие (или наоборот, по желанию)
        text = text.replace('—', '-')
        # Удаление лишних точек в конце (если больше одной)
        text = re.sub(r'\.{2,}$', '.', text)
        # Добавление точки в конце, если нет знака препинания
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    @classmethod
    def _capitalize_first_char(cls, text: str) -> str:
        """Приведение первого символа текста к заглавному"""
        if not text:
            return text
        return text[0].upper() + text[1:]

    @classmethod
    def _clean_extra_punctuation(cls, text: str) -> str:
        """Удаление лишних скобок, кавычек, если они пустые или лишние"""
        # Удаление пустых скобок
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        # Удаление лишних кавычек в начале и конце (если текст полностью в кавычках)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        return text