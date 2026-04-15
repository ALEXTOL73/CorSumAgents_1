"""
Препроцессор текстов для обработки edge cases
Версия 3.0 - Проблема 7: Обработка краевых случаев
"""
from typing import Dict, Any, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger("TextPreprocessor")


class TextPreprocessor:
    """
    Предварительная обработка текстов

    Проверяет и обрабатывает:
    - Пустые тексты
    - Слишком длинные тексты
    - Смешанные языки
    - Специальные символы
    - Кодировки
    """

    # Константы
    MAX_TEXT_LENGTH = 10000  # Максимальная длина текста
    MIN_TEXT_LENGTH = 10  # Минимальная длина текста
    MAX_SENTENCE_LENGTH = 500  # Максимальная длина предложения

    def __init__(self):
        self.cyrillic_chars = set('абвгдеёжзиклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        self.latin_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def validate(self, text: str, text_type: str = "input") -> Dict[str, Any]:
        """
        Валидация текста

        Args:
            text: Текст для проверки
            text_type: Тип текста (input, reference, summary)

        Returns:
            Словарь {valid: bool, error: str, warnings: list}
        """
        result = {
            "valid": True,
            "error": None,
            "warnings": [],
            "processed_text": text
        }

        # Проверка на None
        if text is None:
            result["valid"] = False
            result["error"] = f"Текст {text_type} равен None"
            logger.error(f"[Preprocessor] {result['error']}")
            return result

        # Проверка на пустой текст
        if not isinstance(text, str) or len(text.strip()) == 0:
            result["valid"] = False
            result["error"] = f"Текст {text_type} пустой"
            logger.error(f"[Preprocessor] {result['error']}")
            return result

        text = text.strip()

        # Проверка на слишком короткий текст
        if len(text) < self.MIN_TEXT_LENGTH:
            result["warnings"].append(f"Текст {text_type} очень короткий ({len(text)} символов)")
            logger.warning(f"[Preprocessor] {result['warnings'][-1]}")

        # Проверка на слишком длинный текст
        if len(text) > self.MAX_TEXT_LENGTH:
            result["valid"] = False
            result[
                "error"] = f"Текст {text_type} слишком длинный ({len(text)} символов, максимум {self.MAX_TEXT_LENGTH})"
            logger.error(f"[Preprocessor] {result['error']}")
            return result

        # Проверка на смешанные языки
        language_info = self.detect_language(text)
        if language_info["mixed"]:
            result["warnings"].append(f"Обнаружены смешанные языки: {language_info['languages']}")
            logger.warning(f"[Preprocessor] {result['warnings'][-1]}")

        # Проверка на специальные символы
        special_chars = self._check_special_characters(text)
        if special_chars:
            result["warnings"].append(f"Обнаружены специальные символы: {special_chars}")
            logger.warning(f"[Preprocessor] {result['warnings'][-1]}")

        # Проверка на очень длинные предложения
        long_sentences = self._check_long_sentences(text)
        if long_sentences:
            result["warnings"].append(f"Обнаружены длинные предложения ({len(long_sentences)} шт)")
            logger.warning(f"[Preprocessor] {result['warnings'][-1]}")

        result["processed_text"] = text
        result["language"] = language_info["primary"]
        result["language_info"] = language_info

        return result

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Определение языка текста

        Args:
            text: Текст для анализа

        Returns:
            Словарь {primary: str, mixed: bool, languages: list, confidence: float}
        """
        cyrillic_count = sum(1 for c in text if c in self.cyrillic_chars)
        latin_count = sum(1 for c in text if c in self.latin_chars)
        total = cyrillic_count + latin_count

        if total == 0:
            return {"primary": "unknown", "mixed": False, "languages": [], "confidence": 0.0}

        cyrillic_ratio = cyrillic_count / total
        latin_ratio = latin_count / total

        languages = []
        if cyrillic_ratio > 0.1:
            languages.append("ru")
        if latin_ratio > 0.1:
            languages.append("en")

        mixed = len(languages) > 1
        primary = "ru" if cyrillic_ratio > latin_ratio else "en"
        confidence = max(cyrillic_ratio, latin_ratio)

        return {
            "primary": primary,
            "mixed": mixed,
            "languages": languages,
            "confidence": confidence
        }

    def _check_special_characters(self, text: str) -> list:
        """Проверка на специальные символы"""
        special = []
        for char in text:
            if ord(char) > 127 and char not in self.cyrillic_chars:
                if char not in special and char not in '—–…':
                    special.append(char)
        return special[:5]  # Возвращаем первые 5

    def _check_long_sentences(self, text: str) -> list:
        """Проверка на длинные предложения"""
        long_sentences = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > self.MAX_SENTENCE_LENGTH:
                long_sentences.append(i)
        return long_sentences

    def truncate_if_needed(self, text: str, max_length: int = None) -> Tuple[str, bool]:
        """
        Обрезка текста если слишком длинный

        Args:
            text: Исходный текст
            max_length: Максимальная длина

        Returns:
            (обрезанный текст, было ли обрезание)
        """
        if max_length is None:
            max_length = self.MAX_TEXT_LENGTH

        if len(text) <= max_length:
            return text, False

        # Обрезаем по последнему предложению
        truncated = text[:max_length]
        last_period = truncated.rfind('.')

        if last_period > max_length * 0.8:
            truncated = truncated[:last_period + 1]

        truncated += "..."
        logger.warning(f"[Preprocessor] Текст обрезан с {len(text)} до {len(truncated)} символов")

        return truncated, True

    def normalize_whitespace(self, text: str) -> str:
        """Нормализация пробелов"""
        import re
        # Замена множественных пробелов на один
        text = re.sub(r'\s+', ' ', text)
        # Удаление пробелов перед знаками препинания
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Добавление пробела после знаков препинания
        text = re.sub(r'([.,!?;:])(\S)', r'\1 \2', text)
        return text.strip()

    def sanitize_text(self, text: str) -> str:
        """
        Очистка текста от проблемных символов

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        # Замена невидимых символов
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\ufeff', '')  # BOM

        # Замена тире
        text = text.replace('—', '-').replace('–', '-')

        # Замена кавычек
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text