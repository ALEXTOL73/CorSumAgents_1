"""
Детектор ошибок на основе Levenshtein расстояния
Версия 5.2.0 - Приоритет 1: Word-Level Levenshtein Pre-Analysis
Особенности:
- Находит слова на расстоянии 1-2 от словарных
- Предлагает исправления
- Интеграция с промптом коррекции
"""
import re
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from utils.logger import setup_logger
from metrics.levenstein_calculator import LevenshteinCalculator

logger = setup_logger("ErrorDetector")


class ErrorDetector:
    """
    Детектор ошибок на основе Levenshtein расстояния
    Находит подозрительные слова и предлагает исправления
    """

    # Базовый словарь частых русских слов (можно расширить)
    RUSSIAN_DICTIONARY: Set[str] = {
        "и", "в", "не", "на", "я", "быть", "он", "что", "то", "она", "с", "как",
        "а", "по", "мы", "к", "у", "ты", "из", "но", "за", "вы", "все", "так",
        "это", "который", "мочь", "свой", "человек", "год", "раз", "должен",
        "да", "даже", "если", "когда", "только", "уже", "бы", "же", "еще",
        "при", "о", "для", "от", "до", "без", "под", "над", "перед", "после",
        "привет", "как", "дела", "хорошо", "плохо", "спасибо", "пожалуйста",
        "да", "нет", "можно", "нельзя", "нужно", "надо", "хочу", "буду",
        "время", "место", "дело", "жизнь", "день", "ночь", "утро", "вечер",
        "работа", "учеба", "школа", "университет", "дом", "квартира", "город",
        "страна", "мир", "люди", "друг", "семья", "мама", "папа", "брат", "сестра",
        "вода", "еда", "деньги", "книга", "слово", "язык", "текст", "информация",
        "проблема", "вопрос", "ответ", "решение", "результат", "цель", "план",
        "коррекция", "исправление", "ошибка", "правильно", "неправильно"
    }

    def __init__(self, dictionary_path: Optional[str] = None, language: str = "ru"):
        """
        Инициализация детектора ошибок

        Args:
            dictionary_path: Путь к файлу словаря (опционально)
            language: Язык текста ("ru" или "en")
        """
        self.language = language
        self.dictionary = self.RUSSIAN_DICTIONARY.copy()

        # Загрузка дополнительного словаря если указан
        if dictionary_path and Path(dictionary_path).exists():
            self._load_dictionary(dictionary_path)

        self.lev_calc = LevenshteinCalculator()
        logger.info(f"[ErrorDetector] Инициализирован для языка: {language}")
        logger.info(f"[ErrorDetector] Размер словаря: {len(self.dictionary)} слов")

    def _load_dictionary(self, path: str):
        """Загрузка словаря из файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and len(word) > 1:
                        self.dictionary.add(word)
            logger.info(f"[ErrorDetector] Загружен словарь из {path}")
        except Exception as e:
            logger.warning(f"[ErrorDetector] Ошибка загрузки словаря: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста на слова"""
        if not text:
            return []
        # Удаляем пунктуацию, оставляем только слова
        text = re.sub(r'[^\w\sа-яА-ЯёЁ]', ' ', text)
        tokens = [w.lower() for w in text.split() if len(w) > 1]
        return tokens

    def find_suspicious_words(self, text: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        Найти подозрительные слова (на расстоянии 1-2 от словарных)

        Args:
            text: Текст для анализа
            max_distance: Максимальное расстояние Левенштейна

        Returns:
            Список подозрительных слов с предложениями
        """
        if not text:
            return []

        tokens = self._tokenize(text)
        suspicious = []

        for token in tokens:
            # Пропускаем очень короткие слова
            if len(token) < 3:
                continue

            # Проверяем есть ли слово в словаре
            if token in self.dictionary:
                continue

            # Ищем ближайшие слова в словаре
            suggestions = self._get_suggestions(token, max_distance)

            if suggestions:
                suspicious.append({
                    "word": token,
                    "distance": min(s["distance"] for s in suggestions),
                    "suggestions": [s["word"] for s in suggestions[:5]],  # Топ-5 предложений
                    "position": text.lower().find(token)
                })

        logger.debug(f"[ErrorDetector] Найдено {len(suspicious)} подозрительных слов")

        return suspicious

    def _get_suggestions(self, word: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        Получить предложения исправлений для слова

        Args:
            word: Слово с ошибкой
            max_distance: Максимальное расстояние

        Returns:
            Список предложений с расстояниями
        """
        suggestions = []

        for dict_word in self.dictionary:
            # Быстрая фильтрация по длине
            if abs(len(dict_word) - len(word)) > max_distance:
                continue

            distance = self.lev_calc.calculate(word, dict_word)

            if 0 < distance <= max_distance:
                suggestions.append({
                    "word": dict_word,
                    "distance": distance
                })

        # Сортируем по расстоянию
        suggestions.sort(key=lambda x: x["distance"])

        return suggestions[:10]  # Возвращаем топ-10

    def generate_enhanced_prompt(self, text: str, base_prompt: str) -> str:
        """
        Сгенерировать улучшенный промпт с подсветкой ошибок

        Args:
            text: Исходный текст
            base_prompt: Базовый промпт коррекции

        Returns:
            Улучшенный промпт с информацией об ошибках
        """
        suspicious = self.find_suspicious_words(text, max_distance=2)

        if not suspicious:
            return base_prompt

        # Формируем блок с подозрительными словами
        error_info = "\n\n⚠️ ПОДОЗРИТЕЛЬНЫЕ СЛОВА (возможные ошибки):\n"

        for i, item in enumerate(suspicious[:10], 1):  # Максимум 10 слов
            suggestions_str = ", ".join(item["suggestions"][:3])
            error_info += f"  {i}. '{item['word']}' → возможно: {suggestions_str} (расстояние: {item['distance']})\n"

        error_info += "\nИсправь эти слова в первую очередь!\n"

        return base_prompt + error_info

    def get_error_density(self, text: str) -> float:
        """
        Получить плотность ошибок в тексте

        Returns:
            Плотность ошибок (0.0-1.0)
        """
        tokens = self._tokenize(text)

        if not tokens:
            return 0.0

        suspicious = self.find_suspicious_words(text, max_distance=2)

        return len(suspicious) / len(tokens)

    def get_error_severity(self, text: str) -> str:
        """
        Определить серьёзность ошибок

        Returns:
            "low", "medium", "high"
        """
        density = self.get_error_density(text)

        if density < 0.05:
            return "low"
        elif density < 0.15:
            return "medium"
        else:
            return "high"