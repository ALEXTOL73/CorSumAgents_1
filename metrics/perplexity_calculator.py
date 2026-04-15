"""
Калькулятор метрики Perplexity для оценки качества коррекции
Версия 4.9.3 - Добавлен Singleton паттерн для экономии памяти
Особенности:
- Singleton: все экземпляры разделяют состояние (легковесный)
- Эвристическая оценка перплексии на основе лингвистических признаков
- Нормализация к диапазону 0-1 (меньше = лучше)
- Интеграция с общим скорингом коррекции
- Полная защита от ошибок
"""
import re
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger("PerplexityCalculator", "perplexity_calculator")


class PerplexityCalculator:
    """
    Эвристический калькулятор перплексии
    Singleton: все экземпляры используют одни и те же настройки

    Перплексия оценивается на основе:
    - Частоты редких слов
    - Длины предложений
    - Грамматических паттернов

    Диапазон: 0.0 (идеально) до 1.0 (плохо)
    """

    # Singleton instance
    _instance = None
    _initialized = False

    # Пороговые значения для интерпретации (классовые атрибуты)
    THRESHOLDS = {
        "excellent": 0.3,  # Отличное качество
        "good": 0.5,  # Хорошее качество
        "acceptable": 0.7,  # Приемлемое
        "poor": 1.0  # Плохое
    }

    # Частотный словарь русских слов
    COMMON_RU_WORDS = {
        "и", "в", "не", "на", "я", "быть", "он", "что", "то", "она", "с", "как",
        "а", "по", "мы", "к", "у", "ты", "из", "но", "за", "вы", "все", "так",
        "это", "который", "мочь", "свой", "человек", "год", "раз", "должен",
        "да", "даже", "если", "когда", "только", "уже", "бы", "же", "еще",
        "при", "о", "для", "от", "до", "без", "под", "над", "перед", "после"
    }

    # Частотный словарь английских слов
    COMMON_EN_WORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or"
    }

    def __new__(cls, language: str = "ru"):
        """Singleton паттерн - все экземпляры разделяют состояние"""
        if cls._instance is None:
            cls._instance = super(PerplexityCalculator, cls).__new__(cls)
        return cls._instance

    def __init__(self, language: str = "ru"):
        """
        Инициализация калькулятора (один раз для всех экземпляров)

        Args:
            language: Язык текста ("ru" или "en")
        """
        if self._initialized:
            # Если уже инициализирован, проверяем что язык совпадает (или игнорируем)
            if self.language != language:
                logger.debug(f"[Perplexity] Игнорируем смену языка с {self.language} на {language} (синглтон)")
            return

        self.language = language
        self.common_words = self.COMMON_RU_WORDS if language == "ru" else self.COMMON_EN_WORDS
        self._initialized = True
        logger.debug(f"[Perplexity] Инициализирован (Singleton) для языка: {language}")

    def _tokenize(self, text: str) -> list:
        """Токенизация текста на слова"""
        if not text:
            return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if w]

    def _calculate_word_rarity_score(self, tokens: list) -> float:
        """
        Оценка редкости слов (чем больше редких слов, тем выше перплексия)

        Returns:
            Значение 0-1 (0 = все слова частые, 1 = все слова редкие)
        """
        if not tokens:
            return 0.5

        rare_count = sum(1 for t in tokens if t not in self.common_words)
        return rare_count / len(tokens)

    def _calculate_sentence_complexity(self, text: str) -> float:
        """
        Оценка сложности предложений

        Returns:
            Значение 0-1 (0 = простые предложения, 1 = очень сложные)
        """
        if not text:
            return 0.5

        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.5

        tokens = self._tokenize(text)
        avg_sent_len = len(tokens) / max(1, len(sentences))

        # Нормализуем: 15-20 слов = норма
        if avg_sent_len <= 15:
            return 0.3
        elif avg_sent_len <= 25:
            return 0.5
        elif avg_sent_len <= 40:
            return 0.7
        else:
            return 1.0

    def _calculate_grammar_score(self, text: str) -> float:
        """
        Простая оценка грамматической корректности

        Returns:
            Значение 0-1 (0 = много ошибок, 1 = грамматически чисто)
        """
        if not text:
            return 0.5

        errors = 0

        # Проверка на повторяющиеся слова
        tokens = self._tokenize(text)
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1] and tokens[i] not in {"и", "а", "the"}:
                errors += 1

        # Проверка на лишние пробелы
        if re.search(r'\s{2,}', text):
            errors += text.count('  ')

        # Проверка на несоответствие кавычек
        if text.count('"') % 2 != 0 or text.count("'") % 2 != 0:
            errors += 1

        max_errors = max(5, len(tokens) // 20)
        return max(0, 1 - errors / max_errors)

    def calculate(self, text: str, reference: str = None) -> Dict[str, Any]:
        """
        Расчёт перплексии текста

        Args:
            text: Текст для оценки
            reference: Эталонный текст (опционально, для сравнения)

        Returns:
            Словарь с метриками перплексии
        """
        if not text:
            return {
                "perplexity": 1.0,
                "perplexity_normalized": 1.0,
                "perplexity_interpretation": "❌ Пустой текст",
                "word_rarity": 1.0,
                "sentence_complexity": 1.0,
                "grammar_score": 0.0
            }

        tokens = self._tokenize(text)

        # Компоненты перплексии
        word_rarity = self._calculate_word_rarity_score(tokens)
        sentence_complexity = self._calculate_sentence_complexity(text)
        grammar_score = self._calculate_grammar_score(text)

        # Эвристическая формула перплексии
        # Чем больше редких слов и сложнее предложения, тем выше перплексия
        perplexity = (
                word_rarity * 0.4 +
                sentence_complexity * 0.3 +
                (1 - grammar_score) * 0.3
        )

        # Если есть эталон, корректируем на схожесть
        if reference:
            from metrics.levenstein_calculator import LevenshteinCalculator
            lev_similarity = LevenshteinCalculator.calculate(reference, text)
            # Чем больше схожесть с эталоном, тем ниже перплексия
            perplexity *= (1 - lev_similarity * 0.3)

        # Ограничиваем диапазон 0-1
        perplexity = max(0.0, min(1.0, perplexity))
        perplexity = round(perplexity, 6)

        # Нормализованная перплексия для скоринга (1 - perplexity)
        perplexity_normalized = round(1.0 - perplexity, 6)

        # Интерпретация
        if perplexity < self.THRESHOLDS["excellent"]:
            interpretation = "🟢 Отличное качество (перплексия < 0.3)"
        elif perplexity < self.THRESHOLDS["good"]:
            interpretation = "🟢 Хорошее качество (перплексия < 0.5)"
        elif perplexity < self.THRESHOLDS["acceptable"]:
            interpretation = "🟡 Приемлемое качество (перплексия < 0.7)"
        else:
            interpretation = "🔴 Низкое качество (перплексия >= 0.7)"

        logger.debug(f"[Perplexity] Результат: {perplexity:.4f} (норм: {perplexity_normalized:.4f})")

        return {
            "perplexity": perplexity,
            "perplexity_normalized": perplexity_normalized,
            "perplexity_interpretation": interpretation,
            "word_rarity": round(word_rarity, 3),
            "sentence_complexity": round(sentence_complexity, 3),
            "grammar_score": round(grammar_score, 3)
        }

    def interpret_perplexity(self, perplexity: float) -> str:
        """Интерпретация значения перплексии"""
        if perplexity < self.THRESHOLDS["excellent"]:
            return "🟢 Отлично"
        elif perplexity < self.THRESHOLDS["good"]:
            return "🟢 Хорошо"
        elif perplexity < self.THRESHOLDS["acceptable"]:
            return "🟡 Приемлемо"
        else:
            return "🔴 Низко"

    def get_info(self) -> Dict[str, Any]:
        """Получение информации о состоянии калькулятора"""
        return {
            "singleton": True,
            "language": self.language,
            "common_words_count": len(self.common_words),
            "thresholds": self.THRESHOLDS
        }