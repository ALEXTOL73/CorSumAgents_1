"""
Калькулятор метрики METEOR для оценки качества суммаризации
Версия 3.27 - Добавлен Singleton паттерн для экономии ресурсов

Особенности:
- Singleton: NLTK ресурсы загружаются только один раз
- Простая токенизация без NLTK (работает всегда)
- Обработка SSL ошибок при загрузке NLTK
- Fallback на Jaccard similarity если METEOR недоступен
- Диапазон: 0.0 (плохо) до 1.0 (идеально)
- Полная защита от ошибок
"""

from typing import Dict, Any
import re
import os
import sys
from utils.logger import setup_logger

logger = setup_logger("METEORCalculator", "meteor_calculator")


# =============================================================================
# ПРОСТАЯ ТОКЕНИЗАЦИЯ (БЕЗ NLTK ЗАВИСИМОСТИ)
# =============================================================================

def simple_tokenize(text: str) -> list:
    """
    Простая токенизация текста без зависимости от NLTK

    Args:
        text: Исходный текст

    Returns:
        Список токенов (слов)
    """
    if not text or not isinstance(text, str):
        return []

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем знаки препинания и разделяем по пробелам
    tokens = re.findall(r'\b\w+\b', text)

    # Фильтруем короткие токены (меньше 2 символов)
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens


# =============================================================================
# ПОПЫТКА ИМПОРТА NLTK (с обработкой SSL ошибок) - глобальная для синглтона
# =============================================================================

_NLTK_AVAILABLE = False
_meteor_score_func = None

try:
    import nltk

    # Попытка загрузить ресурсы с обработкой SSL ошибок
    try:
        # Проверяем доступны ли ресурсы
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/wordnet')
        _NLTK_AVAILABLE = True
        logger.info("[METEOR] NLTK доступен с полными ресурсами")
    except LookupError:
        logger.warning("[METEOR] NLTK ресурсы не найдены, пробуем загрузить...")

        # Пробуем загрузить с отключенной SSL проверкой
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

            resources = ['punkt_tab', 'punkt', 'wordnet']
            download_dir = os.path.expanduser("~/nltk_data")

            for resource in resources:
                try:
                    nltk.download(resource, download_dir=download_dir, quiet=True, halt_on_error=False)
                except Exception as e:
                    logger.debug(f"[METEOR] Не удалось загрузить {resource}: {e}")

            # Проверяем что загрузилось
            try:
                nltk.data.find('tokenizers/punkt_tab')
                nltk.data.find('corpora/wordnet')
                _NLTK_AVAILABLE = True
                logger.info("[METEOR] NLTK ресурсы успешно загружены")
            except:
                logger.warning("[METEOR] Не удалось загрузить NLTK ресурсы, используем fallback")
                _NLTK_AVAILABLE = False
        except Exception as e:
            logger.warning(f"[METEOR] Ошибка загрузки NLTK: {e}")
            _NLTK_AVAILABLE = False

    # Импортируем METEOR только если ресурсы доступны
    if _NLTK_AVAILABLE:
        try:
            from nltk.translate.meteor_score import meteor_score as meteor_score_func
            _meteor_score_func = meteor_score_func
        except Exception as e:
            logger.error(f"[METEOR] Ошибка импорта meteor_score: {e}")
            _NLTK_AVAILABLE = False
            _meteor_score_func = None

except ImportError:
    logger.warning("[METEOR] NLTK не установлен, используем упрощённую метрику")
    _NLTK_AVAILABLE = False
    _meteor_score_func = None
except Exception as e:
    logger.error(f"[METEOR] Ошибка инициализации NLTK: {e}")
    _NLTK_AVAILABLE = False
    _meteor_score_func = None


# =============================================================================
# КЛАСС METEORCalculator (СИНГЛТОН)
# =============================================================================

class METEORCalculator:
    """
    Расчет метрики METEOR для оценки суммаризации
    Singleton: все экземпляры используют одни и те же NLTK ресурсы

    Если NLTK доступен: использует настоящий METEOR
    Если NLTK недоступен: использует Jaccard similarity (fallback)

    Диапазон: 0.0 (плохо) до 1.0 (идеально)
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton паттерн - все экземпляры разделяют состояние"""
        if cls._instance is None:
            cls._instance = super(METEORCalculator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация калькулятора (один раз для всех экземпляров)"""
        if self._initialized:
            return
        self._initialized = True
        self.nltk_available = _NLTK_AVAILABLE
        self.meteor_score_func = _meteor_score_func
        logger.debug(f"[METEOR] Калькулятор инициализирован (Singleton, NLTK: {self.nltk_available})")

    def _tokenize_text(self, text: str) -> list:
        """
        Токенизация текста на слова

        Args:
            text: Исходный текст

        Returns:
            Список токенов (слов)
        """
        return simple_tokenize(text)

    def calculate(self, reference: str, hypothesis: str) -> float:
        """
        Вычисляет METEOR между эталонным и гипотетическим текстом

        Args:
            reference: Эталонный текст
            hypothesis: Гипотетический текст (суммаризация)

        Returns:
            METEOR score (0.0 - 1.0, где 1.0 = идеально)
        """
        # Проверка на None
        if reference is None:
            logger.warning("[METEOR] Reference is None, возвращаем 0.0")
            return 0.0

        if hypothesis is None:
            logger.warning("[METEOR] Hypothesis is None, возвращаем 0.0")
            return 0.0

        # Конвертация в строку
        try:
            reference = str(reference).strip()
            hypothesis = str(hypothesis).strip()
        except Exception as e:
            logger.error(f"[METEOR] Ошибка конвертации в строку: {e}")
            return 0.0

        logger.debug(f"[METEOR] Расчет: ref={len(reference)} символов, hyp={len(hypothesis)} символов")

        # Проверка на пустые строки
        if len(reference) == 0:
            logger.warning("[METEOR] Пустой эталонный текст")
            return 0.0

        if len(hypothesis) == 0:
            logger.warning("[METEOR] Пустой гипотетический текст")
            return 0.0

        # Токенизация (простая, без NLTK)
        ref_tokens = self._tokenize_text(reference)
        hyp_tokens = self._tokenize_text(hypothesis)

        logger.debug(f"[METEOR] Токенизация: ref={len(ref_tokens)} токенов, hyp={len(hyp_tokens)} токенов")

        # Проверка после токенизации
        if len(ref_tokens) == 0:
            logger.warning("[METEOR] Пустой эталон (после токенизации)")
            return 0.0

        if len(hyp_tokens) == 0:
            logger.warning("[METEOR] Пустая гипотеза (после токенизации)")
            return 0.0

        # Расчет METEOR или fallback
        if self.nltk_available and self.meteor_score_func:
            try:
                # meteor_score принимает список эталонов и гипотезу
                meteor = self.meteor_score_func([ref_tokens], hyp_tokens)
                meteor = round(meteor, 6)
                logger.debug(f"[METEOR] Результат (NLTK): {meteor}")
                return meteor
            except Exception as e:
                logger.warning(f"[METEOR] Ошибка NLTK METEOR: {e}, используем fallback")

        # Fallback: Jaccard similarity
        return self._jaccard_similarity(ref_tokens, hyp_tokens)

    def _jaccard_similarity(self, ref_tokens: list, hyp_tokens: list) -> float:
        """
        Jaccard similarity как fallback если NLTK недоступен

        Args:
            ref_tokens: Токены эталона
            hyp_tokens: Токены гипотезы

        Returns:
            Коэффициент схожести (0.0 - 1.0)
        """
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)

        if len(ref_set) == 0 and len(hyp_set) == 0:
            return 0.0

        intersection = len(ref_set & hyp_set)
        union = len(ref_set | hyp_set)

        if union == 0:
            return 0.0

        similarity = intersection / union
        return round(similarity, 6)

    def compute_all_metrics(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Вычисление всех METEOR метрик

        Args:
            reference: Эталонный текст
            hypothesis: Гипотетический текст (суммаризация)

        Returns:
            Dict с метриками
        """
        meteor_score_value = self.calculate(reference, hypothesis)

        return {
            "meteor": meteor_score_value,
            "meteor_interpretation": self._interpret_score(meteor_score_value),
            "nltk_used": self.nltk_available
        }

    def _interpret_score(self, score: float) -> str:
        """
        Интерпретация METEOR score

        Args:
            score: METEOR score (0.0 - 1.0)

        Returns:
            Строка с интерпретацией
        """
        if score >= 0.5:
            return "🟢 Отличное качество (METEOR >= 0.5)"
        elif score >= 0.4:
            return "🟢 Хорошее качество (METEOR >= 0.4)"
        elif score >= 0.3:
            return "🟡 Удовлетворительное качество (METEOR >= 0.3)"
        elif score >= 0.2:
            return "🟠 Низкое качество (METEOR >= 0.2)"
        else:
            return "🔴 Критически низкое качество (METEOR < 0.2)"

    def get_info(self) -> Dict[str, Any]:
        """Получение информации о состоянии калькулятора"""
        return {
            "singleton": True,
            "nltk_available": self.nltk_available,
            "meteor_function_loaded": self.meteor_score_func is not None
        }