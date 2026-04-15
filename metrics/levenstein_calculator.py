"""
Калькулятор расстояния Левенштейна и метрики схожести
Версия 3.10 - Точно по образцу: Levenshtein.ratio()

Особенности:
- Использует Levenshtein.ratio() для расчета схожести (0-1, БОЛЬШЕ = ЛУЧШЕ)
- LevRating = схожесть (НЕ расстояние!)
- delta_LEV = lev_corr - lev_inc (ПОЛОЖИТЕЛЬНОЕ = улучшение)
- Полная защита от None и ошибок
"""
import Levenshtein
from utils.logger import setup_logger

logger = setup_logger("LevenshteinCalculator", "levenstein_calculator")

class LevenshteinCalculator:
    """
    Расчет метрики Левенштейна через Levenshtein.ratio()

    ТОЧНО ПО ОБРАЗЦУ:
    - Levenshtein.ratio() возвращает схожесть (0-1)
    - БОЛЬШЕ = ЛУЧШЕ (0.985 > 0.955 = улучшение)
    - delta_LEV = lev_corr - lev_inc (ПОЛОЖИТЕЛЬНОЕ = улучшение)
    """

    @staticmethod
    def _normalize(text: str, normalization: str = "lowercase", remove_punctuation: bool = True) -> str:
        """
        Нормализация текста: удаление пунктуации, приведение к нижнему регистру.
        ТОЧНО ПО ОБРАЗЦУ delta_WER.py
        """
        import re

        if not text:
            return ""

        result = text

        if remove_punctuation:
            result = re.sub(r'[^\w\s]', ' ', result)

        if normalization == "lowercase":
            result = result.lower()

        return ' '.join(result.split())

    @staticmethod
    def calculate(reference: str, hypothesis: str, normalize: bool = True) -> float:
        """
        Вычисляет схожесть Левенштейна через Levenshtein.ratio().

        ТОЧНО ПО ОБРАЗЦУ:
        LevRating = Levenshtein.ratio() (ЧЕМ БОЛЬШЕ, тем ЛУЧШЕ)

        Args:
            reference: Эталонный текст
            hypothesis: Гипотетический текст
            normalize: Применять ли нормализацию

        Returns:
            Схожесть Левенштейна (0-1, БОЛЬШЕ = ЛУЧШЕ)
        """
        # Проверка на None
        if reference is None:
            logger.warning("[Lev] Reference is None, возвращаем 0.0")
            return 0.0

        if hypothesis is None:
            logger.warning("[Lev] Hypothesis is None, возвращаем 0.0")
            return 0.0

        # Конвертация в строку
        try:
            reference = str(reference).strip()
            hypothesis = str(hypothesis).strip()
        except Exception as e:
            logger.error(f"[Lev] Ошибка конвертации в строку: {e}")
            return 0.0

        # Нормализация (по образцу delta_WER.py)
        if normalize:
            reference = LevenshteinCalculator._normalize(reference)
            hypothesis = LevenshteinCalculator._normalize(hypothesis)

        logger.debug(f"[Lev] Расчет: ref={len(reference)} символов, hyp={len(hypothesis)} символов")

        try:
            # ТОЧНО ПО ОБРАЗЦУ: Levenshtein.ratio() возвращает схожесть
            # ЧЕМ БОЛЬШЕ, тем ЛУЧШЕ (0.985 > 0.955 = улучшение)
            lev_rating = round(Levenshtein.ratio(reference, hypothesis), 6)

            logger.debug(f"[Lev] Levenshtein.ratio: {lev_rating}")
            return lev_rating

        except Exception as e:
            logger.error(f"[Lev] Ошибка расчета: {e}")
            return 0.0

    @staticmethod
    def calculate_distance(reference: str, hypothesis: str, normalize: bool = False) -> int:
        """
        Вычисляет расстояние Левенштейна (количество редакций).

        Args:
            reference: Эталонный текст
            hypothesis: Гипотетический текст
            normalize: Применять ли нормализацию

        Returns:
            Расстояние Левенштейна (целое число)
        """
        if reference is None or hypothesis is None:
            logger.warning("[Lev] None значение, возвращаем -1")
            return -1

        try:
            reference = str(reference).strip()
            hypothesis = str(hypothesis).strip()

            if normalize:
                reference = LevenshteinCalculator._normalize(reference)
                hypothesis = LevenshteinCalculator._normalize(hypothesis)

            dist = Levenshtein.distance(reference, hypothesis)
            logger.debug(f"[Lev-Dist] Расстояние: {dist}")
            return dist

        except Exception as e:
            logger.error(f"[Lev-Dist] Ошибка расчета: {e}")
            return -1

    @staticmethod
    def calculate_delta(lev_before: float, lev_after: float) -> float:
        """
        Расчет delta_LEV

        ТОЧНО ПО ОБРАЗЦУ:
        delta_LEV = lev_corr - lev_inc
        ПОЛОЖИТЕЛЬНОЕ = улучшение (схожесть увеличилась)

        Args:
            lev_before: LevRating до коррекции (LevRating_0)
            lev_after: LevRating после коррекции (LevRating)

        Returns:
            Delta LevRating (положительное = улучшение)
        """
        if lev_before is None:
            logger.warning("[Lev] lev_before is None, используем 0.0")
            lev_before = 0.0
        if lev_after is None:
            logger.warning("[Lev] lev_after is None, используем 0.0")
            lev_after = 0.0

        try:
            lev_before = float(lev_before)
            lev_after = float(lev_after)
        except (TypeError, ValueError) as e:
            logger.error(f"[Lev] Ошибка конвертации delta: {e}")
            return 0.0

        # ТОЧНО ПО ОБРАЗЦУ: lev_corr - lev_inc
        # ПОЛОЖИТЕЛЬНОЕ = улучшение (схожесть увеличилась)
        delta_lev = round(lev_after - lev_before, 6)

        logger.info(f"[Lev] Delta: {lev_before} -> {lev_after} = {delta_lev}")
        return delta_lev

    @staticmethod
    def compute_all_metrics(reference: str, incorrect: str, corrected: str, normalize: bool = True) -> dict:
        """
        Вычисление всех Lev метрик: LevRating_0, LevRating, delta_LEV

        ТОЧНО ПО ОБРАЗЦУ delta_WER.py

        Args:
            reference: Эталонный текст
            incorrect: Исходный искажённый текст
            corrected: Скорректированный текст
            normalize: Применять ли нормализацию

        Returns:
            Dict с метриками
        """
        lev_0 = LevenshteinCalculator.calculate(reference, incorrect, normalize)
        lev = LevenshteinCalculator.calculate(reference, corrected, normalize)
        delta_lev = LevenshteinCalculator.calculate_delta(lev_0, lev)

        return {
            "LevRating_0": lev_0,
            "LevRating": lev,
            "delta_LEV": delta_lev,
            "improved": delta_lev > 0  # Положительное = улучшение
        }
