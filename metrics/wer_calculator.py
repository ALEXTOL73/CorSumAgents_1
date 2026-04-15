"""
Калькулятор метрики WER (Word Error Rate)
Версия 3.10 - Точно по образцу delta_WER.py

Особенности:
- Использует библиотеку werpy для вычисления WER
- delta_WER = wer_inc - wer_corr (ПОЛОЖИТЕЛЬНОЕ = улучшение)
- Полная защита от None и ошибок
"""
import werpy
from typing import List, Union, Optional
from utils.logger import setup_logger

logger = setup_logger("WERCalculator", "wer_calculator")

class WERCalculator:
    """
    Расчет метрики WER для оценки качества коррекции

    WER (Word Error Rate) = (S + D + I) / N
    где S = замены, D = удаления, I = вставки, N = слов в эталоне

    Диапазон: 0.0 (идеально) до 1.0 (все слова ошибочны)
    """

    @staticmethod
    def _normalize(text: str, normalization: str = "lowercase", remove_punctuation: bool = True) -> str:
        """
        Нормализация текста: удаление пунктуации, приведение к нижнему регистру.
        ТОЧНО ПО ОБРАЗЦУ delta_WER.py

        Args:
            text: Исходный текст
            normalization: Тип нормализации ("lowercase" или "none")
            remove_punctuation: Удалять ли пунктуацию

        Returns:
            Нормализованный текст
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
        Расчет WER между эталонным и гипотетическим текстом

        ТОЧНО ПО ОБРАЗЦУ delta_WER.py

        Args:
            reference: Эталонный текст
            hypothesis: Тестовый текст
            normalize: Применять ли нормализацию

        Returns:
            WER score (0.0 - 1.0, где 0 = идеально)
        """
        # Проверка на None
        if reference is None:
            logger.warning("[WER] Reference is None, возвращаем 1.0")
            return 1.0

        if hypothesis is None:
            logger.warning("[WER] Hypothesis is None, возвращаем 1.0")
            return 1.0

        # Конвертация в строку
        try:
            reference = str(reference).strip()
            hypothesis = str(hypothesis).strip()
        except Exception as e:
            logger.error(f"[WER] Ошибка конвертации в строку: {e}")
            return 1.0

        # Нормализация (по образцу delta_WER.py)
        if normalize:
            reference = WERCalculator._normalize(reference)
            hypothesis = WERCalculator._normalize(hypothesis)

        logger.debug(f"[WER] Расчет: ref={len(reference)} символов, hyp={len(hypothesis)} символов")

        # Проверка на пустые строки
        if len(reference) == 0:
            logger.warning("[WER] Пустой эталонный текст")
            return 1.0

        if len(hypothesis) == 0:
            logger.warning("[WER] Пустой гипотетический текст")
            return 1.0

        # Расчет WER через werpy (ТОЧНО ПО ОБРАЗЦУ delta_WER.py)
        try:
            wer = werpy.wer(reference, hypothesis)
            wer = round(wer, 6)
            logger.debug(f"[WER] Результат: {wer}")
            return wer
        except Exception as e:
            logger.error(f"[WER] Ошибка расчета: {e}")
            return 1.0

    @staticmethod
    def calculate_delta(wer_before: float, wer_after: float) -> float:
        """
        Расчет delta_WER

        ТОЧНО ПО ОБРАЗЦУ:
        delta_WER = wer_inc - wer_corr
        ПОЛОЖИТЕЛЬНОЕ = улучшение

        Args:
            wer_before: WER до коррекции (WER_0)
            wer_after: WER после коррекции (WER)

        Returns:
            Delta WER (положительное = улучшение)
        """
        if wer_before is None:
            logger.warning("[WER] wer_before is None, используем 1.0")
            wer_before = 1.0
        if wer_after is None:
            logger.warning("[WER] wer_after is None, используем 1.0")
            wer_after = 1.0

        try:
            wer_before = float(wer_before)
            wer_after = float(wer_after)
        except (TypeError, ValueError) as e:
            logger.error(f"[WER] Ошибка конвертации delta: {e}")
            return 0.0

        # ТОЧНО ПО ОБРАЗЦУ
        delta_wer = round(wer_before - wer_after, 6)

        logger.info(f"[WER] Delta: {wer_before} -> {wer_after} = {delta_wer}")
        return delta_wer

    @staticmethod
    def compute_all_metrics(reference: str, incorrect: str, corrected: str, normalize: bool = True) -> dict:
        """
        Вычисление всех WER метрик: WER_0, WER, delta_WER

        ТОЧНО ПО ОБРАЗЦУ delta_WER.py

        Args:
            reference: Эталонный текст
            incorrect: Исходный искажённый текст
            corrected: Скорректированный текст
            normalize: Применять ли нормализацию

        Returns:
            Dict с метриками
        """
        wer_0 = WERCalculator.calculate(reference, incorrect, normalize)
        wer = WERCalculator.calculate(reference, corrected, normalize)
        delta_wer = WERCalculator.calculate_delta(wer_0, wer)

        improvement_percent = (delta_wer / wer_0 * 100) if wer_0 > 0 else 0

        return {
            "WER_0": wer_0,
            "WER": wer,
            "delta_WER": delta_wer,
            "improved": delta_wer > 0,
            "improvement_percent": improvement_percent
        }