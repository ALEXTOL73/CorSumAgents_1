"""
Метрики качества CorSumAgentsAI
Версия 5.0.3 - Исправлено имя класса: BertScoreCalculator
Импортирует все классы метрик для удобного использования
"""
from .wer_calculator import WERCalculator
from .levenstein_calculator import LevenshteinCalculator
from .meteor_calculator import METEORCalculator
from .geval_calculator import GEvalCalculator, GEvalResult
from .bertscore_calculator import BertScoreCalculator  # ✅ ИСПРАВЛЕНО: BertScoreCalculator

__all__ = [
    "WERCalculator",
    "LevenshteinCalculator",
    "METEORCalculator",
    "GEvalCalculator",
    "GEvalResult",
    "BertScoreCalculator"  # ✅ ИСПРАВЛЕНО: BertScoreCalculator
]