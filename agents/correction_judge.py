"""
Агент-судья для оценки качества коррекции
Версия 3.13 - Добавлен детальный вывод метрик в консоль
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from metrics.wer_calculator import WERCalculator
from metrics.levenstein_calculator import LevenshteinCalculator


class CorrectionJudge(BaseAgent):
    """Оценка качества коррекции через метрики"""

    def __init__(self):
        """Инициализация судьи"""
        super().__init__(None, "CorrectionJudge")
        self.wer_calc = WERCalculator()
        self.lev_calc = LevenshteinCalculator()

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка коррекции"""
        self.log_execution("=" * 60)
        self.log_execution("НАЧАЛО ОЦЕНКИ КОРРЕКЦИИ")
        self.log_execution("=" * 60)

        reference = state.get("reference_text", "") or ""
        corrected = state.get("corrected_text", "") or ""
        original = state.get("input_text", "") or ""

        if not reference:
            reference = "Пустой эталон"
        if not corrected:
            corrected = "Пустой результат"
        if not original:
            original = "Пустой вход"

        # ========== МЕТРИКИ ==========
        wer_metrics = WERCalculator.compute_all_metrics(reference, original, corrected)
        lev_metrics = LevenshteinCalculator.compute_all_metrics(reference, original, corrected)

        wer_before = wer_metrics.get("WER_0", 1.0)
        wer_after = wer_metrics.get("WER", 1.0)
        delta_wer = wer_metrics.get("delta_WER", 0.0)

        lev_rating_0 = lev_metrics.get("LevRating_0", 0.0)
        lev_rating = lev_metrics.get("LevRating", 0.0)
        delta_lev = lev_metrics.get("delta_LEV", 0.0)

        # ========== ДЕТАЛЬНЫЙ ВЫВОД В КОНСОЛЬ ==========
        print("\n" + "=" * 80)
        print("  📊 СУДЬЯ КОРРЕКЦИИ - МЕТРИКИ")
        print("=" * 80)
        print(f"  📄 Эталонный текст: {len(reference)} символов")
        print(f"  📄 Исходный текст: {len(original)} символов")
        print(f"  📄 Скорректированный текст: {len(corrected)} символов")
        print()
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │                      ПОКАЗАТЕЛИ КАЧЕСТВА                     │")
        print("  ├─────────────────────────────────────────────────────────────┤")
        print(f"  │  WER (до)     : {wer_before:.6f}                              │")
        print(f"  │  WER (после)  : {wer_after:.6f}                              │")
        print(f"  │  ΔWER         : {delta_wer:+.6f}                              │")
        print(f"  │  LevRating (до)   : {lev_rating_0:.6f}                        │")
        print(f"  │  LevRating (после): {lev_rating:.6f}                          │")
        print(f"  │  ΔLevRating   : {delta_lev:+.6f}                              │")
        print("  └─────────────────────────────────────────────────────────────┘")
        print()

        # ========== УМНАЯ ЛОГИКА RETRY ==========
        needs_retry = self._should_retry_smart(
            wer_before=wer_before,
            wer_after=wer_after,
            delta_wer=delta_wer,
            delta_lev=delta_lev,
            lev_rating=lev_rating
        )

        quality_assessment = self._assess_quality(delta_wer, delta_lev, wer_after)

        print(f"  🏆 Оценка качества: {quality_assessment}")
        print(f"  🔄 Требуется повтор: {'✅ ДА' if needs_retry else '❌ НЕТ'}")
        print("=" * 80 + "\n")

        # Логирование в файл
        self.logger.info(f"\n📊 Эталонный текст: {len(reference)} символов")
        self.logger.info(f"📊 Исходный текст: {len(original)} символов")
        self.logger.info(f"📊 Скорректированный текст: {len(corrected)} символов")
        self.logger.info(f"\n--- МЕТРИКИ ---")
        self.logger.info(f"  WER: {wer_before:.6f} → {wer_after:.6f} (Δ={delta_wer:.6f})")
        self.logger.info(f"  LevRating: {lev_rating_0:.6f} → {lev_rating:.6f} (Δ={delta_lev:.6f})")
        self.logger.info(f"  SUM (delta_WER + delta_Lev): {delta_wer + delta_lev:.6f}")
        self.logger.info(f"\n--- ИНТЕРПРЕТАЦИЯ ---")
        self.logger.info(f"  {quality_assessment}")
        self.logger.info(f"  🔄 Требуется повтор: {needs_retry}")

        metrics_data = {
            "WER_0": wer_before,
            "WER": wer_after,
            "delta_WER": delta_wer,
            "LevRating_0": lev_rating_0,
            "LevRating": lev_rating,
            "delta_LEV": delta_lev,
            "quality_assessment": quality_assessment
        }

        self.log_execution("=" * 60)
        self.log_execution("ОЦЕНКА КОРРЕКЦИИ ЗАВЕРШЕНА")
        self.log_execution("=" * 60)

        return {
            "metrics_correction": metrics_data,
            "needs_retry_correction": needs_retry
        }

    def _should_retry_smart(self, wer_before: float, wer_after: float, delta_wer: float,
                            delta_lev: float, lev_rating: float) -> bool:
        """Умная логика RETRY"""
        total_improvement = delta_wer + delta_lev

        if delta_wer > 0:
            self.logger.info(f"  ✅ ΔWER > 0 ({delta_wer:.6f})")
            return False
        if delta_lev > 0:
            self.logger.info(f"  ✅ ΔLevRating > 0 ({delta_lev:.6f})")
            return False
        if total_improvement > 0:
            self.logger.info(f"  ✅ SUM > 0 ({total_improvement:.6f})")
            return False
        if wer_after < 0.1:
            self.logger.info(f"  ✅ WER после < 0.1 ({wer_after:.6f})")
            return False
        if lev_rating > 0.9:
            self.logger.info(f"  ✅ LevRating после > 0.9 ({lev_rating:.6f})")
            return False

        self.logger.info(f"  ❌ Нет достаточных улучшений, требуется повтор")
        return True

    def _assess_quality(self, delta_wer: float, delta_lev: float, wer_after: float) -> str:
        """Оценка качества коррекции"""
        total_improvement = delta_wer + delta_lev

        if wer_after == 0.0:
            return "⭐⭐⭐⭐⭐ ОТЛИЧНО (WER = 0)"
        elif wer_after < 0.05:
            return "⭐⭐⭐⭐⭐ ОТЛИЧНО (WER < 0.05)"
        elif total_improvement > 0.3:
            return "⭐⭐⭐⭐⭐ ОТЛИЧНО (значительное улучшение)"
        elif total_improvement > 0.15:
            return "⭐⭐⭐⭐ ХОРОШО (заметное улучшение)"
        elif total_improvement > 0:
            return "⭐⭐⭐ УДОВЛЕТВОРИТЕЛЬНО (небольшое улучшение)"
        else:
            return "⭐⭐ ТРЕБУЕТСЯ УЛУЧШЕНИЕ"

    def log_execution(self, message: str):
        self.logger.info(f"[{self.name}] {message}")