#!/usr/bin/env python3
"""
Агент-оптимизатор гиперпараметров
Версия 1.0 - Анализирует историю и предлагает оптимальные параметры
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory


class HyperparameterOptimizer(BaseAgent):
    def __init__(self, client: LMStudioClient, memory: AgentMemory):
        super().__init__(client, "HyperparameterOptimizer")
        self.memory = memory

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Запускает оптимизацию на основе истории успешных тестов"""
        history = self.memory.get_correction_history(limit=50)
        if len(history) < 10:
            return {}

        # Анализ температур для коррекции
        temps_corr = [float(h.get("best_temperature", 0.5)) for h in history if h.get("best_temperature")]
        scores_corr = [h.get("metrics", {}).get("CorScore", 0) for h in history]
        best_temp_corr = self._best_parameter(temps_corr, scores_corr)

        # Анализ температур для суммаризации (если есть)
        sum_history = self.memory.get_summary_history(limit=50)
        temps_sum = [float(h.get("best_temperature_summary", 0.5)) for h in sum_history if h.get("best_temperature_summary")]
        scores_sum = [h.get("metrics", {}).get("SumScore", 0) for h in sum_history]
        best_temp_sum = self._best_parameter(temps_sum, scores_sum) if temps_sum else 0.5

        # Предложения по изменению весов в формуле (опционально)
        # Здесь можно добавить более сложную логику

        print("\n" + "="*80)
        print("  📊 ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ")
        print("="*80)
        print(f"  Рекомендуемая температура коррекции: {best_temp_corr:.2f}")
        print(f"  Рекомендуемая температура суммаризации: {best_temp_sum:.2f}")
        print("="*80 + "\n")

        # Сохраняем результат в память
        self.memory.save_optimization_result({
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "best_temp_correction": best_temp_corr,
            "best_temp_summary": best_temp_sum
        })

        # Можно также обновить глобальные настройки через конфиг (не рекомендуется)
        # Возвращаем предложения для использования в оркестраторе
        return {
            "optimized_temperature_correction": best_temp_corr,
            "optimized_temperature_summary": best_temp_sum
        }

    def _best_parameter(self, params: List[float], scores: List[float]) -> float:
        """Находит параметр (например, температуру), который даёт максимальный средний score"""
        if not params:
            return 0.5
        # Группируем по параметру
        param_score_map = {}
        for p, s in zip(params, scores):
            if p not in param_score_map:
                param_score_map[p] = []
            param_score_map[p].append(s)
        # Вычисляем средний score для каждого параметра
        avg_scores = {p: sum(ss)/len(ss) for p, ss in param_score_map.items()}
        # Выбираем параметр с максимальным средним score
        best_param = max(avg_scores.items(), key=lambda x: x[1])[0]
        return best_param