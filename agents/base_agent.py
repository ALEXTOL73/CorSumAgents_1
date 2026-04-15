"""
Базовый класс для всех агентов системы
Версия 4.0 - Стандартная версия
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from utils.logger import setup_logger
from utils.lmstudio_client import LMStudioClient


class BaseAgent(ABC):
    """Базовый класс агента"""

    def __init__(self, client: Optional[LMStudioClient] = None, name: str = "BaseAgent"):
        self.client = client
        self.name = name  # ✅ Атрибут называется self.name (НЕ self.agent_name!)
        self.logger = setup_logger(name, name.lower())
        self.execution_count = 0

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение задачи агента

        Args:
            state: Текущее состояние графа

        Returns:
            Обновленное состояние
        """
        pass

    def log_execution(self, message: str):
        """Логирование выполнения"""
        self.execution_count += 1
        self.logger.info(f"[{self.name}] #{self.execution_count}: {message}")

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики агента"""
        return {
            "name": self.name,
            "execution_count": self.execution_count
        }