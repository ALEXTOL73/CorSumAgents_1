#!/usr/bin/env python3
"""
Агент-генератор задач для коэволюции
Версия 1.0 - Создаёт более сложные тексты на основе предыдущих ошибок
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory


class TaskGenerator(BaseAgent):
    def __init__(self, client: LMStudioClient, memory: AgentMemory):
        super().__init__(client, "TaskGenerator")
        self.memory = memory

    def generate_harder_test(self, original_text: str, error_profile: Dict[str, int] = None) -> str:
        """На основе исходного текста и профиля ошибок генерирует более сложный вариант"""
        error_prompt = ""
        if error_profile:
            top_errors = sorted(error_profile.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_errors:
                error_prompt = f"Особенно часто встречаются ошибки: {', '.join([f'{e[0]}' for e in top_errors])}."

        prompt = f"""Усложни следующий текст, добавив орфографические, грамматические и пунктуационные ошибки. Сохрани исходный смысл.
{error_prompt}
Текст:
{original_text[:1000]}

Усложнённый текст (только текст, без комментариев):"""
        response = self.client.generate(
            prompt=prompt,
            temperature=0.7,
            system_prompt="Ты создаёшь сложные учебные тексты. Добавляй ошибки, но не искажай смысл.",
            max_tokens=1500
        )
        return response.strip() if response else original_text

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует новый тест на основе неудачного предыдущего"""
        # Эта функция будет вызываться из оркестратора при необходимости
        return {}