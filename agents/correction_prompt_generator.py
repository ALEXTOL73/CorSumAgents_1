"""
Агент генерации промптов для коррекции текста
Версия 5.5.0 - Динамический few-shot из памяти
"""
import random
from typing import Dict, Any, List, Tuple, Optional
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory
from config import DYNAMIC_FEW_SHOT_ENABLED, MAX_FEW_SHOT_EXAMPLES, FEW_SHOT_SIMILARITY_THRESHOLD


class CorrectionPromptGenerator(BaseAgent):
    SYSTEM_PROMPT_VARIANTS = {
        "ru": [
            "Ты - эксперт по созданию промптов для профессиональных редакторов текстов. Будь строгим и детальным.",
            "Ты - креативный копирайтер, помогающий писать эффективные инструкции для ИИ. Используй живые формулировки.",
            "Ты - технический писатель, который ценит точность и краткость. Избегай лишних слов.",
            "Ты - исследователь в области NLP. Твои промпты должны быть основаны на лучших практиках few-shot и chain-of-thought.",
            "Ты - преподаватель русского языка. Объясняй требования доступно, но чётко."
        ],
        "en": [
            "You are an expert at creating prompts for professional text editors. Be strict and detailed.",
            "You are a creative copywriter helping to write effective instructions for AI. Use vivid wording.",
            "You are a technical writer who values precision and brevity. Avoid unnecessary words.",
            "You are an NLP researcher. Your prompts should be based on best practices of few-shot and chain-of-thought.",
            "You are a language teacher. Explain requirements clearly but accessibly."
        ]
    }

    FEW_SHOT_SETS = {
        "ru": [
            """ПРИМЕРЫ:
Вход: "првиет как дила?" → Выход: "привет как дела?"
Вход: "сегодн я иду в шклу" → Выход: "сегодня я иду в школу"
Вход: "он не пришол потаму что был бален" → Выход: "он не пришёл потому что был болен" """,
            """ПРИМЕРЫ С РАЗБОРОМ:
Ошибка: "првиет" (пропущена буква 'в') → исправление: "привет"
Ошибка: "дила" (неправильное окончание) → исправление: "дела"
Ошибка: "шклу" (неправильный падеж) → исправление: "школу"
Ошибка: "пришол" → исправление: "пришёл"
Ошибка: "потаму" → исправление: "потому"
Ошибка: "бален" → исправление: "болен" """,
            """ПРИМЕРЫ РАЗНЫХ ТИПОВ ОШИБОК:
1. Орфография: "првиет" → "привет"
2. Грамматика: "он пошли" → "он пошёл"
3. Пунктуация: "привет как дела" → "привет, как дела?"
4. Стиль: "очень очень очень плохо" → "очень плохо" """
        ],
        "en": [
            """EXAMPLES:
Input: "helo how are you" → Output: "hello how are you"
Input: "i go to scool" → Output: "i go to school"
Input: "he didnt came" → Output: "he didn't come" """,
            """EXAMPLES WITH EXPLANATION:
Error: "helo" (missing letter) → correction: "hello"
Error: "scool" (wrong spelling) → correction: "school"
Error: "didnt came" (grammar) → correction: "didn't come" """,
            """ERROR TYPES:
1. Spelling: "recieve" → "receive"
2. Grammar: "She go" → "She goes"
3. Punctuation: "Hello how are you" → "Hello, how are you?" """
        ]
    }

    def __init__(self, client: LMStudioClient, num_variants: int = 3, memory: Optional[AgentMemory] = None):
        super().__init__(client, "CorrectionPromptGenerator")
        self.num_variants = num_variants
        self.memory = memory

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Генерация промптов для коррекции")
        input_text = state.get("input_text", "")
        has_cyrillic = any(ord(c) > 127 for c in str(input_text))
        language = "русском" if has_cyrillic else "английском"
        language_code = "ru" if has_cyrillic else "en"
        domain = state.get("domain", "general")

        # Динамические примеры из памяти
        examples_text = ""
        if DYNAMIC_FEW_SHOT_ENABLED and self.memory:
            examples = self.memory.get_few_shot_examples(
                input_text=input_text,
                domain=domain,
                max_examples=MAX_FEW_SHOT_EXAMPLES,
                similarity_threshold=FEW_SHOT_SIMILARITY_THRESHOLD
            )
            if examples:
                example_list = []
                for i, ex in enumerate(examples, 1):
                    example_list.append(f"Пример {i}:\nВход: {ex['input']}\nВыход: {ex['output']}")
                examples_text = "ПРИМЕРЫ УСПЕШНОЙ КОРРЕКЦИИ:\n" + "\n\n".join(example_list)
                self.logger.info(f"[PromptGen] Добавлено {len(examples)} динамических примеров")

        prompt_variants = self._generate_prompts_via_llm(language, language_code, input_text, examples_text)
        prompt_variants = [self._ensure_text_placeholder(p) for p in prompt_variants]

        return {
            "prompt_correction_variants": prompt_variants,
            "prompt_correction": prompt_variants[0],
            "detected_language": language,
            "input_text_for_correction": input_text
        }

    def _generate_prompts_via_llm(self, language: str, language_code: str, input_text: str, examples: str) -> List[str]:
        variants = []
        system_prompt = self._get_system_prompt(language, language_code)
        user_prompt = self._get_user_prompt(language, language_code, examples)
        temperatures = [0.3, 0.6, 0.9]

        for i, temp in enumerate(temperatures[:self.num_variants]):
            try:
                response = self.client.generate(
                    prompt=user_prompt,
                    temperature=temp,
                    system_prompt=system_prompt,
                    max_tokens=2048
                )
                if response and response.strip():
                    prompt = self._extract_prompt_from_response(response)
                    if prompt:
                        variants.append(prompt)
            except Exception as e:
                self.logger.error(f"[PromptGen] Ошибка генерации промпта #{i+1}: {e}")

        if not variants:
            variants = self._get_fallback_prompts(language, language_code)
        return variants

    def _get_system_prompt(self, language: str, language_code: str) -> str:
        if language_code == "ru":
            return f"""Ты - эксперт по созданию промптов для исправления ошибок в текстах на {language} языке.

ТВОЯ ЗАДАЧА: Создать эффективный промпт для задачи исправления ошибок.

ТРЕБОВАНИЯ К ПРОМПТУ:
1. Чётко определи роль (профессиональный редактор)
2. Укажи конкретные задачи (орфография, грамматика, пунктуация)
3. Добавь требования к сохранению стиля
4. Включи инструкцию о формате вывода (только исправленный текст)
5. ОБЯЗАТЕЛЬНО используй плейсхолдер {{text}} для подстановки текста

ВЕРНИ ТОЛЬКО текст промпта без дополнительных комментариев."""
        else:
            return f"""You are an expert at creating prompts for text error correction in {language}.

YOUR TASK: Create an effective prompt for error correction.

PROMPT REQUIREMENTS:
1. Clearly define the role (professional editor)
2. Specify tasks (spelling, grammar, punctuation)
3. Add requirements to preserve style
4. Include output format instruction (only corrected text)
5. MUST use the {{text}} placeholder for text substitution

Return ONLY the prompt text without additional comments."""

    def _get_user_prompt(self, language: str, language_code: str, examples: str) -> str:
        if language_code == "ru":
            return f"""Создай промпт для исправления ошибок в тексте на русском языке.

{examples}

Создай НОВЫЙ промпт, похожий на эти примеры, но со своими формулировками. ОБЯЗАТЕЛЬНО включи плейсхолдер {{text}}.

ВЕРНИ ТОЛЬКО текст промпта:"""
        else:
            return f"""Create a prompt for text error correction in English.

{examples}

Create a NEW prompt similar to these examples but with your own wording. MUST include the {{text}} placeholder.

Return ONLY the prompt text:"""

    def _ensure_text_placeholder(self, prompt: str) -> str:
        if "{text}" not in prompt:
            prompt += "\n\nТЕКСТ ДЛЯ КОРРЕКЦИИ:\n{text}\n\nИСПРАВЛЕННЫЙ ТЕКСТ:"
        return prompt

    def _extract_prompt_from_response(self, response: str) -> str:
        prompt = response.strip()
        prefixes_to_remove = ["Вот промпт:", "Промпт:", "Prompt:", "Here is the prompt:", "Вот:"]
        for prefix in prefixes_to_remove:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):].strip()
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1].strip()
        return prompt

    def _get_fallback_prompts(self, language: str, language_code: str) -> List[str]:
        if language_code == "ru":
            return [
                f"Ты - профессиональный редактор текстов на {language} языке. Исправь все ошибки в тексте, сохраняя смысл. ТЕКСТ: {{text}} ИСПРАВЛЕННЫЙ ТЕКСТ:",
                f"Ты - опытный редактор на {language} языке. Найди и исправь все ошибки. Сохрани стиль. Текст: {{text}} Результат:",
                f"Ты - эксперт по коррекции текстов на {language} языке. Внимательно проверь текст и исправь ошибки. Текст для исправления: {{text}} Исправленный текст:",
            ]
        else:
            return [
                f"You are a professional text editor in {language}. Correct all errors while preserving meaning. TEXT: {{text}} CORRECTED TEXT:",
                f"You are an experienced editor in {language}. Find and fix all errors. Preserve style. Text: {{text}} Result:",
                f"You are a text correction expert in {language}. Carefully check and fix errors. Text to correct: {{text}} Corrected text:",
            ]

    def log_execution(self, message: str):
        self.logger.debug(f"[{self.name}] {message}")