"""
Агент генерации промптов для суммаризации
Версия 5.6.0 - Семантический поиск few-shot примеров, усиление влияния динамических примеров
Особенности:
- System Prompt: роль эксперта по промптам для суммаризации
- LLM генерирует несколько вариантов промптов
- Few-Shot примеры только для LLM (не в выходных файлах)
- Выбор лучшего промпта по метрикам суммаризации
- ✅ ДОБАВЛЕНО: динамический подбор примеров из памяти агента
- ✅ ДОБАВЛЕНА поддержка memory и передача примеров в user prompt
- ✅ УЛУЧШЕНО: семантический поиск примеров через эмбеддинги (если доступно)
- ✅ УВЕЛИЧЕНО количество динамических примеров до 3
- ✅ ПОНИЖЕН порог схожести для более широкого выбора
"""
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from utils.lmstudio_client import LMStudioClient
from utils.agent_memory import AgentMemory
from config import ENSEMBLE_SIZE, SUMMARY_DYNAMIC_FEW_SHOT_ENABLED, SUMMARY_MAX_FEW_SHOT_EXAMPLES, SUMMARY_FEW_SHOT_LENGTH_RATIO

# Для семантического поиска (опционально)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

class SummarizationPromptGenerator(BaseAgent):
    """
    Генерация промптов для задачи суммаризации через LLM

    LLM создаёт несколько вариантов промптов на основе:
    - System Prompt (роль эксперта по промптам)
    - Few-Shot Examples (примеры хороших промптов)
    - Chain of Thought (пошаговое создание промпта)
    - ✅ Динамические примеры из памяти агента
    - ✅ Семантический поиск примеров (через эмбеддинги)
    """

    def __init__(self, client: LMStudioClient, num_variants: int = 3, memory: Optional[AgentMemory] = None):
        """
        Инициализация агента

        Args:
            client: Клиент LM Studio
            num_variants: Количество вариантов промптов для генерации
            memory: Память агента для динамических примеров
        """
        super().__init__(client, "SummarizationPromptGenerator")
        self.num_variants = num_variants
        self.memory = memory
        self._embedding_model = None
        self.logger.info(f"[SummarizationPromptGenerator] Инициализирован v5.6.0 (семантический few-shot: {SUMMARY_DYNAMIC_FEW_SHOT_ENABLED})")
        if _HAS_SENTENCE_TRANSFORMERS:
            self.logger.info("[SummarizationPromptGenerator] sentence-transformers доступен для семантического поиска")
        else:
            self.logger.warning("[SummarizationPromptGenerator] sentence-transformers не установлен, семантический поиск недоступен")

    def _get_embedding_model(self):
        """Ленивая загрузка модели эмбеддингов"""
        if self._embedding_model is not None:
            return self._embedding_model
        if not _HAS_SENTENCE_TRANSFORMERS:
            return None
        try:
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.logger.info("[SummarizationPromptGenerator] Модель эмбеддингов загружена")
        except Exception as e:
            self.logger.warning(f"[SummarizationPromptGenerator] Не удалось загрузить модель эмбеддингов: {e}")
            self._embedding_model = None
        return self._embedding_model

    def _get_embedding(self, text: str):
        """Получение эмбеддинга текста"""
        model = self._get_embedding_model()
        if model is None:
            return None
        try:
            return model.encode(text, convert_to_numpy=True)
        except Exception:
            return None

    def _semantic_search_examples(self, input_text: str, examples: List[Dict], top_k: int = 3) -> List[Dict]:
        """Поиск семантически близких примеров через эмбеддинги"""
        if not examples or self._get_embedding_model() is None:
            return examples[:top_k] if examples else []
        input_emb = self._get_embedding(input_text)
        if input_emb is None:
            return examples[:top_k]
        scored = []
        for ex in examples:
            ex_emb = self._get_embedding(ex.get("input", ""))
            if ex_emb is None:
                similarity = 0
            else:
                import numpy as np
                similarity = np.dot(input_emb, ex_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(ex_emb))
            scored.append((similarity, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация вариантов промптов для суммаризации через LLM

        Args:
            state: Текущее состояние графа

        Returns:
            Обновлённое состояние с вариантами промптов
        """
        self.log_execution("Генерация промптов для суммаризации через LLM")

        input_text = state.get("corrected_text", "") or state.get("input_text", "")
        domain = state.get("summary_domain", "general")

        # Определение языка
        has_cyrillic = any(ord(c) > 127 for c in str(input_text))
        language = "русском" if has_cyrillic else "английском"
        language_code = "ru" if has_cyrillic else "en"

        # ✅ ДИНАМИЧЕСКИЕ ПРИМЕРЫ ИЗ ПАМЯТИ (с семантическим поиском)
        examples_text = ""
        if SUMMARY_DYNAMIC_FEW_SHOT_ENABLED and self.memory:
            # Получаем примеры из памяти (по умолчанию метод get_summary_few_shot_examples)
            examples = self.memory.get_summary_few_shot_examples(
                input_text=input_text,
                domain=domain,
                max_examples=SUMMARY_MAX_FEW_SHOT_EXAMPLES,
                length_ratio=SUMMARY_FEW_SHOT_LENGTH_RATIO
            )
            # Улучшаем семантическим поиском (если доступен)
            if examples and self._get_embedding_model() is not None:
                examples = self._semantic_search_examples(input_text, examples, top_k=SUMMARY_MAX_FEW_SHOT_EXAMPLES)
                self.logger.info(f"[PromptGen] Семантический поиск: выбрано {len(examples)} примеров")
            if examples:
                example_list = []
                for i, ex in enumerate(examples, 1):
                    example_list.append(f"Пример {i}:\nВход: {ex['input']}\nВыход: {ex['output']}")
                examples_text = "ПРИМЕРЫ ХОРОШИХ РЕЗЮМЕ:\n" + "\n\n".join(example_list)
                self.logger.info(f"[PromptGen] Добавлено {len(examples)} динамических примеров суммаризации (семантический поиск)")

        # Генерация нескольких вариантов промптов через LLM
        prompt_variants = self._generate_prompts_via_llm(language, language_code, input_text, examples_text)

        self.logger.info(f"[PromptGen] Сгенерировано {len(prompt_variants)} вариантов промптов через LLM")

        return {
            "prompt_summary_variants": prompt_variants,
            "prompt_summary": prompt_variants[0],  # Первый по умолчанию
            "summary_language": language,
            "input_text_for_summary": input_text
        }

    def _generate_prompts_via_llm(self, language: str, language_code: str, input_text: str, examples: str = "") -> List[str]:
        """
        Генерация вариантов промптов через LLM

        Args:
            language: Название языка
            language_code: Код языка
            input_text: Исходный текст для анализа
            examples: Динамические примеры для few-shot

        Returns:
            Список сгенерированных промптов
        """
        variants = []

        # System Prompt для генерации промптов
        system_prompt = self._get_system_prompt(language, language_code)

        # User Prompt с Few-Shot примерами (статическими + динамическими)
        user_prompt = self._get_user_prompt(language, language_code, examples)

        # Генерация нескольких вариантов через разные температуры
        temperatures = [0.4, 0.6, 0.8]

        for i, temp in enumerate(temperatures[:self.num_variants]):
            try:
                response = self.client.generate(
                    prompt=user_prompt,
                    temperature=temp,
                    system_prompt=system_prompt,
                    max_tokens=2048
                )

                if response and response.strip():
                    # Извлекаем только промпт из ответа
                    prompt = self._extract_prompt_from_response(response)
                    if prompt:
                        variants.append(prompt)
                        self.logger.debug(f"[PromptGen] Вариант #{i+1} (temp={temp}) сгенерирован")

            except Exception as e:
                self.logger.error(f"[PromptGen] Ошибка генерации промпта #{i+1}: {e}")

        # Если LLM не сгенерировал промпты, используем fallback
        if not variants:
            self.logger.warning("[PromptGen] LLM не сгенерировал промпты, используем fallback")
            variants = self._get_fallback_prompts(language, language_code)

        return variants

    def _get_system_prompt(self, language: str, language_code: str) -> str:
        """
        System Prompt для LLM-генератора промптов суммаризации

        Args:
            language: Название языка
            language_code: Код языка

        Returns:
            Текст system prompt
        """
        if language_code == "ru":
            return f"""Ты - эксперт по созданию промптов для суммаризации текстов на {language} языке.

ТВОЯ ЗАДАЧА: Создать эффективный промпт для задачи создания краткой выжимки текста.

ТРЕБОВАНИЯ К ПРОМПТУ:
1. Чётко определи роль (эксперт по суммаризации)
2. Укажи конкретные задачи (сохранить ключевые факты, убрать второстепенное)
3. Добавь требования к длине (1-4 предложения)
4. Включи инструкцию о формате вывода (только текст суммаризации)
5. Будь лаконичным, но подробным

ВЕРНИ ТОЛЬКО текст промпта без дополнительных комментариев."""

        else:  # English
            return f"""You are an expert at creating prompts for text summarization in {language}.

YOUR TASK: Create an effective prompt for creating a brief summary of text.

PROMPT REQUIREMENTS:
1. Clearly define the role (summarization expert)
2. Specify tasks (preserve key facts, remove secondary details)
3. Add length requirements (1-4 sentences)
4. Include output format instruction (only summary text)
5. Be concise but detailed

Return ONLY the prompt text without additional comments."""

    def _get_user_prompt(self, language: str, language_code: str, dynamic_examples: str = "") -> str:
        """
        User Prompt с Few-Shot примерами для LLM (статическими + динамическими)

        Args:
            language: Название языка
            language_code: Код языка
            dynamic_examples: Динамические примеры из памяти

        Returns:
            Текст user prompt
        """
        # Статические примеры
        static_examples = self._get_static_examples(language_code)

        # Объединяем примеры (динамические – первыми, так как они релевантнее)
        all_examples = ""
        if dynamic_examples:
            all_examples = dynamic_examples + "\n\n" + static_examples
        else:
            all_examples = static_examples

        if language_code == "ru":
            return f"""Создай промпт для суммаризации текста.

{all_examples}

Создай НОВЫЙ промпт, похожий на эти примеры, но со своими формулировками.

ВЕРНИ ТОЛЬКО текст промпта:"""

        else:  # English
            return f"""Create a prompt for text summarization.

{all_examples}

Create a NEW prompt similar to these examples but with your own wording.

Return ONLY the prompt text:"""

    def _get_static_examples(self, language_code: str) -> str:
        """Возвращает статические few-shot примеры"""
        if language_code == "ru":
            return """ПРИМЕРЫ ХОРОШИХ ПРОМПТОВ:

Пример 1 (базовый):
"Ты - эксперт по суммаризации текстов на русском языке. Создай краткую выжимку текста. Сохрани ключевые идеи и факты. Убери второстепенные детали. Длина: 1-4 предложения. Стиль: формальный, информативный. Верни ТОЛЬКО текст суммаризации без дополнительных комментариев."

Пример 2 (с акцентом на факты):
"Ты - эксперт по суммаризации на русском языке. Выдели ключевые факты (кто, что, где, когда). Сохрани числа, даты, имена. Убери эмоциональные оценки. Создай краткое содержание в 1-4 предложениях. Верни ТОЛЬКО текст суммаризации."

Пример 3 (с пошаговым подходом):
"Ты - эксперт по суммаризации на русском языке. 1) Прочитай текст внимательно. 2) Выдели ключевые факты. 3) Отбрось второстепенное. 4) Сформулируй краткое содержание в 1-4 предложениях. Верни ТОЛЬКО текст суммаризации."""
        else:
            return """EXAMPLES OF GOOD PROMPTS:

Example 1 (basic):
"You are a text summarization expert in English. Create a brief summary of the text. Preserve key ideas and facts. Remove secondary details. Length: 1-4 sentences. Style: formal, informative. Return ONLY the summary text without additional comments."

Example 2 (facts-focused):
"You are a summarization expert in English. Identify key facts (who, what, where, when). Preserve numbers, dates, names. Remove emotional assessments. Create a brief summary in 1-4 sentences. Return ONLY the summary text."

Example 3 (step-by-step):
"You are a summarization expert in English. 1) Read the text carefully. 2) Identify key facts. 3) Discard secondary details. 4) Formulate a brief summary in 1-4 sentences. Return ONLY the summary text."""

    def _extract_prompt_from_response(self, response: str) -> str:
        """
        Извлечение промпта из ответа LLM

        Args:
            response: Ответ от LLM

        Returns:
            Очищенный текст промпта
        """
        # Удаляем возможные префиксы
        prefixes_to_remove = [
            "Вот промпт:",
            "Промпт:",
            "Prompt:",
            "Here is the prompt:",
            "Вот:",
        ]

        prompt = response.strip()

        for prefix in prefixes_to_remove:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):].strip()

        # Удаляем кавычки если они есть
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1].strip()

        return prompt

    def _get_fallback_prompts(self, language: str, language_code: str) -> List[str]:
        """
        Fallback промпты если LLM не сгенерировал

        Args:
            language: Название языка
            language_code: Код языка

        Returns:
            Список fallback промптов
        """
        if language_code == "ru":
            return [
                f"Ты - эксперт по суммаризации текстов на {language} языке. Создай краткую выжимку, сохранив ключевые факты. Длина: 1-4 предложения. Верни ТОЛЬКО текст суммаризации.",
                f"Ты - специалист по суммаризации на {language} языке. Выдели главное, убери второстепенное. Создай краткое содержание. Длина: 1-4 предложения. Верни ТОЛЬКО текст суммаризации.",
                f"Ты - эксперт по созданию саммари на {language} языке. Сохрани ключевую информацию в 1-4 предложениях. Верни ТОЛЬКО текст суммаризации.",
            ]
        else:
            return [
                f"You are a text summarization expert in {language}. Create a brief summary preserving key facts. Length: 1-4 sentences. Return ONLY the summary text.",
                f"You are a summarization specialist in {language}. Highlight the main points, remove secondary details. Create a brief summary. Length: 1-4 sentences. Return ONLY the summary text.",
                f"You are a summary expert in {language}. Preserve key information in 1-4 sentences. Return ONLY the summary text.",
            ]

    def log_execution(self, message: str):
        self.logger.debug(f"[{self.name}] {message}")