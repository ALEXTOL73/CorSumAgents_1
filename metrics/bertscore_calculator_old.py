"""
Калькулятор метрики BertScore (семантическая схожесть)
Версия 5.1.0 - Добавлена поддержка GPU (device='cuda')
Особенности:
- Singleton паттерн - модель загружается только один раз
- Явное указание устройства (CPU/GPU)
- Подавление предупреждений от transformers
- Метрика BertScore (косинусная схожесть)
- Кэширование эмбеддингов для повторяющихся текстов
- Полная обратная совместимость с предыдущими версиями
"""
from typing import Dict, Any
from utils.logger import setup_logger
import warnings
import torch

logger = setup_logger("BertScoreCalculator", "bertscore_calculator")


class BertScoreCalculator:
    """
    Расчет метрики BertScore (семантическая схожесть)
    Улучшение 1.3: Singleton паттерн для экономии памяти
    - Модель загружается только один раз для всех экземпляров
    - Кэширование эмбеддингов для повторяющихся текстов
    - Явная поддержка GPU (device='cuda')

    Метрика BertScore = косинусная схожесть эмбеддингов (0-1)
    """

    # Singleton instance
    _instance = None
    _model = None
    _device = None
    _initialized = False

    def __new__(cls):
        """Singleton паттерн для загрузки модели один раз"""
        if cls._instance is None:
            cls._instance = super(BertScoreCalculator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация (модель загружается лениво)"""
        if self._initialized:
            return

        self._model = None
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._initialized = True
        self._device = self._get_device()

        logger.info(f"[BertScore] Калькулятор инициализирован (Singleton), device={self._device}")

    def _get_device(self):
        """Определение доступного устройства (GPU / CPU)"""
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("[BertScore] ✅ GPU (CUDA) доступен, будет использован")
        else:
            device = 'cpu'
            logger.info("[BertScore] GPU не доступен, используется CPU")
        return device

    def _initialize_model(self):
        """
        Ленивая инициализация модели с подавлением предупреждений
        Улучшение 1.3: Модель загружается только один раз
        Улучшение 5.1.0: Явное указание устройства
        """
        if self._model is not None:
            return

        # Подавляем warnings от sentence-transformers о position_ids
        warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("[BertScore] Загрузка модели paraphrase-multilingual-MiniLM-L12-v2...")

            # Загружаем модель с указанием устройства
            self._model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                trust_remote_code=True,
                device=self._device  # Явное указание устройства (GPU/CPU)
            )

            # Дополнительная проверка устройства
            if self._device == 'cuda':
                logger.info("[BertScore] Модель загружена на GPU")
            else:
                logger.info("[BertScore] Модель загружена на CPU")

        except Exception as e:
            logger.error(f"[BertScore] Ошибка загрузки модели: {e}")
            self._model = None

    def _get_embedding(self, text: str):
        """
        Получение эмбеддинга с кэшированием
        Улучшение 1.3: Кэширование эмбеддингов

        Args:
            text: Текст для получения эмбеддинга

        Returns:
            Эмбеддинг текста (numpy array)
        """
        # Проверка кэша
        if text in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text]

        self.cache_misses += 1

        # Инициализация модели если нужно
        self._initialize_model()

        if self._model is None:
            logger.warning("[BertScore] Модель не загружена, возвращаем None")
            return None

        # Получение эмбеддинга (SentenceTransformer автоматически использует указанное устройство)
        embedding = self._model.encode(text, convert_to_numpy=True)

        # Сохранение в кэш (ограничиваем размер)
        if len(self.embedding_cache) < 1000:
            self.embedding_cache[text] = embedding

        return embedding

    def calculate_p_umfd(self, reference: str, hypothesis: str) -> float:
        """
        Расчет метрики BertScore (семантическая схожесть на основе эмбеддингов)
        Формула: cosine_similarity = (A · B) / (||A|| × ||B||)

        Args:
            reference: Эталонный текст
            hypothesis: Сгенерированный текст

        Returns:
            Cosine similarity (0.0 - 1.0)
        """
        # Проверка на None
        if reference is None or hypothesis is None:
            logger.warning("[BertScore] Получены None значения")
            return 0.0

        try:
            reference = str(reference).strip()
            hypothesis = str(hypothesis).strip()
        except Exception as e:
            logger.error(f"[BertScore] Ошибка конвертации: {e}")
            return 0.0

        if len(reference) == 0 or len(hypothesis) == 0:
            logger.warning("[BertScore] Пустой текст")
            return 0.0

        # Получение эмбеддингов с кэшированием
        ref_embedding = self._get_embedding(reference)
        hyp_embedding = self._get_embedding(hypothesis)

        if ref_embedding is None or hyp_embedding is None:
            logger.warning("[BertScore] Не удалось получить эмбеддинги")
            return 0.0

        # Расчет схожести
        try:
            import numpy as np

            # Cosine similarity
            similarity = np.dot(ref_embedding, hyp_embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(hyp_embedding)
            )

            # Ограничение диапазона [0, 1]
            similarity = float(max(0.0, min(1.0, similarity)))

            logger.info(f"[BertScore] Семантическая схожесть: {similarity:.4f}")
            logger.debug(f"[BertScore] Cache stats: hits={self.cache_hits}, misses={self.cache_misses}")

            return similarity

        except Exception as e:
            logger.error(f"[BertScore] Ошибка расчета: {e}")
            return 0.0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кэша эмбеддингов

        Returns:
            Словарь со статистикой кэша
        """
        return {
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (
                                                                                                       self.cache_hits + self.cache_misses) > 0 else 0,
            "device": self._device,
            "gpu_available": torch.cuda.is_available()
        }

    def clear_cache(self):
        """Очистка кэша эмбеддингов"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("[BertScore] Кэш эмбеддингов очищен")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели

        Returns:
            Словарь с информацией о модели
        """
        if self._model is None:
            return {"loaded": False, "device": self._device}

        return {
            "loaded": True,
            "model_type": type(self._model).__name__,
            "cache_size": len(self.embedding_cache),
            "device": self._device,
            "gpu_available": torch.cuda.is_available()
        }

    # Алиас для совместимости с другими модулями
    def calculate(self, reference: str, candidate: str) -> float:
        """Алиас для calculate_p_umfd (для единообразия)"""
        return self.calculate_p_umfd(reference, candidate)