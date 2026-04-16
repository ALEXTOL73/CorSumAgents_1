"""
Глобальный синглтон для модели эмбеддингов (sentence-transformers)
Версия 1.0 - Единая модель для всех компонентов системы
"""
import torch
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("EmbeddingModel")


class EmbeddingModel:
    """Синглтон для модели эмбеддингов, загружается один раз"""
    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._device = self._get_device()
        self._load_model()
        logger.info(f"[EmbeddingModel] Инициализирован, device={self._device}")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=self._device
            )
            logger.info("[EmbeddingModel] Модель загружена")
        except Exception as e:
            logger.error(f"[EmbeddingModel] Ошибка загрузки модели: {e}")
            self._model = None

    def encode(self, text: str):
        if self._model is None:
            return None
        return self._model.encode(text, convert_to_numpy=True)

    def is_available(self) -> bool:
        return self._model is not None