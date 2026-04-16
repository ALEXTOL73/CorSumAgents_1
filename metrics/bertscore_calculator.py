"""
Калькулятор метрики BertScore (семантическая схожесть)
Версия 5.3.0 - Гарантированная однократная загрузка модели (глобальная переменная)
"""
from typing import Dict, Any
from utils.logger import setup_logger
import warnings
import torch

logger = setup_logger("BertScoreCalculator", "bertscore_calculator")

# Глобальные переменные модуля (загружаются один раз)
_BERTSCORE_MODEL = None
_BERTSCORE_DEVICE = None


def _get_device():
    """Определение доступного устройства"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _load_model():
    """Загрузка модели (вызывается один раз)"""
    global _BERTSCORE_MODEL, _BERTSCORE_DEVICE
    if _BERTSCORE_MODEL is not None:
        return _BERTSCORE_MODEL

    warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    try:
        from sentence_transformers import SentenceTransformer
        device = _get_device()
        logger.info(f"[BertScore] Загрузка модели paraphrase-multilingual-MiniLM-L12-v2 на {device} (ОДИН РАЗ)...")
        _BERTSCORE_MODEL = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            trust_remote_code=True,
            device=device
        )
        _BERTSCORE_DEVICE = device
        logger.info(f"[BertScore] Модель успешно загружена на {device}")
    except Exception as e:
        logger.error(f"[BertScore] Ошибка загрузки модели: {e}")
        _BERTSCORE_MODEL = None
    return _BERTSCORE_MODEL


class BertScoreCalculator:
    """
    Расчет метрики BertScore (семантическая схожесть)
    Модель загружается один раз на уровне модуля.
    """

    def __init__(self):
        """Инициализация – модель загружается лениво при первом вызове calculate"""
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._device = _get_device()
        logger.info(f"[BertScore] Калькулятор инициализирован, device={self._device}")

    def _get_embedding(self, text: str):
        """Получение эмбеддинга с кэшированием"""
        if text in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text]

        self.cache_misses += 1
        model = _load_model()
        if model is None:
            return None

        embedding = model.encode(text, convert_to_numpy=True)

        if len(self.embedding_cache) < 1000:
            self.embedding_cache[text] = embedding
        return embedding

    def calculate_p_umfd(self, reference: str, hypothesis: str) -> float:
        """Расчёт BertScore (косинусная схожесть)"""
        if reference is None or hypothesis is None:
            logger.warning("[BertScore] Получены None значения")
            return 0.0
        try:
            reference = str(reference).strip()
            hypothesis = str(hypothesis).strip()
        except Exception as e:
            logger.error(f"[BertScore] Ошибка конвертации: {e}")
            return 0.0

        if not reference or not hypothesis:
            logger.warning("[BertScore] Пустой текст")
            return 0.0

        ref_emb = self._get_embedding(reference)
        hyp_emb = self._get_embedding(hypothesis)

        if ref_emb is None or hyp_emb is None:
            logger.warning("[BertScore] Не удалось получить эмбеддинги")
            return 0.0

        try:
            import numpy as np
            similarity = np.dot(ref_emb, hyp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb))
            similarity = float(max(0.0, min(1.0, similarity)))
            logger.info(f"[BertScore] Семантическая схожесть: {similarity:.4f}")
            logger.debug(f"[BertScore] Cache stats: hits={self.cache_hits}, misses={self.cache_misses}")
            return similarity
        except Exception as e:
            logger.error(f"[BertScore] Ошибка расчета: {e}")
            return 0.0

    def calculate(self, reference: str, candidate: str) -> float:
        """Алиас для calculate_p_umfd"""
        return self.calculate_p_umfd(reference, candidate)

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "device": self._device,
            "gpu_available": torch.cuda.is_available(),
            "model_loaded": _BERTSCORE_MODEL is not None
        }

    def clear_cache(self):
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("[BertScore] Кэш эмбеддингов очищен")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "loaded": _BERTSCORE_MODEL is not None,
            "model_type": type(_BERTSCORE_MODEL).__name__ if _BERTSCORE_MODEL else None,
            "cache_size": len(self.embedding_cache),
            "device": self._device,
            "gpu_available": torch.cuda.is_available()
        }