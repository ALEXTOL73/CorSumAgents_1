"""
Память агентов для обучения на предыдущих ошибках
Версия 5.6.0 - Добавлен семантический поиск few‑shot через эмбеддинги
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import PROMPT_CACHE_ENABLED, PROMPT_CACHE_MAX_SIZE, PROMPT_CACHE_MIN_IMPROVEMENT, DATA_LANG_DIR
from utils.logger import setup_logger

logger = setup_logger("AgentMemory")


class AgentMemory:
    def __init__(self, memory_dir: str = None):
        # ✅ Если memory_dir не указан, используем DATA_LANG_DIR
        if memory_dir is None:
            memory_dir = f"data/{DATA_LANG_DIR}/memory"
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # История коррекции и суммаризации
        self.correction_history = []
        self.summary_history = []

        self.common_errors = {}
        self.success_patterns = {}
        self.domain_stats = {}

        # Лучшие промпты
        self.best_prompts = {}
        self.best_summary_prompts = {}

        self.prompt_cache = {}
        self.prompt_usage_stats = {}

        # Few-shot примеры
        self.few_shot_examples = []
        self.summary_few_shot_examples = []

        # Кэш эмбеддингов для few-shot (опционально, лениво)
        self._embedding_cache = {}
        self._embedding_model = None

        self._load_memory()
        logger.info(f"[AgentMemory] Память загружена из {self.memory_dir}")
        logger.info(f"[AgentMemory] История коррекции: {len(self.correction_history)} записей")
        logger.info(f"[AgentMemory] История суммаризации: {len(self.summary_history)} записей")
        logger.info(f"[AgentMemory] Few-shot примеров коррекции: {len(self.few_shot_examples)}")
        logger.info(f"[AgentMemory] Few-shot примеров суммаризации: {len(self.summary_few_shot_examples)}")

    def _load_memory(self):
        try:
            history_file = self.memory_dir / "correction_history.json"
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    self.correction_history = json.load(f)

            summary_history_file = self.memory_dir / "summary_history.json"
            if summary_history_file.exists():
                with open(summary_history_file, "r", encoding="utf-8") as f:
                    self.summary_history = json.load(f)

            errors_file = self.memory_dir / "common_errors.json"
            if errors_file.exists():
                with open(errors_file, "r", encoding="utf-8") as f:
                    self.common_errors = json.load(f)

            patterns_file = self.memory_dir / "success_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, "r", encoding="utf-8") as f:
                    self.success_patterns = json.load(f)

            stats_file = self.memory_dir / "domain_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    loaded_stats = json.load(f)
                    for domain, stats in loaded_stats.items():
                        self.domain_stats[domain] = stats

            prompts_file = self.memory_dir / "best_prompts.json"
            if prompts_file.exists():
                with open(prompts_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.best_prompts = loaded
                    elif isinstance(loaded, list):
                        self.best_prompts = {"general": loaded}
                    else:
                        self.best_prompts = {}

            summary_prompts_file = self.memory_dir / "best_summary_prompts.json"
            if summary_prompts_file.exists():
                with open(summary_prompts_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.best_summary_prompts = loaded
                    elif isinstance(loaded, list):
                        self.best_summary_prompts = {"general": loaded}
                    else:
                        self.best_summary_prompts = {}

            cache_file = self.memory_dir / "prompt_cache.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.prompt_cache = json.load(f)

            usage_file = self.memory_dir / "prompt_usage_stats.json"
            if usage_file.exists():
                with open(usage_file, "r", encoding="utf-8") as f:
                    self.prompt_usage_stats = json.load(f)

            # Загрузка few‑shot примеров
            few_shot_file = self.memory_dir / "few_shot_examples.json"
            if few_shot_file.exists():
                with open(few_shot_file, "r", encoding="utf-8") as f:
                    self.few_shot_examples = json.load(f)

            summary_few_shot_file = self.memory_dir / "summary_few_shot_examples.json"
            if summary_few_shot_file.exists():
                with open(summary_few_shot_file, "r", encoding="utf-8") as f:
                    self.summary_few_shot_examples = json.load(f)

        except Exception as e:
            logger.warning(f"[AgentMemory] Ошибка загрузки памяти: {e}")

    def _save_memory(self):
        try:
            if len(self.correction_history) > 1000:
                self.correction_history = self.correction_history[-1000:]
            if len(self.summary_history) > 1000:
                self.summary_history = self.summary_history[-1000:]
            if len(self.few_shot_examples) > 100:
                self.few_shot_examples = self.few_shot_examples[:100]
            if len(self.summary_few_shot_examples) > 100:
                self.summary_few_shot_examples = self.summary_few_shot_examples[:100]

            if len(self.prompt_cache) > PROMPT_CACHE_MAX_SIZE:
                sorted_cache = sorted(
                    self.prompt_cache.items(),
                    key=lambda x: x[1].get('last_used', 0),
                    reverse=True
                )[:PROMPT_CACHE_MAX_SIZE]
                self.prompt_cache = dict(sorted_cache)

            with open(self.memory_dir / "correction_history.json", "w", encoding="utf-8") as f:
                json.dump(self.correction_history, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "summary_history.json", "w", encoding="utf-8") as f:
                json.dump(self.summary_history, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "common_errors.json", "w", encoding="utf-8") as f:
                json.dump(self.common_errors, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "success_patterns.json", "w", encoding="utf-8") as f:
                json.dump(self.success_patterns, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "domain_stats.json", "w", encoding="utf-8") as f:
                json.dump(self.domain_stats, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "best_prompts.json", "w", encoding="utf-8") as f:
                json.dump(self.best_prompts, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "best_summary_prompts.json", "w", encoding="utf-8") as f:
                json.dump(self.best_summary_prompts, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "prompt_cache.json", "w", encoding="utf-8") as f:
                json.dump(self.prompt_cache, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "prompt_usage_stats.json", "w", encoding="utf-8") as f:
                json.dump(self.prompt_usage_stats, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "few_shot_examples.json", "w", encoding="utf-8") as f:
                json.dump(self.few_shot_examples, f, ensure_ascii=False, indent=2)

            with open(self.memory_dir / "summary_few_shot_examples.json", "w", encoding="utf-8") as f:
                json.dump(self.summary_few_shot_examples, f, ensure_ascii=False, indent=2)

            logger.debug("[AgentMemory] Память сохранена")
        except Exception as e:
            logger.error(f"[AgentMemory] Ошибка сохранения памяти: {e}")

    # ========== Базовые методы ==========
    def get_correction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.correction_history[-limit:] if limit > 0 else self.correction_history

    def get_summary_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.summary_history[-limit:] if limit > 0 else self.summary_history

    def get_error_profile(self, domain: str) -> Dict[str, int]:
        if not self.common_errors:
            return {}
        top_errors = sorted(self.common_errors.items(), key=lambda x: x[1], reverse=True)[:10]
        return dict(top_errors)

    def save_optimization_result(self, result: Dict[str, Any]):
        opt_file = self.memory_dir / "optimization_results.json"
        try:
            if opt_file.exists():
                with open(opt_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(result)
            history = history[-100:]
            with open(opt_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[AgentMemory] Ошибка сохранения оптимизации: {e}")

    # ========== Обучение ==========
    def learn_from_correction(self, original: str, corrected: str, reference: str,
                              wer_before: float, wer_after: float, domain: str = "general",
                              prompt_used: str = "", model_used: str = "", error_profile: Dict = None):
        improvement = wer_before - wer_after
        record = {
            "timestamp": datetime.now().isoformat(),
            "original_length": len(original),
            "corrected_length": len(corrected),
            "wer_before": wer_before,
            "wer_after": wer_after,
            "improvement": improvement,
            "domain": domain,
            "prompt_used": prompt_used[:200] if prompt_used else "",
            "model_used": model_used,
            "best_temperature": "0.5",
            "metrics": {"CorScore": improvement * 1.0}
        }
        self.correction_history.append(record)

        differences = self._find_differences(original, corrected)
        for diff in differences:
            key = f"{diff['original']}→{diff['corrected']}"
            self.common_errors[key] = self.common_errors.get(key, 0) + 1

        if domain not in self.domain_stats:
            self.domain_stats[domain] = {"count": 0, "avg_improvement": 0, "successes": 0, "success_rate": 0}
        self.domain_stats[domain]["count"] += 1
        old_avg = self.domain_stats[domain]["avg_improvement"]
        count = self.domain_stats[domain]["count"]
        self.domain_stats[domain]["avg_improvement"] = old_avg + (improvement - old_avg) / count
        if improvement > 0:
            self.domain_stats[domain]["successes"] += 1
        self.domain_stats[domain]["success_rate"] = self.domain_stats[domain]["successes"] / count

        if PROMPT_CACHE_ENABLED and improvement >= PROMPT_CACHE_MIN_IMPROVEMENT and prompt_used:
            prompt_hash = hashlib.md5(prompt_used.encode('utf-8')).hexdigest()
            self.prompt_cache[prompt_hash] = {
                "prompt": prompt_used[:500],
                "improvement": improvement,
                "domain": domain,
                "model": model_used,
                "timestamp": record["timestamp"],
                "last_used": record["timestamp"]
            }
            if domain not in self.best_prompts:
                self.best_prompts[domain] = []
            if not any(p.get("prompt") == prompt_used[:200] for p in self.best_prompts[domain]):
                self.best_prompts[domain].append({
                    "prompt": prompt_used[:200],
                    "improvement": improvement,
                    "timestamp": record["timestamp"],
                    "model": model_used,
                    "prompt_hash": prompt_hash
                })
                if isinstance(self.best_prompts[domain], list):
                    self.best_prompts[domain].sort(key=lambda x: x.get("improvement", 0), reverse=True)
                    self.best_prompts[domain] = self.best_prompts[domain][:10]

            if prompt_hash not in self.prompt_usage_stats:
                self.prompt_usage_stats[prompt_hash] = {
                    "prompt": prompt_used[:200],
                    "usage_count": 0,
                    "total_improvement": 0,
                    "avg_improvement": 0,
                    "success_count": 0,
                    "last_used": record["timestamp"],
                    "domain": domain,
                    "model": model_used
                }
            stats = self.prompt_usage_stats[prompt_hash]
            stats["usage_count"] += 1
            stats["total_improvement"] += improvement
            stats["avg_improvement"] = stats["total_improvement"] / stats["usage_count"]
            stats["last_used"] = record["timestamp"]
            if improvement > 0:
                stats["success_count"] += 1

        if improvement > 0:
            self.add_few_shot_example(original, corrected, error_type="mixed", domain=domain, success=True)

        self._save_memory()
        logger.debug(f"[AgentMemory] Запомнено исправление: {len(differences)} ошибок, улучшение: {improvement:.4f}")

    def learn_from_summarization(self, original: str, summary: str, reference: str,
                                 prompt_used: str = "", model_used: str = "",
                                 metrics: Dict[str, Any] = None):
        sumscore = metrics.get("SumScore", 0) if metrics else 0
        improvement = sumscore
        record = {
            "timestamp": datetime.now().isoformat(),
            "original_length": len(original),
            "summary_length": len(summary),
            "reference_length": len(reference) if reference else 0,
            "sumscore": sumscore,
            "improvement": improvement,
            "prompt_used": prompt_used[:200] if prompt_used else "",
            "model_used": model_used,
            "metrics": metrics or {}
        }
        self.summary_history.append(record)

        if PROMPT_CACHE_ENABLED and improvement >= PROMPT_CACHE_MIN_IMPROVEMENT and prompt_used:
            prompt_hash = hashlib.md5(prompt_used.encode('utf-8')).hexdigest()
            domain = "general"
            if domain not in self.best_summary_prompts:
                self.best_summary_prompts[domain] = []
            if not any(p.get("prompt") == prompt_used[:200] for p in self.best_summary_prompts[domain]):
                self.best_summary_prompts[domain].append({
                    "prompt": prompt_used[:200],
                    "sumscore": sumscore,
                    "improvement": improvement,
                    "timestamp": record["timestamp"],
                    "model": model_used,
                    "prompt_hash": prompt_hash
                })
                if isinstance(self.best_summary_prompts[domain], list):
                    self.best_summary_prompts[domain].sort(key=lambda x: x.get("sumscore", 0), reverse=True)
                    self.best_summary_prompts[domain] = self.best_summary_prompts[domain][:10]

            if prompt_hash not in self.prompt_usage_stats:
                self.prompt_usage_stats[prompt_hash] = {
                    "prompt": prompt_used[:200],
                    "usage_count": 0,
                    "total_improvement": 0,
                    "avg_improvement": 0,
                    "success_count": 0,
                    "last_used": record["timestamp"],
                    "domain": domain,
                    "model": model_used
                }
            stats = self.prompt_usage_stats[prompt_hash]
            stats["usage_count"] += 1
            stats["total_improvement"] += improvement
            stats["avg_improvement"] = stats["total_improvement"] / stats["usage_count"]
            stats["last_used"] = record["timestamp"]
            if improvement > 0.5:
                stats["success_count"] += 1

        if sumscore > 0.6:
            self.add_summary_few_shot_example(original, summary, length=len(summary.split()), style="neutral", domain="general")

        self._save_memory()
        logger.debug(f"[AgentMemory] Запомнена суммаризация: SumScore={sumscore:.4f}")

    def _find_differences(self, original: str, corrected: str) -> List[Dict]:
        orig_words = original.split()
        corr_words = corrected.split()
        differences = []
        for i, (o, c) in enumerate(zip(orig_words, corr_words)):
            if o != c:
                differences.append({"position": i, "original": o, "corrected": c})
        return differences

    # ========== Получение лучших промптов ==========
    def get_best_prompt_for_domain(self, domain: str = "general") -> Optional[str]:
        prompts = self.best_prompts.get(domain, [])
        if prompts and isinstance(prompts, list) and prompts:
            return prompts[0].get("prompt")
        return None

    def get_best_summary_prompt_for_domain(self, domain: str = "general") -> Optional[str]:
        prompts = self.best_summary_prompts.get(domain, [])
        if prompts and isinstance(prompts, list) and prompts:
            return prompts[0].get("prompt")
        return None

    def get_cached_prompts_by_domain(self, domain: str, limit: int = 5) -> List[Dict]:
        if not PROMPT_CACHE_ENABLED:
            return []
        domain_prompts = [p for p in self.prompt_cache.values() if p.get("domain") == domain]
        domain_prompts.sort(key=lambda x: x.get("improvement", 0), reverse=True)
        return domain_prompts[:limit]

    # ========== Статистика использования промптов ==========
    def get_prompt_usage_stats(self) -> Dict[str, Any]:
        if not self.prompt_usage_stats:
            return {"total_prompts": 0, "avg_improvement": 0, "best_prompt": None, "success_rate": 0}
        total_usage = sum(s["usage_count"] for s in self.prompt_usage_stats.values())
        avg_improvement = sum(s["avg_improvement"] for s in self.prompt_usage_stats.values()) / max(1, len(self.prompt_usage_stats))
        best_prompt = max(self.prompt_usage_stats.values(), key=lambda x: x["avg_improvement"], default=None)
        total_successes = sum(s["success_count"] for s in self.prompt_usage_stats.values())
        success_rate = total_successes / max(1, total_usage)
        return {
            "total_prompts": len(self.prompt_usage_stats),
            "total_usage": total_usage,
            "avg_improvement": avg_improvement,
            "best_prompt": best_prompt["prompt"][:100] if best_prompt else None,
            "best_prompt_improvement": best_prompt["avg_improvement"] if best_prompt else 0,
            "success_rate": success_rate
        }

    # ========== Адаптивная конфигурация ==========
    def get_adaptive_config(self, text: str, reference: str, wer_before: float) -> Dict[str, int]:
        if wer_before > 0.3:
            return {"ensemble_size": 5, "max_retries": 5}
        elif wer_before > 0.15:
            return {"ensemble_size": 3, "max_retries": 3}
        elif wer_before > 0.05:
            return {"ensemble_size": 2, "max_retries": 2}
        else:
            return {"ensemble_size": 1, "max_retries": 1}

    # ========== Продвинутые методы для самообучения ==========
    def save_trajectory(self, trajectory: Dict[str, Any]):
        trajectories_file = self.memory_dir / "trajectories.json"
        try:
            if trajectories_file.exists():
                with open(trajectories_file, "r", encoding="utf-8") as f:
                    trajs = json.load(f)
            else:
                trajs = []
            trajs.append(trajectory)
            if len(trajs) > 500:
                trajs = trajs[-500:]
            with open(trajectories_file, "w", encoding="utf-8") as f:
                json.dump(trajs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[AgentMemory] Ошибка сохранения траектории: {e}")

    def get_relevant_experiences(self, current_context: str, limit: int = 3) -> List[Dict]:
        traj_file = self.memory_dir / "trajectories.json"
        if not traj_file.exists():
            return []
        try:
            with open(traj_file, "r", encoding="utf-8") as f:
                trajs = json.load(f)
            successful = [t for t in trajs if t.get("metrics", {}).get("SumScore", 0) > 0.7 or t.get("metrics", {}).get("CorScore", 0) > 0.5]
            return successful[-limit:]
        except Exception:
            return []

    # ========== СЕМАНТИЧЕСКИЙ ПОИСК ДЛЯ FEW-SHOT (с эмбеддингами) ==========
    def _get_embedding_model(self):
        """Ленивая загрузка модели эмбеддингов (той же, что в BertScore)"""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("[AgentMemory] Загружена модель эмбеддингов для few-shot")
        except Exception as e:
            logger.warning(f"[AgentMemory] Не удалось загрузить модель эмбеддингов: {e}, будет использован fallback")
            self._embedding_model = None
        return self._embedding_model

    def _get_embedding(self, text: str):
        """Получение эмбеддинга с кэшированием"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        model = self._get_embedding_model()
        if model is None:
            return None
        emb = model.encode(text, convert_to_numpy=True)
        self._embedding_cache[text] = emb
        if len(self._embedding_cache) > 500:
            keys = list(self._embedding_cache.keys())[:250]
            for k in keys:
                del self._embedding_cache[k]
        return emb

    def add_few_shot_example(self, input_text: str, corrected_text: str, error_type: str = None,
                             domain: str = "general", success: bool = True):
        example = {
            "input": input_text[:500],
            "output": corrected_text[:500],
            "error_type": error_type or "mixed",
            "domain": domain,
            "success": success,
            "timestamp": time.time(),
            "length": len(input_text)
        }
        self.few_shot_examples.insert(0, example)
        self.few_shot_examples = self.few_shot_examples[:100]
        self._save_memory()
        logger.debug(f"[Memory] Добавлен few-shot пример для {domain}")

    def add_summary_few_shot_example(self, original_text: str, summary: str, length: int,
                                      style: str = "neutral", domain: str = "general"):
        example = {
            "input": original_text[:800],
            "output": summary[:300],
            "length": length,
            "style": style,
            "domain": domain,
            "timestamp": time.time(),
            "char_count": len(original_text)
        }
        self.summary_few_shot_examples.insert(0, example)
        self.summary_few_shot_examples = self.summary_few_shot_examples[:100]
        self._save_memory()
        logger.debug(f"[Memory] Добавлен few-shot пример суммаризации для {domain}")

    def get_few_shot_examples(self, input_text: str, domain: str = None, max_examples: int = 3,
                              similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Получение релевантных примеров с использованием семантического поиска (эмбеддинги).
        Если эмбеддинги недоступны, использует fallback на пересечение слов.
        """
        if not self.few_shot_examples:
            return []

        candidates = self.few_shot_examples
        if domain:
            candidates = [ex for ex in candidates if ex.get("domain") == domain]
        if len(candidates) < max_examples:
            candidates = [ex for ex in self.few_shot_examples if ex.get("success", True)]

        # Пытаемся использовать эмбеддинги
        model = self._get_embedding_model()
        if model is not None:
            try:
                input_emb = self._get_embedding(input_text)
                if input_emb is not None:
                    import numpy as np
                    scored = []
                    for ex in candidates:
                        ex_emb = self._get_embedding(ex["input"])
                        if ex_emb is None:
                            similarity = 0
                        else:
                            similarity = np.dot(input_emb, ex_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(ex_emb))
                        scored.append((similarity, ex))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    filtered = [(sim, ex) for sim, ex in scored if sim >= similarity_threshold]
                    if not filtered:
                        filtered = scored[:max_examples]
                    examples = [ex for _, ex in filtered[:max_examples]]
                    logger.debug(f"[Memory] Найдено {len(examples)} семантических few-shot примеров")
                    return examples
            except Exception as e:
                logger.warning(f"[Memory] Ошибка семантического поиска: {e}, переключение на fallback")

        # Fallback: пересечение слов
        input_words = set(input_text.lower().split())
        scored = []
        for ex in candidates:
            ex_words = set(ex["input"].lower().split())
            if not ex_words:
                similarity = 0
            else:
                intersection = len(input_words & ex_words)
                similarity = intersection / max(len(input_words), len(ex_words))
            scored.append((similarity, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [(sim, ex) for sim, ex in scored if sim >= similarity_threshold]
        if not filtered:
            filtered = scored[:max_examples]
        examples = [ex for _, ex in filtered[:max_examples]]
        logger.debug(f"[Memory] Найдено {len(examples)} fallback few-shot примеров")
        return examples

    def get_summary_few_shot_examples(self, input_text: str, domain: str = None,
                                      max_examples: int = 2, length_ratio: float = 0.3) -> List[Dict]:
        """
        Получение примеров для суммаризации – использует семантический поиск по содержанию.
        """
        if not self.summary_few_shot_examples:
            return []

        candidates = self.summary_few_shot_examples
        if domain:
            candidates = [ex for ex in candidates if ex.get("domain") == domain]

        model = self._get_embedding_model()
        if model is not None:
            try:
                input_emb = self._get_embedding(input_text)
                if input_emb is not None:
                    import numpy as np
                    scored = []
                    for ex in candidates:
                        ex_emb = self._get_embedding(ex["input"])
                        if ex_emb is None:
                            similarity = 0
                        else:
                            similarity = np.dot(input_emb, ex_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(ex_emb))
                        scored.append((similarity, ex))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    filtered = [(sim, ex) for sim, ex in scored if sim >= 0.5]  # порог чуть ниже
                    if not filtered:
                        filtered = scored[:max_examples]
                    examples = [ex for _, ex in filtered[:max_examples]]
                    logger.debug(f"[Memory] Найдено {len(examples)} семантических примеров суммаризации")
                    return examples
            except Exception as e:
                logger.warning(f"[Memory] Ошибка семантического поиска для суммаризации: {e}")

        # Fallback: оценка по длине
        input_len = len(input_text)
        scored = []
        for ex in candidates:
            ex_len = ex.get("char_count", 0)
            if ex_len == 0:
                ratio = 0
            else:
                ratio = min(input_len, ex_len) / max(input_len, ex_len)
            scored.append((ratio, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [(ratio, ex) for ratio, ex in scored if ratio >= (1 - length_ratio)]
        if not filtered:
            filtered = scored[:max_examples]
        examples = [ex for _, ex in filtered[:max_examples]]
        logger.debug(f"[Memory] Найдено {len(examples)} fallback примеров суммаризации")
        return examples

    # ========== Общая статистика ==========
    def get_memory_stats(self) -> Dict[str, Any]:
        prompt_stats = self.get_prompt_usage_stats()
        return {
            "history_size": len(self.correction_history),
            "summary_history_size": len(self.summary_history),
            "common_errors_count": len(self.common_errors),
            "success_patterns_count": len(self.success_patterns),
            "domains_count": len(self.domain_stats),
            "best_prompts_count": sum(len(v) for v in self.best_prompts.values()) if isinstance(self.best_prompts, dict) else 0,
            "best_summary_prompts_count": sum(len(v) for v in self.best_summary_prompts.values()) if isinstance(self.best_summary_prompts, dict) else 0,
            "prompt_cache_size": len(self.prompt_cache),
            "prompt_usage_stats": prompt_stats,
            "few_shot_examples_count": len(self.few_shot_examples),
            "summary_few_shot_examples_count": len(self.summary_few_shot_examples)
        }

    def clear_memory(self):
        self.correction_history = []
        self.summary_history = []
        self.common_errors = {}
        self.success_patterns = {}
        self.domain_stats = {}
        self.best_prompts = {}
        self.best_summary_prompts = {}
        self.prompt_cache = {}
        self.prompt_usage_stats = {}
        self.few_shot_examples = []
        self.summary_few_shot_examples = []
        self._embedding_cache = {}
        self._embedding_model = None
        self._save_memory()
        logger.info("[AgentMemory] Память очищена")