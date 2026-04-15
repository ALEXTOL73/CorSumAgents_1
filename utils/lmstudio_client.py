#!/usr/bin/env python3
"""
Клиент для взаимодействия с LM Studio API
Версия 5.0.0 - Расширенное кэширование для всех вызовов + персистентное хранение
Особенности:
- Кэширование всех типов запросов (коррекция, суммаризация, G-Eval, LLM-Judge)
- Персистентное хранение кэша на диске (между запусками)
- Параллельное выполнение запросов через ThreadPoolExecutor
- Rate limiting для предотвращения перегрузки сервера
- Обработка всех ошибок LM Studio (400, channel, compute)
- Авто-определение URL и модели
- Улучшенный парсинг JSON с fallback для обрезанных ответов
"""
import requests
import json
import re
import time
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from utils.logger import setup_logger
from config import LM_STUDIO_URL, MODEL_NAME, REQUEST_TIMEOUT, LM_STUDIO_MIN_REQUEST_INTERVAL

logger = setup_logger("LMStudioClient")


class LMStudioClient:
    """
    Клиент для отправки запросов к локальному LM Studio серверу

    Функции:
    - Параллельная генерация нескольких вариантов
    - Кэширование ответов для повторяющихся запросов (всех типов)
    - Персистентное хранение кэша на диске
    - Rate limiting для предотвращения перегрузки
    - Обработка ошибок (400, channel, compute)
    - Авто-определение URL и модели
    - Улучшенный парсинг JSON с fallback
    """

    # Singleton instance
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton паттерн для оптимизации памяти"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_url: str = None, model: str = None, cache_enabled: bool = True):
        """
        Инициализация клиента

        Args:
            base_url: URL LM Studio сервера
            model: Имя модели
            cache_enabled: Включить кэширование
        """
        if self._initialized:
            return

        self.base_url = base_url or LM_STUDIO_URL
        self.model = model or self._auto_detect_model() or MODEL_NAME
        self.request_count = 0
        self.cache_hits = 0
        self._last_request_time = 0
        self._min_request_interval = LM_STUDIO_MIN_REQUEST_INTERVAL

        # Кэш для запросов (ключ -> ответ)
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, str] = {}
        self.cache_lock = Lock()

        # Персистентное хранилище кэша на диске
        self.cache_file = Path.home() / ".cache" / "cor_sum_agents" / "llm_cache.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache_from_disk()

        # ThreadPoolExecutor для параллельного выполнения
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Счётчик channel errors
        self._consecutive_channel_errors = 0

        # Авто-определение URL если не указан
        if base_url is None:
            self.base_url = self._auto_detect_url()

        self._initialized = True
        logger.info(f"[{self.model}] Клиент инициализирован: url={self.base_url}")
        logger.info(f"[{self.model}] Кэширование: {'включено' if cache_enabled else 'отключено'}")
        logger.info(f"[{self.model}] Персистентный кэш: {self.cache_file} (размер {len(self.cache)})")

    def _load_cache_from_disk(self):
        """Загрузка кэша с диска при запуске"""
        if not self.cache_enabled:
            return
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Ограничиваем размер загружаемого кэша
                    if len(loaded) > 2000:
                        loaded = dict(list(loaded.items())[-2000:])
                    self.cache = loaded
                    logger.info(f"[{self.model}] Загружено {len(self.cache)} записей из персистентного кэша")
            else:
                logger.debug(f"[{self.model}] Файл кэша не найден, создадим при первом сохранении")
        except Exception as e:
            logger.warning(f"[{self.model}] Ошибка загрузки кэша с диска: {e}")

    def _save_cache_to_disk(self):
        """Сохранение кэша на диск"""
        if not self.cache_enabled:
            return
        try:
            with self.cache_lock:
                # Ограничиваем размер кэша перед сохранением
                if len(self.cache) > 2000:
                    # Оставляем только последние 2000 записей (по времени добавления не храним, просто по ключам)
                    items = list(self.cache.items())
                    self.cache = dict(items[-2000:])
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                logger.debug(f"[{self.model}] Кэш сохранён на диск ({len(self.cache)} записей)")
        except Exception as e:
            logger.warning(f"[{self.model}] Ошибка сохранения кэша на диск: {e}")

    def _generate_cache_key(self, prompt: str, temperature: float, system_prompt: str = "", model: str = None, max_tokens: int = None) -> str:
        """Генерация уникального ключа для кэша (учитывает все параметры)"""
        # Включаем в ключ все параметры, влияющие на ответ
        key_string = f"{prompt}|{temperature}|{system_prompt}|{model or 'default'}|{max_tokens or 'default'}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Получение ответа из кэша (сначала из оперативной памяти, потом с диска - но диск уже загружен)"""
        if not self.cache_enabled:
            return None
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.debug(f"[{self.model}] Cache hit: {self.cache_hits}/{self.request_count}")
                return self.cache[cache_key]
        return None

    def _save_to_cache(self, cache_key: str, response: str):
        """Сохранение ответа в кэш (оперативная память + периодически на диск)"""
        if not self.cache_enabled:
            return
        with self.cache_lock:
            self.cache[cache_key] = response
            # Ограничиваем размер оперативного кэша
            if len(self.cache) > 2000:
                # Удаляем самые старые (простейшая стратегия - удалить первые 100)
                keys_to_remove = list(self.cache.keys())[:100]
                for key in keys_to_remove:
                    del self.cache[key]
            logger.debug(f"[{self.model}] Cache saved. Total cache size: {len(self.cache)}")
        # Сохраняем на диск асинхронно (но для простоты синхронно, но не каждый раз)
        # Для оптимизации можно сохранять каждые N запросов, но пока сохраняем всегда
        self._save_cache_to_disk()

    def _auto_detect_model(self) -> Optional[str]:
        """Авто-определение загруженной модели в LM Studio"""
        try:
            models_url = self.base_url.replace("/chat/completions", "/models")
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    model_id = models[0].get("id")
                    logger.info(f"[{model_id}] Авто-определение модели")
                    return model_id
        except Exception as e:
            logger.debug(f"[{self.model}] Не удалось авто-определить модель: {e}")
        return None

    def _auto_detect_url(self) -> str:
        """Авто-определение работающего URL LM Studio"""
        candidates = [
            "http://localhost:1234/v1/chat/completions",
            "http://127.0.0.1:1234/v1/chat/completions",
            "http://localhost:1234/v1",
            "http://127.0.0.1:1234/v1",
        ]
        for url in candidates:
            try:
                base = url.replace("/chat/completions", "").replace("/v1", "")
                test_url = f"{base}/v1/models"
                response = requests.get(test_url, timeout=3)
                if response.status_code == 200:
                    logger.info(f"[{self.model}] Найден работающий URL: {url}")
                    return url
            except:
                continue
        logger.warning(f"[{self.model}] Не удалось авто-определить URL, используем дефолт")
        return "http://localhost:1234/v1/chat/completions"

    def _check_port_open(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Проверка открыт ли порт"""
        import socket
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, socket.error, OSError):
            return False

    def _validate_payload(self, payload: Dict) -> bool:
        """Валидация payload перед отправкой"""
        required_fields = ["model", "messages"]
        for field in required_fields:
            if field not in payload:
                logger.error(f"[{self.model}] Отсутствует обязательное поле: {field}")
                return False

        if not payload["model"] or not isinstance(payload["model"], str):
            logger.error(f"[{self.model}] Неверное значение model: {payload.get('model')}")
            return False

        if not payload["messages"] or not isinstance(payload["messages"], list):
            logger.error(f"[{self.model}] Неверное значение messages: {payload.get('messages')}")
            return False

        for i, msg in enumerate(payload["messages"]):
            if not isinstance(msg, dict):
                logger.error(f"[{self.model}] Сообщение {i} не является словарём")
                return False
            if "role" not in msg or "content" not in msg:
                logger.error(f"[{self.model}] Сообщение {i} не имеет role или content")
                return False
            if msg["role"] not in ["system", "user", "assistant"]:
                logger.error(f"[{self.model}] Неверная роль в сообщении {i}: {msg.get('role')}")
                return False
            if not isinstance(msg.get("content"), str):
                logger.error(f"[{self.model}] Content в сообщении {i} не является строкой")
                return False

        if "temperature" in payload:
            temp = payload["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 1:
                logger.warning(f"[{self.model}] Температура вне диапазона [0,1]: {temp}")
                payload["temperature"] = max(0.0, min(1.0, float(temp)))

        if "max_tokens" in payload:
            tokens = payload["max_tokens"]
            if not isinstance(tokens, int) or tokens < 1:
                logger.warning(f"[{self.model}] Неверное max_tokens: {tokens}")
                payload["max_tokens"] = 1024

        return True

    def _rate_limit(self):
        """Ограничение частоты запросов"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _handle_channel_error(self) -> bool:
        """Обработка channel error"""
        self._consecutive_channel_errors += 1
        logger.error(f"[{self.model}] ❌ Channel error (подряд: {self._consecutive_channel_errors})")

        if self._consecutive_channel_errors >= 3:
            logger.error(f"[{self.model}] 💥 Слишком много channel errors!")
            logger.error(f"[{self.model}] 💡 РЕКОМЕНДАЦИИ:")
            logger.error(f"[{self.model}]    1. Перезагрузите модель в LM Studio")
            logger.error(f"[{self.model}]    2. Нажмите 'Restart Server' в интерфейсе")
            logger.error(f"[{self.model}]    3. Полностью перезапустите LM Studio")
            return False

        wait_time = min(2.0 ** self._consecutive_channel_errors, 10.0)
        logger.info(f"[{self.model}] ⏳ Ожидание {wait_time:.1f}с перед повтором...")
        time.sleep(wait_time)

        if not self._quick_health_check():
            logger.error(f"[{self.model}] ❌ Сервер не отвечает после ожидания")
            return False

        logger.info(f"[{self.model}] 🔄 Пробуем повторный запрос...")
        return True

    def _quick_health_check(self) -> bool:
        """Быстрая проверка что сервер жив"""
        try:
            import re
            match = re.search(r'http://([^:/]+):(\d+)', self.base_url)
            if match:
                host, port = match.groups()
                if not self._check_port_open(host, int(port), timeout=0.5):
                    return False
            test = {
                "model": self.model,
                "messages": [{"role": "user", "content": "."}],
                "temperature": 0.1,
                "max_tokens": 1,
                "stream": False
            }
            resp = requests.post(self.base_url, json=test, timeout=5, headers={"Content-Type": "application/json"})
            return resp.status_code in [200, 400]
        except:
            return False

    def _send_request(self, payload: Dict, retry_count: int = 0, is_retry_after_compute: bool = False) -> Optional[Dict]:
        """Отправка запроса с обработкой всех ошибок"""
        max_retries = 3

        try:
            self._rate_limit()
            logger.debug(f"[{self.model}] Отправка: {json.dumps(payload, ensure_ascii=False)[:200]}...")

            response = requests.post(
                self.base_url,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 500 or response.status_code == 503:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "") or response.text
                    if "channel" in error_msg.lower() or "channel error" in error_msg.lower():
                        logger.error(f"[{self.model}] ❌ Channel error: {error_msg[:100]}")
                        if self._handle_channel_error():
                            if retry_count == 0:
                                self._consecutive_channel_errors = 0
                            return self._send_request(payload, retry_count + 1, is_retry_after_compute=True)
                        else:
                            return None
                except:
                    pass

            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "")
                    if "Compute error" in error_msg:
                        logger.error(f"[{self.model}] ❌ Compute error — проверьте загрузку модели")
                        if retry_count < 1:
                            fixed = self._simplify_payload(payload)
                            if fixed != payload:
                                time.sleep(1.0)
                                return self._send_request(fixed, retry_count + 1)
                        return None
                except:
                    pass
                logger.error(f"[{self.model}] HTTP 400: {response.text[:200]}")
                if retry_count < max_retries:
                    simplified = self._simplify_payload(payload)
                    if simplified != payload:
                        time.sleep(0.5)
                        return self._send_request(simplified, retry_count + 1)
                return None

            if response.status_code == 404:
                logger.error(f"[{self.model}] HTTP 404 — неверный эндпоинт: {self.base_url}")
                return None

            if response.status_code == 503:
                logger.error(f"[{self.model}] HTTP 503 — сервис недоступен")
                if retry_count < max_retries:
                    time.sleep(2.0)
                    return self._send_request(payload, retry_count + 1)
                return None

            response.raise_for_status()

            if self._consecutive_channel_errors > 0:
                logger.info(f"[{self.model}] ✅ Запрос успешен, сброс счётчика channel error")
                self._consecutive_channel_errors = 0

            return response.json()

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{self.model}] Ошибка соединения: {e}")
            import re
            match = re.search(r'http://([^:/]+):(\d+)', self.base_url)
            if match:
                host, port = match.groups()
                if not self._check_port_open(host, int(port)):
                    logger.error(f"[{self.model}] ❌ Порт {port} на {host} НЕ ОТКРЫТ!")
            if retry_count < max_retries:
                logger.info(f"[{self.model}] Повтор соединения {retry_count + 1}/{max_retries}")
                time.sleep(1.0)
                return self._send_request(payload, retry_count + 1)
            return None

        except requests.exceptions.Timeout as e:
            logger.error(f"[{self.model}] Таймаут: {e}")
            if retry_count < max_retries:
                logger.info(f"[{self.model}] Повтор после таймаута {retry_count + 1}/{max_retries}")
                time.sleep(0.5)
                return self._send_request(payload, retry_count + 1)
            return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"[{self.model}] HTTP ошибка: {e}, Status: {response.status_code}")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"[{self.model}] Ошибка парсинга JSON: {e}")
            return None

        except Exception as e:
            logger.error(f"[{self.model}] Неизвестная ошибка: {type(e).__name__}: {e}")
            return None

    def _simplify_payload(self, payload: Dict) -> Dict:
        """Упрощение payload для совместимости с LM Studio"""
        simplified = payload.copy()
        for field in ["stop", "presence_penalty", "frequency_penalty", "top_p", "top_k", "logprobs"]:
            simplified.pop(field, None)
        if "messages" in simplified:
            simplified["messages"] = [
                {"role": msg.get("role", "user"), "content": str(msg.get("content", ""))[:8000]}
                for msg in simplified["messages"]
            ]
        return simplified

    def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            system_prompt: str = "",
            max_tokens: int = 1024,
            use_cache: bool = True,
            model: str = None
    ) -> str:
        """
        Генерация текста через LM Studio с кэшированием (универсальный метод)

        Args:
            prompt: Пользовательский промпт
            temperature: Температура генерации (0.0-1.0)
            system_prompt: Системный промпт
            max_tokens: Максимальное количество токенов
            use_cache: Использовать ли кэш
            model: Имя модели (опционально, по умолчанию self.model)

        Returns:
            Сгенерированный текст
        """
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            logger.warning(f"[{self.model}] Пустой промпт")
            return ""

        model_to_use = model or self.model

        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(prompt, temperature, system_prompt, model_to_use, max_tokens)
            cached_response = self._get_from_cache(cache_key)
            if cached_response is not None:
                return cached_response

        logger.info(f"[{model_to_use}] Запрос к модели")

        messages = []
        if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload = {
            "model": str(model_to_use).strip() if model_to_use else "local-model",
            "messages": messages,
            "temperature": float(max(0.0, min(1.0, temperature))),
            "max_tokens": int(max(1, min(4096, max_tokens))),
            "stream": False
        }

        if not self._validate_payload(payload):
            logger.error(f"[{model_to_use}] Payload не прошёл валидацию")
            return ""

        result = self._send_request(payload)

        if result is None:
            logger.error(f"[{model_to_use}] Не удалось получить ответ")
            return ""

        try:
            if 'choices' not in result or not result['choices']:
                logger.error(f"[{model_to_use}] Неверная структура ответа: {result}")
                return ""

            content = result['choices'][0].get('message', {}).get('content', '')

            if not content:
                logger.warning(f"[{model_to_use}] Пустой контент")
                return ""

            self.request_count += 1
            logger.info(
                f"[{model_to_use}] Ответ получен. Всего запросов: {self.request_count}, Cache hits: {self.cache_hits}")

            if use_cache and self.cache_enabled:
                cache_key = self._generate_cache_key(prompt, temperature, system_prompt, model_to_use, max_tokens)
                self._save_to_cache(cache_key, content)

            return content.strip()

        except Exception as e:
            logger.error(f"[{model_to_use}] Ошибка обработки ответа: {e}")
            return ""

    def generate_parallel(
            self,
            prompts: List[str],
            temperatures: List[float] = None,
            system_prompt: str = "",
            max_tokens: int = 1024,
            use_cache: bool = True,
            model: str = None
    ) -> List[str]:
        """
        Параллельная генерация нескольких вариантов

        Args:
            prompts: Список промптов
            temperatures: Список температур
            system_prompt: Системный промпт
            max_tokens: Максимальное количество токенов
            use_cache: Использовать ли кэш
            model: Имя модели (опционально, по умолчанию self.model)

        Returns:
            Список сгенерированных текстов
        """
        if not prompts:
            logger.warning(f"[{self.model}] Пустой список промптов")
            return []

        n = len(prompts)

        if temperatures is None:
            temperatures = [0.7] * n
        elif len(temperatures) != n:
            logger.warning(f"[{self.model}] Несовпадение количества промптов и температур")
            temperatures = temperatures[:n] if len(temperatures) >= n else temperatures + [0.7] * (
                        n - len(temperatures))

        model_to_use = model or self.model

        logger.info(f"[{model_to_use}] Параллельная генерация {n} вариантов")

        if self._consecutive_channel_errors > 0:
            logger.warning(f"[{model_to_use}] Channel error detected, используем последовательное выполнение")
            return self._generate_sequential(prompts, temperatures, system_prompt, max_tokens, use_cache, model_to_use)

        try:
            results = self._generate_parallel_safe(prompts, temperatures, system_prompt, max_tokens, use_cache,
                                                   model_to_use)
            successful = len([r for r in results if r])
            logger.info(f"[{model_to_use}] Параллельная генерация: {successful}/{n} успешных")

            if successful < n * 0.5 and n > 1:
                logger.warning(
                    f"[{model_to_use}] Мало успешных параллельных запросов, переключаемся на последовательное")
                return self._generate_sequential(prompts, temperatures, system_prompt, max_tokens, use_cache,
                                                 model_to_use)

            return results

        except Exception as e:
            logger.warning(f"[{self.model}] Ошибка параллельной генерации: {e}, переключаемся на последовательное")
            return self._generate_sequential(prompts, temperatures, system_prompt, max_tokens, use_cache, model_to_use)

    def _generate_parallel_safe(self, prompts: List[str], temperatures: List[float],
                                system_prompt: str, max_tokens: int, use_cache: bool, model: str = None) -> List[str]:
        """Безопасное параллельное выполнение с ограничением конкуренции"""
        results = [None] * len(prompts)

        def generate_with_index(idx, prompt, temp):
            try:
                result = self.generate(
                    prompt,
                    temperature=temp,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    use_cache=use_cache,
                    model=model
                )
                return idx, result
            except Exception as e:
                logger.error(f"[{self.model}] Ошибка в потоке {idx}: {e}")
                return idx, ""

        max_workers = 1 if self._consecutive_channel_errors > 0 else min(3, len(prompts))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(generate_with_index, i, prompt, temp)
                for i, (prompt, temp) in enumerate(zip(prompts, temperatures))
            ]

            for future in as_completed(futures):
                try:
                    idx, result = future.result(timeout=REQUEST_TIMEOUT)
                    results[idx] = result
                except Exception as e:
                    logger.error(f"[{self.model}] Ошибка получения результата: {e}")

        return results

    def _generate_sequential(self, prompts: List[str], temperatures: List[float],
                             system_prompt: str, max_tokens: int, use_cache: bool, model: str = None) -> List[str]:
        """Последовательная генерация (фоллбэк)"""
        results = []

        for i, (prompt, temp) in enumerate(zip(prompts, temperatures)):
            logger.debug(f"[{self.model}] Последовательная генерация {i + 1}/{len(prompts)} (temp={temp})")
            result = self.generate(
                prompt,
                temperature=temp,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                use_cache=use_cache,
                model=model
            )
            results.append(result)

            if i < len(prompts) - 1:
                time.sleep(0.3)

        return results

    def generate_json(
            self,
            prompt: str,
            temperature: float = 0.2,
            system_prompt: str = "Отвечай ТОЛЬКО в формате JSON",
            use_cache: bool = True,
            model: str = None,
            max_tokens: int = 1024  # ✅ УВЕЛИЧЕНО для полных ответов
    ) -> Dict[str, Any]:
        """
        Генерация JSON ответа с улучшенным парсингом (использует кэш)

        Args:
            prompt: Пользовательский промпт
            temperature: Температура (низкая для стабильности)
            system_prompt: Системный промпт
            use_cache: Использовать ли кэш
            model: Имя модели
            max_tokens: Максимальное количество токенов (увеличено для JSON)

        Returns:
            Словарь с распарсенным JSON
        """
        response = self.generate(
            prompt,
            temperature,
            system_prompt,
            max_tokens=max_tokens,
            use_cache=use_cache,
            model=model
        )

        if not response:
            return {"error": "empty response", "raw": ""}

        # Улучшенный парсинг JSON
        parsed = self._parse_json_response(response)

        if "error" in parsed:
            logger.warning(f"[{self.model}] Ошибка парсинга JSON: {parsed.get('error', 'Unknown')}")
            logger.warning(f"[{self.model}] Сырой ответ: {response[:500]}")

        return parsed

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Улучшенный парсинг JSON с fallback для обрезанных ответов
        """
        # Шаг 1: Удаляем markdown code blocks
        clean_response = response.replace("```json", "").replace("```", "").strip()

        # Шаг 2: Ищем JSON между фигурными скобками
        json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)

        if not json_match:
            logger.warning(f"[{self.model}] Не найден JSON блок в ответе")
            return {"error": "no json block found", "raw": response}

        json_str = json_match.group()

        # Шаг 3: Пробуем распарсить полный JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"[{self.model}] Ошибка парсинга полного JSON: {e}")

        # Шаг 4: Пытаемся восстановить обрезанный JSON
        try:
            fixed_json = json_str
            open_braces = fixed_json.count('{')
            close_braces = fixed_json.count('}')
            open_brackets = fixed_json.count('[')
            close_brackets = fixed_json.count(']')

            if open_braces > close_braces:
                fixed_json += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                fixed_json += ']' * (open_brackets - close_brackets)

            return json.loads(fixed_json)

        except json.JSONDecodeError as e:
            logger.debug(f"[{self.model}] Ошибка парсинга восстановленного JSON: {e}")

        # Шаг 5: Извлекаем поля вручную через regex
        result = {}
        num_pattern = r'"(\w+)":\s*([0-9.]+)'
        for match in re.finditer(num_pattern, clean_response):
            key = match.group(1)
            try:
                result[key] = float(match.group(2))
            except ValueError:
                pass

        str_pattern = r'"(\w+)":\s*"([^"]*)"'
        for match in re.finditer(str_pattern, clean_response):
            key = match.group(1)
            result[key] = match.group(2)

        # Особая обработка explanation (может содержать кавычки)
        expl_pattern = r'"explanation":\s*"([^"]*(?:"[^"]*)*)"'
        expl_match = re.search(expl_pattern, clean_response, re.DOTALL)
        if expl_match:
            result["explanation"] = expl_match.group(1)

        if "score" in result:
            if "explanation" not in result:
                result["explanation"] = "Объяснение обрезано или недоступно"
            return result

        # Полный fallback
        logger.warning(f"[{self.model}] Не удалось извлечь JSON, используем значения по умолчанию")
        return {
            "error": "json parse failed",
            "raw": response,
            "score": 5,
            "explanation": "Не удалось получить оценку от судьи"
        }

    def health_check(self) -> bool:
        """Улучшенная проверка доступности сервера"""
        import re
        match = re.search(r'http://([^:/]+):(\d+)', self.base_url)
        if not match:
            logger.error(f"[{self.model}] Не удалось извлечь хост:порт из {self.base_url}")
            return False

        host, port = match.groups()
        port = int(port)

        logger.debug(f"[{self.model}] Health check: проверка {host}:{port}")

        if not self._check_port_open(host, port):
            logger.error(f"[{self.model}] ❌ Порт {port} на {host} НЕ ОТКРЫТ")
            return False

        logger.debug(f"[{self.model}] ✅ Порт {port} открыт")

        try:
            models_url = self.base_url.replace("/chat/completions", "/models")
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                logger.debug(f"[{self.model}] ✅ Эндпоинт /v1/models работает")
            else:
                logger.warning(f"[{self.model}] ⚠️  Эндпоинт /v1/models вернул статус {response.status_code}")
        except Exception as e:
            logger.warning(f"[{self.model}] ⚠️  Ошибка при проверке /v1/models: {e}")

        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "OK"}],
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False
            }

            response = requests.post(
                self.base_url,
                json=test_payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                logger.info(f"[{self.model}] ✅ Health check пройден: сервер отвечает")
                self._consecutive_channel_errors = 0
                return True
            else:
                error = response.text[:200]
                logger.error(f"[{self.model}] ❌ Health check failed: {response.status_code} - {error}")
                if "Compute error" in error or "channel" in error.lower():
                    logger.error(f"[{self.model}] 💡 Compute/channel error — проверьте что модель загружена!")
                return False

        except requests.exceptions.ConnectionError:
            logger.error(f"[{self.model}] ❌ Не удалось подключиться к {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"[{self.model}] ❌ Ошибка health check: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        with self.cache_lock:
            return {
                "cache_size": len(self.cache),
                "cache_hits": self.cache_hits,
                "total_requests": self.request_count,
                "hit_rate": (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0,
                "persistent_file": str(self.cache_file),
                "persistent_exists": self.cache_file.exists()
            }

    def clear_cache(self):
        """Очистка кэша (оперативного и на диске)"""
        with self.cache_lock:
            self.cache.clear()
            self.cache_hits = 0
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"[{self.model}] Файл кэша удалён: {self.cache_file}")
        except Exception as e:
            logger.warning(f"[{self.model}] Не удалось удалить файл кэша: {e}")
        logger.info(f"[{self.model}] Кэш очищен")

    def print_connection_info(self):
        """Вывод подробной информации о подключении для отладки"""
        import re
        print("\n" + "=" * 80)
        print("  🔍 ИНФОРМАЦИЯ О ПОДКЛЮЧЕНИИ К LM STUDIO")
        print("=" * 80)
        print(f"  Base URL: {self.base_url}")
        print(f"  Model: {self.model}")
        print(f"  Timeout: {REQUEST_TIMEOUT}s")
        print(f"  Min request interval: {self._min_request_interval}s")
        print(f"  Cache enabled: {self.cache_enabled}")

        cache_stats = self.get_cache_stats()
        print(f"  Cache size (RAM): {cache_stats['cache_size']}")
        print(f"  Cache hits: {cache_stats['cache_hits']}")
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"  Persistent cache file: {cache_stats['persistent_file']}")
        print(f"  Persistent cache exists: {cache_stats['persistent_exists']}")

        match = re.search(r'http://([^:/]+):(\d+)', self.base_url)
        if match:
            host, port = match.groups()
            port = int(port)
            print(f"\n  📡 Проверка сети:")
            print(f"     Host: {host}")
            print(f"     Port: {port}")

            if self._check_port_open(host, port):
                print(f"     ✅ Порт {port} ОТКРЫТ")
            else:
                print(f"     ❌ Порт {port} ЗАКРЫТ")
                print(f"     💡 Попробуйте:")
                print(f"        • Запустить LM Studio Server")
                print(f"        • Проверить порт в настройках LM Studio")
                print(f"        • Отключить фаервол временно")
                print(f"        • Попробовать 127.0.0.1 вместо localhost")

        if self._consecutive_channel_errors > 0:
            print(f"\n  ⚠️  WARNING: {self._consecutive_channel_errors} consecutive channel errors!")
            print(f"  💡 RECOMMENDATION: Restart the model or LM Studio server")

        print("=" * 80 + "\n")

    def reset_channel_error_counter(self):
        """Сброс счётчика channel error (после ручной перезагрузки)"""
        old = self._consecutive_channel_errors
        self._consecutive_channel_errors = 0
        logger.info(f"[{self.model}] Счётчик channel error сброшен: {old} → 0")

    def close(self):
        """Закрытие клиента и освобождение ресурсов, сохранение кэша"""
        self._save_cache_to_disk()
        self.executor.shutdown(wait=False)
        logger.info(f"[{self.model}] Клиент закрыт, кэш сохранён")