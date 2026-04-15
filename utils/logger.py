"""
Модуль логирования для системы агентов
Версия 4.3 - Логи в папках по датам (DDMMYYYY)

Особенности:
- Логи агентов в папках по датам (например, agent_logs/29032026/)
- Формат имени файла: <agent>_<YYYYMMDD>.log
- Ротация логов
- Консольный вывод
"""
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os
import sys

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS, LOG_CONFIG


class DateLogger:
    """
    Утилиты для логирования по датам
    v4.3: Логи в папках по датам (формат DDMMYYYY)
    """

    @staticmethod
    def get_date_folder_name() -> str:
        """
        Получение имени папки для текущей даты

        Returns:
            Имя папки в формате DDMMYYYY (например, 29032026)
        """
        return datetime.now().strftime("%d%m%Y")

    @staticmethod
    def get_date_log_folder() -> Path:
        """
        Получение пути к папке логов для текущей даты

        Returns:
            Путь к папке для текущей даты
        """
        date_folder_name = DateLogger.get_date_folder_name()
        date_folder = DIRS["agent_logs"] / date_folder_name
        date_folder.mkdir(parents=True, exist_ok=True)
        return date_folder

    @staticmethod
    def get_agent_log_filename(agent_name: str) -> str:
        """
        Получение имени файла лога для агента

        Args:
            agent_name: Имя агента

        Returns:
            Имя файла в формате <agent>_<YYYYMMDD>.log
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{agent_name}_{timestamp}.log"

    @staticmethod
    def get_agent_log_path(agent_name: str) -> Path:
        """
        Получение полного пути к логу агента

        Args:
            agent_name: Имя агента

        Returns:
            Полный путь к файлу лога
        """
        date_folder = DateLogger.get_date_log_folder()
        filename = DateLogger.get_agent_log_filename(agent_name)
        return date_folder / filename

    @staticmethod
    def get_system_log_path() -> Path:
        """
        Получение пути к системному логу

        Returns:
            Путь к файлу системного лога
        """
        return DIRS["logs"] / "system.log"


def setup_logger(name: str = "AgentSystem", agent_name: str = None) -> logging.Logger:
    """
    Настройка логгера для агента или системы

    v4.3: Логи агентов в папках по датам

    Args:
        name: Имя логгера
        agent_name: Имя агента (для разделения логов)

    Returns:
        Настроенный logger объект
    """
    logger = logging.getLogger(name)

    # Если логгер уже настроен, возвращаем его
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_CONFIG["level"]))

    # Форматтер
    formatter = logging.Formatter(LOG_CONFIG["format"])

    # Console Handler
    if LOG_CONFIG.get("console_output", True):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File Handler - общий системный лог
    if LOG_CONFIG.get("file_output", True):
        # Убеждаемся что директория логов существует
        DIRS["logs"].mkdir(parents=True, exist_ok=True)

        system_log_path = DateLogger.get_system_log_path()
        fh = RotatingFileHandler(
            system_log_path,
            maxBytes=LOG_CONFIG.get("max_bytes", 10485760),
            backupCount=LOG_CONFIG.get("backup_count", 5),
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Agent-specific log file в папке по дате (v4.3)
    if agent_name and LOG_CONFIG.get("agent_logs", True):
        try:
            # Получаем путь к логу в папке по дате
            agent_log_path = DateLogger.get_agent_log_path(agent_name)

            agent_fh = RotatingFileHandler(
                agent_log_path,
                maxBytes=LOG_CONFIG.get("max_bytes", 10485760),
                backupCount=LOG_CONFIG.get("backup_count", 5),
                encoding='utf-8'
            )
            agent_fh.setLevel(logging.DEBUG)
            agent_fh.setFormatter(formatter)
            logger.addHandler(agent_fh)

            # Логгируем путь к файлу для отладки
            logger.debug(f"[Logger] Agent log file: {agent_log_path}")

        except Exception as e:
            logger.error(f"[Logger] Ошибка создания agent log handler: {e}")

    return logger


def get_current_date_log_folder() -> Path:
    """
    Получение текущей папки логов по дате

    Returns:
        Путь к папке логов для текущей даты
    """
    return DateLogger.get_date_log_folder()


def list_agent_logs_for_today() -> list:
    """
    Получение списка файлов логов за сегодня

    Returns:
        Список путей к файлам логов
    """
    date_folder = DateLogger.get_date_log_folder()
    return list(date_folder.glob("*.log"))


def print_log_folder_info():
    """
    Вывод информации о папке логов
    """
    date_folder = DateLogger.get_date_log_folder()
    print(f"\n📁 Папка логов: {date_folder}")
    print(f"📄 Файлов логов: {len(list(date_folder.glob('*.log')))}")

    log_files = list(date_folder.glob("*.log"))
    for log_file in log_files[:10]:
        print(f"   └─ {log_file.name}")
    if len(log_files) > 10:
        print(f"   └─ ... и ещё {len(log_files) - 10} файлов")
