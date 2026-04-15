"""
Утилиты для цветного вывода в консоль
Версия 4.9.2 - Чёрный=инфо, Красный=ошибка, Зелёный=метрики

Использование:
    from utils.colors import cprint, Color

    cprint("Обычный текст", Color.BLACK)
    cprint("Ошибка!", Color.RED)
    cprint("Метрика: 0.95", Color.GREEN)
"""

import sys
import os


# ✅ ANSI коды цветов
class Color:
    """ANSI коды цветов для терминала"""
    # Основные цвета
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Яркие цвета
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Сброс
    RESET = "\033[0m"

    # ✅ Алиасы для удобства (v4.9.2)
    INFO = BLACK  # Обычная информация - чёрный
    ERROR = RED  # Ошибки - красный
    WARNING = RED  # Предупреждения - красный
    METRIC = GREEN  # Метрики - зелёный
    SUCCESS = GREEN  # Успех - зелёный


# ✅ Проверка поддержки цветов
def supports_color():
    """Проверка поддерживает ли терминал цвета"""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    # Windows 10+ поддерживает ANSI через colorama
    if sys.platform == "win32":
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            return False
    return True


# ✅ Глобальная проверка поддержки цветов
_COLOR_SUPPORTED = supports_color()


def _apply_color(text: str, color: str) -> str:
    """Применение цвета к тексту"""
    if not _COLOR_SUPPORTED:
        return text
    return f"{color}{text}{Color.RESET}"


def cprint(text: str, color: str = Color.BLACK, end: str = "\n", file=None):
    """
    Цветной вывод в консоль

    Args:
        text: Текст для вывода
        color: ANSI код цвета из класса Color
        end: Символ конца строки
        file: Файл для вывода (по умолчанию sys.stdout)
    """
    if file is None:
        file = sys.stdout
    colored_text = _apply_color(text, color)
    print(colored_text, end=end, file=file)


def cformat(text: str, color: str = Color.BLACK) -> str:
    """
    Форматирование текста с цветом (для использования в print/f-string)

    Args:
        text: Текст для форматирования
        color: ANSI код цвета

    Returns:
        Отформатированный текст с цветом
    """
    return _apply_color(text, color)


# ✅ Удобные функции для частых случаев (v4.9.2)
def print_info(text: str, **kwargs):
    """Вывод информации чёрным цветом"""
    cprint(text, Color.INFO, **kwargs)


def print_error(text: str, **kwargs):
    """Вывод ошибки красным цветом"""
    cprint(text, Color.ERROR, **kwargs)


def print_warning(text: str, **kwargs):
    """Вывод предупреждения красным цветом"""
    cprint(text, Color.WARNING, **kwargs)


def print_metric(text: str, **kwargs):
    """Вывод метрики зелёным цветом"""
    cprint(text, Color.METRIC, **kwargs)


def print_success(text: str, **kwargs):
    """Вывод успеха зелёным цветом"""
    cprint(text, Color.SUCCESS, **kwargs)


def print_normal(text: str, **kwargs):
    """Вывод обычного текста чёрным цветом"""
    cprint(text, Color.BLACK, **kwargs)


# ✅ Инициализация для Windows
def init_colors():
    """Инициализация цветного вывода (для Windows)"""
    if sys.platform == "win32" and _COLOR_SUPPORTED:
        try:
            import colorama
            colorama.init()
        except ImportError:
            pass