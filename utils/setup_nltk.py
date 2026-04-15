#!/usr/bin/env python3
"""
Скрипт для загрузки NLTK ресурсов с обходом SSL ошибок
Запустить один раз перед первым запуском системы
"""

import ssl
import nltk
import os
import sys


def download_nltk_resources():
    """Загрузка всех необходимых NLTK ресурсов с обходом SSL"""

    # Отключаем SSL проверку для загрузки
    ssl._create_default_https_context = ssl._create_unverified_context

    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet'),
    ]

    print("=" * 80)
    print("📦 Загрузка NLTK ресурсов (SSL отключен)...")
    print("=" * 80)

    # Определяем директорию для загрузки
    download_dir = os.path.expanduser("~/nltk_data")
    print(f"\n📂 Директория загрузки: {download_dir}")

    # Создаём директорию если не существует
    os.makedirs(download_dir, exist_ok=True)

    for path, name in resources:
        print(f"\n⏳ Загрузка {name}...")

        # Проверяем есть ли уже
        try:
            nltk.data.find(path)
            print(f"  ✅ {name} уже загружен")
            continue
        except LookupError:
            pass

        # Пробуем загрузить
        try:
            nltk.download(name, download_dir=download_dir, quiet=False, halt_on_error=False)

            # Проверяем что загрузилось
            try:
                nltk.data.find(path)
                print(f"  ✅ {name} успешно загружен")
            except:
                print(f"  ⚠️  {name} загружен но не найден")

        except Exception as e:
            print(f"  ❌ Не удалось загрузить {name}: {e}")

    print("\n" + "=" * 80)
    print("✅ Загрузка NLTK ресурсов завершена!")
    print("=" * 80)
    print("\n💡 Теперь можно запускать: python main.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    download_nltk_resources()
