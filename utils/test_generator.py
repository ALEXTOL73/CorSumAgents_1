#!/usr/bin/env python3
"""
Генератор тестовых данных для системы CorSumAgentsAI
Версия 2.1 - Добавлен метод get_test_cases()

Автор: CorSumAgentsAI Team
Дата: 2025
"""

from typing import List, Dict, Any
from utils.logger import setup_logger
from config import BASE_DIR

logger = setup_logger("TestGenerator")


class TestGenerator:
    """
    Генерация тестовых кейсов для коррекции и суммаризации
    
    Предоставляет методы для получения:
    - Тестов коррекции (русский/английский)
    - Тестов суммаризации (русский/английский)
    - Комбинированных тестов
    - Всех тестов через get_test_cases()
    """
    
    @staticmethod
    def get_test_cases() -> List[Dict[str, Any]]:
        """
        Получение всех тестовых кейсов системы
        
        Returns:
            Список всех тестовых кейсов с путями к файлам
        """
        logger.info("[TestGenerator] Загрузка всех тестовых кейсов")
        
        return [
            # =================================================================
            # РУССКИЕ ТЕСТЫ - КОРРЕКЦИЯ + СУММАРИЗАЦИЯ
            # =================================================================
            {
                "id": "ru_news_001",
                "task_type": "combined",
                "language": "ru",
                "domain": "news",
                "input_file": "data/Incorrect_texts/ru_news_001.txt",
                "reference_text_file": "data/etalon_texts/ru_news_001.txt",
                "reference_summary_file": "data/etalon_summaries/ru_news_001.txt",
                "expected_metrics": {
                    "min_wer_improvement": 0.30,
                    "min_lev_similarity_improvement": 0.05,
                    "min_g_eval_score": 8,
                    "min_p_umfd": 0.85
                }
            },
            {
                "id": "ru_tech_002",
                "task_type": "combined",
                "language": "ru",
                "domain": "tech",
                "input_file": "data/Incorrect_texts/ru_tech_002.txt",
                "reference_text_file": "data/etalon_texts/ru_tech_002.txt",
                "reference_summary_file": "data/etalon_summaries/ru_tech_002.txt",
                "expected_metrics": {
                    "min_wer_improvement": 0.25,
                    "min_lev_similarity_improvement": 0.04,
                    "min_g_eval_score": 7,
                    "min_p_umfd": 0.80
                }
            },
            {
                "id": "ru_casual_003",
                "task_type": "combined",
                "language": "ru",
                "domain": "casual",
                "input_file": "data/Incorrect_texts/ru_casual_003.txt",
                "reference_text_file": "data/etalon_texts/ru_casual_003.txt",
                "reference_summary_file": "data/etalon_summaries/ru_casual_003.txt",
                "expected_metrics": {
                    "min_wer_improvement": 0.15,
                    "min_lev_similarity_improvement": 0.03,
                    "min_g_eval_score": 7,
                    "min_p_umfd": 0.78
                }
            },
            # =================================================================
            # АНГЛИЙСКИЕ ТЕСТЫ - КОРРЕКЦИЯ + СУММАРИЗАЦИЯ
            # =================================================================
            {
                "id": "en_news_001",
                "task_type": "combined",
                "language": "en",
                "domain": "news",
                "input_file": "data/Incorrect_texts/en_news_001.txt",
                "reference_text_file": "data/etalon_texts/en_news_001.txt",
                "reference_summary_file": "data/etalon_summaries/en_news_001.txt",
                "expected_metrics": {
                    "min_wer_improvement": 0.35,
                    "min_lev_similarity_improvement": 0.06,
                    "min_g_eval_score": 8,
                    "min_p_umfd": 0.87
                }
            },
            {
                "id": "en_tech_002",
                "task_type": "combined",
                "language": "en",
                "domain": "tech",
                "input_file": "data/Incorrect_texts/en_tech_002.txt",
                "reference_text_file": "data/etalon_texts/en_tech_002.txt",
                "reference_summary_file": "data/etalon_summaries/en_tech_002.txt",
                "expected_metrics": {
                    "min_wer_improvement": 0.28,
                    "min_lev_similarity_improvement": 0.05,
                    "min_g_eval_score": 8,
                    "min_p_umfd": 0.85
                }
            },
            {
                "id": "en_casual_003",
                "task_type": "combined",
                "language": "en",
                "domain": "casual",
                "input_file": "data/Incorrect_texts/en_casual_003.txt",
                "reference_text_file": "data/etalon_texts/en_casual_003.txt",
                "reference_summary_file": "data/etalon_summaries/en_casual_003.txt",
                "expected_metrics": {
                    "min_wer_improvement": 0.20,
                    "min_lev_similarity_improvement": 0.04,
                    "min_g_eval_score": 7,
                    "min_p_umfd": 0.82
                }
            }
        ]
    
    @staticmethod
    def get_correction_test_cases() -> List[Dict[str, Any]]:
        """
        Получение тестовых кейсов только для коррекции
        
        Returns:
            Список тестовых кейсов для коррекции
        """
        all_cases = TestGenerator.get_test_cases()
        return [case for case in all_cases if case["task_type"] in ["correction", "combined"]]
    
    @staticmethod
    def get_summary_test_cases() -> List[Dict[str, Any]]:
        """
        Получение тестовых кейсов только для суммаризации
        
        Returns:
            Список тестовых кейсов для суммаризации
        """
        all_cases = TestGenerator.get_test_cases()
        return [case for case in all_cases if case["task_type"] in ["summary", "combined"]]
    
    @staticmethod
    def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
        """
        Получение конкретного тестового кейса по ID
        
        Args:
            test_id: Идентификатор теста (например, "ru_news_001")
        
        Returns:
            Словарь с данными теста
        
        Raises:
            ValueError: Если тест с таким ID не найден
        """
        all_cases = TestGenerator.get_test_cases()
        for case in all_cases:
            if case["id"] == test_id:
                logger.info(f"[TestGenerator] Найден тест: {test_id}")
                return case
        
        logger.error(f"[TestGenerator] Тест не найден: {test_id}")
        raise ValueError(f"Тест с ID {test_id} не найден. Доступные тесты: {[c['id'] for c in all_cases]}")
    
    @staticmethod
    def get_test_ids_by_language(language: str) -> List[str]:
        """
        Получение всех ID тестов для указанного языка
        
        Args:
            language: "ru" или "en"
        
        Returns:
            Список ID тестов
        """
        all_cases = TestGenerator.get_test_cases()
        ids = [case["id"] for case in all_cases if case["language"] == language]
        logger.info(f"[TestGenerator] Найдено {len(ids)} тестов для языка {language}")
        return ids
    
    @staticmethod
    def get_test_ids_by_domain(domain: str) -> List[str]:
        """
        Получение всех ID тестов для указанного домена
        
        Args:
            domain: "news", "tech", "casual"
        
        Returns:
            Список ID тестов
        """
        all_cases = TestGenerator.get_test_cases()
        ids = [case["id"] for case in all_cases if case["domain"] == domain]
        logger.info(f"[TestGenerator] Найдено {len(ids)} тестов для домена {domain}")
        return ids
    
    @staticmethod
    def generate_additional_tests(output_dir: str = "data/test_cases") -> int:
        """
        Генерация дополнительных тестов с искусственными ошибками
        
        Args:
            output_dir: Директория для сохранения тестов
        
        Returns:
            Количество сгенерированных тестов
        """
        import json
        import random
        from pathlib import Path
        
        logger.info("[TestGenerator] Генерация дополнительных тестов")
        
        # Типы ошибок для генерации искажений
        ERROR_TYPES = {
            'typo': lambda w, p: w[:p] + random.choice('абвгдеёжзиклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz') + w[p+1:] if len(w) > p+1 else w,
            'delete': lambda w, p: w[:p] + w[p+1:] if len(w) > p+1 else w,
            'swap': lambda w, p: w[:p] + w[p+1] + w[p] + w[p+2:] if len(w) > p+2 else w,
            'double': lambda w, p: w[:p] + w[p] + w[p:] if len(w) > p else w,
        }
        
        def distort_text(text: str, error_rate: float = 0.15) -> str:
            """Добавляет случайные ошибки в текст"""
            words = text.split()
            distorted = []
            
            for word in words:
                if random.random() < error_rate and len(word) > 3:
                    error_type = random.choice(list(ERROR_TYPES.keys()))
                    pos = random.randint(1, len(word) - 2)
                    try:
                        distorted_word = ERROR_TYPES[error_type](word, pos)
                        distorted.append(distorted_word)
                    except:
                        distorted.append(word)
                else:
                    distorted.append(word)
            
            return ' '.join(distorted)
        
        # Дополнительные тестовые фразы
        test_phrases = [
            {"ru": "Сегодня прекрасная погода для прогулки в парке.", "en": "Today is a beautiful day for a walk in the park."},
            {"ru": "Машинное обучение революционизирует обработку естественного языка.", "en": "Machine learning is revolutionizing natural language processing."},
            {"ru": "Компания объявила о запуске нового продукта в следующем квартале.", "en": "The company announced the launch of a new product next quarter."},
        ]
        
        additional_tests = []
        
        for i, phrase in enumerate(test_phrases, start=10):
            if "ru" in phrase:
                ref_ru = phrase["ru"]
                distorted_ru = distort_text(ref_ru)
                additional_tests.append({
                    "id": f"corr_ru_auto_{i}",
                    "task_type": "correction",
                    "language": "ru",
                    "input_text": distorted_ru,
                    "reference_text": ref_ru
                })
            
            if "en" in phrase:
                ref_en = phrase["en"]
                distorted_en = distort_text(ref_en)
                additional_tests.append({
                    "id": f"corr_en_auto_{i}",
                    "task_type": "correction",
                    "language": "en",
                    "input_text": distorted_en,
                    "reference_text": ref_en
                })
        
        # Сохранение
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "auto_generated_tests.json", "w", encoding="utf-8") as f:
            json.dump({"tests": additional_tests}, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[TestGenerator] Сгенерировано {len(additional_tests)} дополнительных тестов")
        return len(additional_tests)
    
    @staticmethod
    def validate_test_files() -> Dict[str, Any]:
        """
        Проверка существования всех тестовых файлов
        
        Returns:
            Словарь с результатами валидации
        """
        from pathlib import Path
        
        logger.info("[TestGenerator] Валидация тестовых файлов")
        
        all_cases = TestGenerator.get_test_cases()
        results = {
            "total": len(all_cases),
            "valid": 0,
            "invalid": 0,
            "missing_files": []
        }
        
        for case in all_cases:
            is_valid = True
            
            for file_key in ["input_file", "reference_text_file", "reference_summary_file"]:
                filepath = BASE_DIR / case.get(file_key, "")
                if not filepath.exists():
                    results["missing_files"].append({
                        "test_id": case["id"],
                        "file": file_key,
                        "path": str(filepath)
                    })
                    is_valid = False
            
            if is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
        
        logger.info(f"[TestGenerator] Валидация завершена: {results['valid']}/{results['total']} валидных")
        return results


# =============================================================================
# ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TestGenerator - Тестирование")
    print("=" * 80)
    
    # Тест 1: Получение всех тестов
    print("\n📋 ТЕСТ 1: Получение всех тестовых кейсов")
    try:
        cases = TestGenerator.get_test_cases()
        print(f"  ✅ Найдено {len(cases)} тестов")
        for case in cases:
            print(f"     └─ {case['id']} ({case['language']}/{case['domain']})")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
    
    # Тест 2: Валидация файлов
    print("\n📋 ТЕСТ 2: Валидация тестовых файлов")
    try:
        validation = TestGenerator.validate_test_files()
        print(f"  ✅ Валидных: {validation['valid']}/{validation['total']}")
        if validation['missing_files']:
            print(f"  ⚠️  Отсутствуют файлы:")
            for missing in validation['missing_files']:
                print(f"     └─ {missing['test_id']}: {missing['path']}")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
    
    # Тест 3: Получение по языку
    print("\n📋 ТЕСТ 3: Получение тестов по языку")
    try:
        ru_tests = TestGenerator.get_test_ids_by_language("ru")
        en_tests = TestGenerator.get_test_ids_by_language("en")
        print(f"  ✅ Русские тесты: {len(ru_tests)} - {ru_tests}")
        print(f"  ✅ Английские тесты: {len(en_tests)} - {en_tests}")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
    
    print("\n" + "=" * 80)
    print("Тестирование завершено")
    print("=" * 80)