import os
import importlib
import logging
import subprocess
from .code_surgeon import CodeSurgeon


class ErrorAutoFix:
    @staticmethod
    def diagnose_and_repair(error):
        """Диагностика и автоматическое исправление ошибок"""
        error_type = type(error).__name__
        error_msg = str(error)
        logging.warning(f"Авто-лечение ошибки: {error_type} - {error_msg}")

        # Классификация ошибок
        if "CUDA out of memory" in error_msg:
            return CodeSurgeon.fix_memory_error()
        elif "shape" in error_msg and "mismatch" in error_msg:
            return CodeSurgeon.fix_shape_mismatch(error_msg)
        elif "module not found" in error_msg:
            module_name = error_msg.split()[-1].strip("'")
            return CodeSurgeon.install_missing_module(module_name)
        elif "key error" in error_msg.lower():
            return CodeSurgeon.fix_key_error(error_msg)
        else:
            return CodeSurgeon.general_fix(error_msg)

    @staticmethod
    def model_specific_fix(model_name, error):
        """Специфичные исправления для моделей"""
        logging.info(f"Применение модели-специфичного фикса для {model_name}")
        if "lstm" in model_name.lower():
            return CodeSurgeon.adjust_lstm_parameters()
        elif "transformer" in model_name.lower():
            return CodeSurgeon.adjust_attention_mechanism()
        elif "gnn" in model_name.lower():
            return CodeSurgeon.fix_graph_data_structure()

    @staticmethod
    def api_failure_fix(api_name, error):
        """Исправление сбоев API"""
        logging.warning(f"Сбой API {api_name}: {error}")
        solutions = {
            "rate_limit": "Увеличение интервала запросов",
            "invalid_key": "Проверка и обновление API ключа",
            "network": "Проверка соединения и ретрай запросов"
        }

        for issue, solution in solutions.items():
            if issue in str(error).lower():
                logging.info(f"Применение решения: {solution}")
                return solution

        return "Переключение на резервный API"