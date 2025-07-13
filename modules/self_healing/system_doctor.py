import torch
import gc
import logging
import importlib
from neuro_utils.resource_monitor import ResourceMonitor


class SystemDoctor:
    @staticmethod
    def health_check():
        """Комплексная проверка здоровья системы"""
        report = {
            "gpu_health": SystemDoctor.check_gpu(),
            "memory_health": SystemDoctor.check_memory(),
            "model_health": SystemDoctor.check_models(),
            "api_health": SystemDoctor.check_apis()
        }
        return report

    @staticmethod
    def check_gpu():
        """Диагностика GPU"""
        if not torch.cuda.is_available():
            return {"status": "critical", "message": "CUDA недоступно"}

        try:
            torch.cuda.empty_cache()
            tensor = torch.rand(1000, 1000, device="cuda")
            _ = tensor @ tensor.T
            return {"status": "optimal", "utilization": torch.cuda.utilization()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def emergency_cleanup():
        """Экстренная очистка ресурсов"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logging.warning("Экстренная очистка памяти выполнена")

    @staticmethod
    def critical_recovery(error):
        """Критическое восстановление после сбоя"""
        from modules.self_healing.error_autofix import CodeSurgeon
        SystemDoctor.emergency_cleanup()
        logging.critical(f"КРИТИЧЕСКИЙ СБОЙ: {str(error)}")

        # Авто-лечение
        CodeSurgeon.diagnose_and_repair(error)

        # Перезагрузка подсистем
        SystemDoctor.restart_subsystems()

    @staticmethod
    def restart_subsystems():
        """Перезагрузка подсистем"""
        subsystems = [
            "modules.market_analyzers.sentiment_fusion",
            "modules.ai_factory.model_orchestrator",
            "core.neuromorphic_controller"
        ]

        for sub in subsystems:
            try:
                module = importlib.import_module(sub)
                if hasattr(module, "restart"):
                    module.restart()
                    logging.info(f"Подсистема перезапущена: {sub}")
            except Exception as e:
                logging.error(f"Ошибка перезапуска {sub}: {str(e)}")