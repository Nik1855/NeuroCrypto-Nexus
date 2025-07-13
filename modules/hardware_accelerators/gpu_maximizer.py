import torch
import logging


class GPUMaximizer:
    @staticmethod
    def optimize_for_rtx4060():
        """Специфические оптимизации для RTX 4060"""
        if not torch.cuda.is_available():
            return

        device_name = torch.cuda.get_device_name(0)
        if "RTX 4060" not in device_name:
            logging.warning("Оптимизации для RTX 4060 не применяются: обнаружено другое устройство")
            return

        # Активация Tensor Cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Настройка потоков
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

        # Оптимизация памяти
        torch.cuda.set_per_process_memory_fraction(0.9)  # Оставить 10% для системы
        torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')

        logging.info("Оптимизации для RTX 4060 применены")

    @staticmethod
    def memory_optimization():
        """Экстремальное управление памятью"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')
            logging.info("Настройки распределения памяти обновлены")