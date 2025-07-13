import time
import psutil
import torch
import logging
import threading


class ResourceMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        self.stats = {
            "cpu": [],
            "ram": [],
            "gpu": [],
            "vram": []
        }

    def start(self):
        """Запуск мониторинга ресурсов в отдельном потоке"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("Мониторинг ресурсов запущен")

    def stop(self):
        """Остановка мониторинга"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logging.info("Мониторинг ресурсов остановлен")

    def monitor_loop(self):
        """Цикл мониторинга"""
        while self.running:
            cpu_percent = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            gpu_usage = 0
            gpu_mem = 0

            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
                gpu_mem = torch.cuda.memory_allocated() / 1e9

            # Сохранение статистики
            self.stats["cpu"].append(cpu_percent)
            self.stats["ram"].append(ram_usage)
            self.stats["gpu"].append(gpu_usage)
            self.stats["vram"].append(gpu_mem)

            logging.info(
                f"Ресурсы: CPU {cpu_percent}% | RAM {ram_usage}% | "
                f"GPU {gpu_usage}% | VRAM {gpu_mem:.2f} GB"
            )

            # Предупреждения при высокой нагрузке
            if cpu_percent > 90:
                logging.warning("Высокая загрузка CPU!")
            if ram_usage > 90:
                logging.warning("Высокая загрузка RAM!")
            if gpu_usage > 90:
                logging.warning("Высокая загрузка GPU!")
            if gpu_mem > 7.5:  # > 7.5 GB на 8GB карте
                logging.warning("Высокое использование VRAM!")

            time.sleep(self.interval)

    def get_stats(self):
        """Получение статистики использования ресурсов"""
        return self.stats

    def reset_stats(self):
        """Сброс статистики"""
        self.stats = {
            "cpu": [],
            "ram": [],
            "gpu": [],
            "vram": []
        }