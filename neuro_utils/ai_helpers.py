import numpy as np
import logging
import torch


class AIHelpers:
    @staticmethod
    def optimize_function(params, conditions):
        """Оптимизация функции с использованием эволюционных алгоритмов"""
        # Упрощенная реализация для примера
        new_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                adjustment = np.random.uniform(-0.1, 0.1)
                new_params[key] = value * (1 + adjustment)
        return new_params

    @staticmethod
    def detect_anomalies(data, threshold=3):
        """Обнаружение аномалий в данных"""
        mean = np.mean(data)
        std = np.std(data)
        anomalies = [x for x in data if abs(x - mean) > threshold * std]
        return anomalies

    @staticmethod
    def dynamic_feature_selection(features, importance_scores):
        """Динамический выбор признаков на основе важности"""
        sorted_indices = np.argsort(importance_scores)[::-1]
        selected_features = []
        total_importance = 0

        for idx in sorted_indices:
            if total_importance < 0.95:  # Пока не достигнем 95% важности
                selected_features.append(features[idx])
                total_importance += importance_scores[idx]
            else:
                break

        return selected_features

    @staticmethod
    def tensor_to_numpy(tensor):
        """Конвертация тензора в numpy массив"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return tensor