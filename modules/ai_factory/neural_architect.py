import torch
import logging
from torch import nn
from neuro_utils.ai_helpers import dynamic_feature_selection


class NeuralArchitect:
    @staticmethod
    def design_model(input_size, output_size, complexity="medium"):
        """Автоматическое проектирование архитектуры модели"""
        logging.info(f"Проектирование модели сложности {complexity}")

        if complexity == "simple":
            return nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            elif complexity == "medium":
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_size))
        else:  # complex
            return nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_size))

    @staticmethod
    def optimize_for_device(model, device):
        """Оптимизация модели для целевого устройства"""
        model.to(device)
        if "cuda" in device.type:
            model = torch.compile(model)
            model.half()  # FP16 precision
            logging.info("Модель оптимизирована для GPU")
        return model

    @staticmethod
    def adaptive_feature_selection(model, feature_importance):
        """Адаптивный выбор признаков на основе важности"""
        selected_features = dynamic_feature_selection(
            list(range(len(feature_importance))),
            feature_importance
        )
        logging.info(f"Выбрано {len(selected_features)} признаков из {len(feature_importance)}")
        return selected_features