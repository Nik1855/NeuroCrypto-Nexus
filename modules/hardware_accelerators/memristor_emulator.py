import torch
import torch.nn as nn
import logging


class MemristorMatrix(nn.Module):
    """Эмулятор мемристорной матрицы для аппаратной оптимизации"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight)

    def initialize_weights(self):
        """Инициализация весов для нейроморфных вычислений"""
        torch.nn.init.normal_(self.weight, mean=0.5, std=0.1)
        logging.info("Мемристорные веса инициализированы")


class MemristorEnhancedLinear(nn.Linear):
    """Линейный слой с мемристорной оптимизацией"""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.memristor = MemristorMatrix(in_features, out_features)

    def forward(self, x):
        return self.memristor(x)

    def apply_noise(self, noise_level=0.01):
        """Применение шума для эмуляции реальных мемристоров"""
        with torch.no_grad():
            noise = torch.randn_like(self.memristor.weight) * noise_level
            self.memristor.weight.add_(noise)