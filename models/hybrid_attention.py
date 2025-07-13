import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging


class HybridAttention(nn.Module):
    """Гибридный механизм внимания для анализа крипторынка"""

    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim должно делиться на num_heads"

        # Проекционные слои
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Временное внимание
        self.temporal_attn = TemporalAttention(hidden_dim)

        # Финальный слой
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Проекции
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Масштабированное скалярное произведение
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Применение весов
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)

        # Временное внимание
        temporal_context = self.temporal_attn(context)

        # Финальная проекция
        output = self.fc(temporal_context)
        return output


class TemporalAttention(nn.Module):
    """Внимание к временным паттернам"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        seq_len = x.size(1)
        # Создание временных меток
        positions = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(0).unsqueeze(-1)
        positions = positions.expand(x.size(0), -1, -1)

        # Комбинирование признаков с временными метками
        combined = torch.cat([x, positions], dim=-1)

        # Вычисление весов внимания
        energy = self.tanh(self.attn(combined))
        attn_weights = self.softmax(energy)

        # Взвешенная сумма
        weighted = torch.sum(x * attn_weights, dim=1)
        return weighted