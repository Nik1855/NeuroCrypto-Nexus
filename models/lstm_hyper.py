import torch
import torch.nn as nn
from neuro_utils.memristor_emulator import MemristorEnhancedLinear


class HyperLSTM(nn.Module):
    """LSTM с гиперсетью для динамической адаптации весов"""

    def __init__(self, input_size=10, hidden_size=64, output_size=1, hyper_hidden=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Гиперсеть для генерации весов LSTM
        self.hyper_net = nn.Sequential(
            nn.Linear(input_size, hyper_hidden),
            nn.ReLU(),
            MemristorEnhancedLinear(hyper_hidden, 4 * hidden_size * (input_size + hidden_size))

        # Базовый LSTM слой (веса будут установлены гиперсетью)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Выходной слой
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Генерация весов LSTM гиперсетью
        hyper_input = x.mean(dim=1)  # Усреднение по временному измерению
        generated_weights = self.hyper_net(hyper_input)

        # Динамическая установка весов в LSTMCell
        self.set_dynamic_weights(generated_weights)

        # Обработка последовательности
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        # Последний выход
        last_output = outputs[-1]
        return self.fc(last_output)

    def set_dynamic_weights(self, generated_weights):
        """Установка сгенерированных весов в LSTMCell"""
        # Разделение весов на компоненты
        total_params = 4 * self.hidden_size * (self.input_size + self.hidden_size + 1)
        assert generated_weights.numel() == total_params, "Неверный размер генерируемых весов"

        # Формирование весовых матриц
        weights = generated_weights.view(4, self.hidden_size, self.input_size + self.hidden_size)

        # Установка весов в LSTMCell
        with torch.no_grad():
            # Веса для входных данных
            self.lstm_cell.weight_ih.data = weights[:, :, :self.input_size].view(-1, self.input_size)
            # Веса для скрытого состояния
            self.lstm_cell.weight_hh.data = weights[:, :, self.input_size:].view(-1, self.hidden_size)