import torch
import torchquantum as tq
import logging


class QuantumFeatureExtractor(tq.QuantumModule):
    """Квантово-вдохновленный экстрактор признаков"""

    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'rx', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'rz', 'wires': [2]}
        ])
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=device)

        self.encoder(qdev, x)
        self.q_layer(qdev)
        return self.measure(qdev)


class QuantumEnhancedModel(torch.nn.Module):
    """Гибридная квантово-классическая модель"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.quantum_feature = QuantumFeatureExtractor(n_wires=3)
        self.classical = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )

    def forward(self, x):
        quantum_features = self.quantum_feature(x)
        return self.classical(quantum_features)