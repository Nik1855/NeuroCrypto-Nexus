import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CryptoGNN(nn.Module):
    """Графовая нейросеть для анализа крипторынка"""

    def __init__(self, node_features=10, hidden_channels=64, output_size=1):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        # Глобальное среднее объединение
        x = torch.mean(x, dim=0, keepdim=True)

        return self.fc(x)

    @staticmethod
    def create_market_graph(symbols, market_data):
        """Создание графа рынка на основе корреляций"""
        # Заглушка: реальная реализация будет использовать корреляции
        nodes = len(symbols)
        edge_index = []

        # Пример: полносвязный граф
        for i in range(nodes):
            for j in range(nodes):
                if i != j:
                    edge_index.append([i, j])

        return {
            'x': torch.randn(nodes, 10),  # Признаки узлов
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        }