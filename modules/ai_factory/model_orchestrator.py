import torch
import torch.nn as nn
import logging
from models import HyperLSTM, CryptoGNN, TemporalConvNet, HybridAttention
from neuro_utils.memristor_emulator import MemristorEnhancedLinear


class ModelEnsemble(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.models = {
            "hyper_lstm": HyperLSTM().to(device).half(),
            "temporal_cn": TemporalConvNet().to(device).half(),
            "gnn": CryptoGNN().to(device).half(),
            "hybrid_attention": HybridAttention().to(device).half()
        }
        self.fusion_layer = MemristorEnhancedLinear(256, 1).to(device)

        # Инициализация оптимизаторов
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=0.001)
            for name, model in self.models.items()
        }

        # Компиляция моделей
        self.compile_models()

    def compile_models(self):
        """Компиляция моделей для ускорения"""
        from modules.hardware_accelerators.tensor_optimizer import TensorRTCompiler
        for name in self.models:
            try:
                self.models[name] = TensorRTCompiler.compile(
                    self.models[name],
                    precision="fp16"
                )
                logging.info(f"Модель {name} скомпилирована")
            except Exception as e:
                logging.error(f"Ошибка компиляции {name}: {str(e)}")

    def forward(self, x):
        outputs = []
        for name, model in self.models.items():
            try:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    outputs.append(out)
            except RuntimeError as e:
                self.handle_model_error(name, e)

        combined = torch.cat(outputs, dim=-1)
        return self.fusion_layer(combined)

    def handle_model_error(self, model_name, error):
        from modules.self_healing.error_autofix import ErrorAutoFix
        logging.error(f"Ошибка в модели {model_name}: {str(error)}")
        ErrorAutoFix.model_specific_fix(model_name, error)

    def dynamic_load_unload(self, model_name, action):
        """Динамическая загрузка/выгрузка моделей"""
        if action == "load" and model_name not in self.models:
            self.load_model(model_name)
        elif action == "unload" and model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()

    def load_model(self, model_name):
        """Динамическая загрузка модели по имени"""
        model_loader = {
            "quantum": "models.quantum_attention.QuantumAttentionModel",
            "neuro": "models.neuro_adaptive.NeuroAdaptiveModel"
        }

        if model_name in model_loader:
            module_path, class_name = model_loader[model_name].rsplit('.', 1)
            mod = __import__(module_path, fromlist=[class_name])
            model_class = getattr(mod, class_name)
            self.models[model_name] = model_class().to(self.device).half()
            logging.info(f"Модель {model_name} динамически загружена")

    def adaptive_retraining(self, new_data):
        """Адаптивное переобучение на новых данных"""
        for name, model in self.models.items():
            try:
                # Краткое обучение на новых данных
                model.train()
                optimizer = self.optimizers[name]

                # Пакетное обучение
                for i in range(0, len(new_data), 32):
                    batch = new_data[i:i + 32]
                    inputs, targets = batch[:, :-1], batch[:, -1]

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = nn.MSELoss()(outputs, targets)
                    loss.backward()
                    optimizer.step()

                logging.info(f"Модель {name} адаптирована на новых данных")
            except Exception as e:
                logging.error(f"Ошибка переобучения {name}: {str(e)}")