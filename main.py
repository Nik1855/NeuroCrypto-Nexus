import os

# становка переменных окружения для совместимости
os.environ["NPY_PROMOTION_STATE"] = "weak"
os.environ["NPY_USE_GETITEM"] = "1"
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["TORCHVISION_DISABLE_FBCODE"] = "1"

import torch
import logging
# import torch_tensorrt  # Т  Т
from core.neural_engine import NeuralEngine
from interfaces.neuro_cli import NeuroCLI
from modules.self_healing.system_doctor import SystemDoctor

# астройка прецизионных вычислений
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
# torch_tensorrt.runtime.set_multi_device_safe_mode(True)  # Т  

def initialize_neuro_system():
    """нициализация нейроморфной системы"""
    neuro_engine = NeuralEngine(
        device_map="auto",
        precision="fp16",
        neuromorphic_mode=True
    )
    SystemDoctor.health_check()
    NeuroCLI.activate_control_room()
    return neuro_engine

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("NeuroCortex")
    logger.info("⚡ апуск NeuroCrypto Nexus v4.2")
    
    try:
        neuro_system = initialize_neuro_system()
        neuro_system.activate_autonomous_mode()
    except Exception as e:
        SystemDoctor.critical_recovery(e)
