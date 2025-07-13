NEUROMORPHIC_CONFIG = {
    "memristor": {
        "simulation_mode": "realistic",
        "precision": "fp16",
        "matrix_size": 1024,
        "noise_level": 0.02
    },
    "hardware_acceleration": {
        "tensorrt": True,
        "cuda_graphs": True,
        "mixed_precision": True,
        "quantization": "fp16"
    },
    "autonomous_trading": {
        "enabled": False,
        "risk_tolerance": 0.03,
        "max_exposure": 0.25,
        "exchanges": ["binance", "bybit"],
        "strategies": {
            "volatility": {
                "threshold": 5.0,
                "position_size": 0.05
            },
            "sentiment": {
                "positive_threshold": 0.3,
                "negative_threshold": -0.3
            }
        }
    },
    "ai_models": {
        "primary": ["hyper_lstm", "temporal_cn"],
        "secondary": ["gnn", "hybrid_attention"],
        "fusion_method": "neuro_attention",
        "retrain_interval": 24  # часов
    },
    "self_healing": {
        "auto_restart": True,
        "memory_threshold": 85,
        "error_logging": "verbose",
        "auto_fix_level": "advanced"
    },
    "data_processing": {
        "window_size": 100,
        "update_interval": 5,  # минут
        "max_features": 25
    }
}

PERFORMANCE_OPTIMIZATION = {
    "gpu": {
        "tensor_cores": True,
        "memory_allocation": "aggressive",
        "max_vram_usage": 7.0  # GB
    },
    "cpu": {
        "max_threads": 8,
        "priority": "high"
    },
    "network": {
        "compression": True,
        "batch_requests": True
    }
}