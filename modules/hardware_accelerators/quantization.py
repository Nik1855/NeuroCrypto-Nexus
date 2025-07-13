import torch
import logging
from torch.quantization import quantize_dynamic


class QuantizationEngine:
    @staticmethod
    def dynamic_quantize(model, precision='int8'):
        """Динамическое квантование модели"""
        try:
            if precision == 'int8':
                quantized_model = quantize_dynamic(
                    model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
                )
            elif precision == 'fp16':
                quantized_model = model.half()
            elif precision == 'bf16':
                quantized_model = model.to(torch.bfloat16)
            else:
                quantized_model = model

            logging.info(f"Модель квантована в формате {precision}")
            return quantized_model
        except Exception as e:
            logging.error(f"Ошибка квантования: {str(e)}")
            return model