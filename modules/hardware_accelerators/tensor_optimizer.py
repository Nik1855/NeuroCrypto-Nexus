import torch_tensorrt
import logging


class TensorRTCompiler:
    @staticmethod
    def compile(model, precision="fp16", max_batch_size=16):
        """Компиляция модели с использованием TensorRT"""
        logging.info(f"Компиляция модели в TensorRT с точностью {precision}")

        # Конфигурация компиляции
        compile_settings = {
            "inputs": [torch_tensorrt.Input(
                min_shape=[1, *model.input_shape],
                opt_shape=[max_batch_size, *model.input_shape],
                max_shape=[max_batch_size, *model.input_shape]
            )],
            "enabled_precisions": {torch.float16} if precision == "fp16" else {torch.float32},
            "truncate_long_and_double": True,
            "device": torch_tensorrt.Device("cuda:0")
        }

        try:
            trt_model = torch_tensorrt.compile(model, **compile_settings)
            logging.info("Модель успешно скомпилирована с TensorRT")
            return trt_model
        except Exception as e:
            logging.error(f"Ошибка компиляции TensorRT: {str(e)}")
            return model