import torch
import logging
import numpy as np


class AdaptiveLearner:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_history = []
        self.lr_history = []

    def dynamic_learning_rate(self, loss):
        """Адаптивное изменение learning rate"""
        self.loss_history.append(loss)

        if len(self.loss_history) > 10:
            last_avg = sum(self.loss_history[-10:-5]) / 5
            current_avg = sum(self.loss_history[-5:]) / 5

            if current_avg > last_avg * 1.1:  # Ухудшение
                self.reduce_learning_rate()
            elif current_avg < last_avg * 0.9:  # Улучшение
                self.increase_learning_rate()

    def reduce_learning_rate(self, factor=0.5):
        """Уменьшение learning rate"""
        for param_group in self.optimizer.param_groups:
            new_lr = param_group['lr'] * factor
            param_group['lr'] = new_lr
            self.lr_history.append(new_lr)
        logging.info(f"Learning rate уменьшен до {new_lr}")

    def increase_learning_rate(self, factor=1.2):
        """Увеличение learning rate"""
        for param_group in self.optimizer.param_groups:
            new_lr = param_group['lr'] * factor
            param_group['lr'] = new_lr
            self.lr_history.append(new_lr)
        logging.info(f"Learning rate увеличен до {new_lr}")

    def apply_regularization(self, loss):
        """Динамическое применение регуляризации"""
        if loss < 0.1:
            # Уменьшение регуляризации при хорошем качестве
            self.model.regularization_factor *= 0.9
        else:
            # Увеличение регуляризации при переобучении
            self.model.regularization_factor *= 1.1

    def early_stopping(self, val_loss, patience=5):
        """Ранняя остановка на основе потерь на валидации"""
        if len(self.loss_history) < patience:
            return False

        recent_losses = self.loss_history[-patience:]
        if all(loss > val_loss for loss in recent_losses):
            logging.info("Активация ранней остановки")
            return True
        return False