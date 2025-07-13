import logging
import numpy as np
from neuro_utils.ai_helpers import optimize_function


class StrategyOptimizer:
    def __init__(self, initial_strategy):
        self.strategy = initial_strategy
        self.performance_history = []

    def update_strategy(self, market_conditions, profit_loss):
        """Адаптивное обновление стратегии"""
        self.performance_history.append(profit_loss)

        if len(self.performance_history) > 5:
            avg_perf = np.mean(self.performance_history[-5:])
            if avg_perf < 0:
                self.optimize_strategy(market_conditions)

    def optimize_strategy(self, market_conditions):
        """Оптимизация стратегии с помощью ИИ"""
        logging.info("Оптимизация торговой стратегии...")

        # ИИ-оптимизация параметров
        optimized_params = optimize_function(
            self.strategy['params'],
            market_conditions
        )

        self.strategy['params'] = optimized_params
        logging.info(f"Стратегия обновлена: {optimized_params}")

    def get_current_strategy(self):
        """Получение текущей стратегии"""
        return self.strategy.copy()

    def hybrid_strategy_selection(self, market_condition):
        """Гибридный выбор стратегии на основе рыночных условий"""
        if market_condition['volatility'] > 7:
            return "volatility_strategy"
        elif market_condition['sentiment'] > 0.5:
            return "bull_market_strategy"
        elif market_condition['sentiment'] < -0.5:
            return "bear_market_strategy"
        else:
            return "neutral_strategy"