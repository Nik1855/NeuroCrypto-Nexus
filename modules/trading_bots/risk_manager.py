import logging
import numpy as np
from neuro_utils.crypto_tools import safe_divide


class RiskAssessor:
    VOLATILITY_LEVELS = {
        "low": (0, 2),
        "medium": (2, 5),
        "high": (5, 10),
        "extreme": (10, 100)
    }

    @staticmethod
    def assess_risk(symbol, signal, context):
        """Комплексная оценка риска для торговой операции"""
        volatility = context.get('volatility', 1.0)
        liquidity = context.get('liquidity', 10 ** 6)
        whale_activity = context.get('whale_activity', 0)
        sentiment = context.get('sentiment', 0)

        # Базовый риск
        risk_score = 0

        # Корректировка на основе волатильности
        if volatility > 5:
            risk_score += 4
        elif volatility > 3:
            risk_score += 2

        # Корректировка на основе ликвидности
        if liquidity < 500000:  # Низкая ликвидность
            risk_score += 3

        # Корректировка на основе активности китов
        if whale_activity > 5:  # Высокая активность китов
            risk_score += 2

        # Корректировка на основе сигнала
        if "STRONG" in signal and "BUY" in signal:
            risk_score -= 1  # Снижение риска для сильных сигналов покупки
        elif "SELL" in signal:
            risk_score += 1

        # Корректировка на основе настроений
        if sentiment < -0.3:
            risk_score += 1

        # Ограничение 0-10
        risk_score = max(0, min(risk_score, 10))

        logging.info(f"Оценка риска для {symbol}: {risk_score:.2f}/10")
        return risk_score

    @staticmethod
    def calculate_volatility(prices, window=24):
        """Расчет исторической волатильности"""
        if len(prices) < window:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            returns.append(safe_divide(prices[i] - prices[i - 1], prices[i - 1]))

        volatility = np.std(returns) * np.sqrt(365)  # Годовая волатильность
        return volatility

    @staticmethod
    def classify_volatility(volatility):
        """Классификация уровня волатильности"""
        for level, (low, high) in RiskAssessor.VOLATILITY_LEVELS.items():
            if low <= volatility < high:
                return level
        return "unknown"