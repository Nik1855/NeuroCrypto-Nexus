import logging
import numpy as np


class RiskAssessor:
    @staticmethod
    def assess_risk(symbol, signal, market_conditions=None):
        """Оценка риска для торговой операции"""
        volatility = RiskAssessor.calculate_volatility(symbol)
        liquidity = RiskAssessor.get_liquidity_score(symbol)

        risk_score = volatility * 0.7 + (1 - liquidity) * 0.3

        if "STRONG" in signal:
            risk_score *= 1.2
        elif "CAUTIOUS" in signal:
            risk_score *= 0.8

        logging.info(f"Оценка риска для {symbol}: {risk_score:.2f}/10")
        return risk_score

    @staticmethod
    def calculate_volatility(symbol, window=24):
        """Расчет волатильности за последние N часов"""
        # Заглушка для реальной реализации
        return np.random.uniform(0.5, 9.5)

    @staticmethod
    def get_liquidity_score(symbol):
        """Оценка ликвидности актива"""
        # Заглушка для реальной реализации
        return np.random.uniform(0.3, 0.95)