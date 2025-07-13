import logging
import time
from modules.trading_bots.risk_manager import RiskAssessor
from neuro_utils.crypto_tools import safe_divide


class AutonomousTrader:
    def __init__(self, models, strategy="adaptive_ai"):
        self.models = models
        self.strategy = strategy
        self.risk_manager = RiskAssessor()
        self.active = False
        self.positions = {}

    def execute_trade(self, symbol, signal, price, context):
        """Выполнение торговой операции с учетом контекста"""
        if not self.active:
            return

        risk_score = self.risk_manager.assess_risk(symbol, signal, context)

        if risk_score > 7.5:  # Очень высокий риск
            logging.warning(f"Торговля приостановлена для {symbol}: риск {risk_score}/10")
            return

        # Расчет размера позиции
        position_size = self.calculate_position_size(risk_score, context)

        # Логика исполнения ордера
        order_type = "MARKET"
        if signal.startswith("STRONG"):
            order_type = "LIMIT"  # Для лучшего исполнения

        logging.info(
            f"Исполнение: {signal} {symbol} {position_size:.4f} по цене ~{price:.2f} "
            f"(риск: {risk_score:.2f}/10)"
        )

        # Обновление позиций
        self.positions[symbol] = {
            "entry_price": price,
            "size": position_size,
            "signal": signal,
            "timestamp": time.time(),
            "risk_score": risk_score
        }

    def calculate_position_size(self, risk_score, context):
        """Расчет размера позиции на основе риска и капитала"""
        max_position = 0.1  # Макс 10% капитала на одну позицию
        risk_adjustment = 1 - (risk_score / 15)  # Корректировка на основе риска
        volatility_adjustment = 1 / (1 + context.get('volatility', 0.05))

        position_size = max_position * risk_adjustment * volatility_adjustment
        return max(position_size, 0.01)  # Минимум 1%

    def manage_open_positions(self, market_data):
        """Управление открытыми позициями"""
        for symbol, position in self.positions.items():
            current_price = market_data[symbol]['price']
            profit = safe_divide(current_price - position['entry_price'], position['entry_price'])

            # Проверка стоп-лосса
            if profit < -0.05:  # -5%
                self.close_position(symbol, current_price, "stop_loss")

            # Проверка тейк-профита
            elif profit > 0.15:  # +15%
                self.close_position(symbol, current_price, "take_profit")

    def close_position(self, symbol, price, reason):
        """Закрытие позиции"""
        position = self.positions.pop(symbol, None)
        if position:
            profit = (price - position['entry_price']) * position['size']
            logging.info(
                f"Закрытие позиции {symbol} по {price:.2f} ({reason}). "
                f"Прибыль: ${profit:.2f}"
            )