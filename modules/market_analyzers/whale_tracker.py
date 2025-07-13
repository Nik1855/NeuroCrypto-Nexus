import logging
from modules.market_analyzers.blockchain_scanner import BlockchainScanner
from neuro_utils.crypto_tools import normalize_address, calculate_token_hash


class WhaleTracker:
    WHALE_THRESHOLDS = {
        "BTC": 500000,  # $500,000
        "ETH": 1000000,  # $1,000,000
        "BNB": 500000,  # $500,000
        "SOL": 1000000,  # $1,000,000
        "DEFAULT": 500000  # $500,000
    }

    @staticmethod
    def track_whale_activity(symbol, timeframe="1h"):
        """Отслеживание активности китов в реальном времени"""
        threshold = WhaleTracker.WHALE_THRESHOLDS.get(
            symbol.split('/')[0],
            WhaleTracker.WHALE_THRESHOLDS["DEFAULT"]
        )

        transactions = BlockchainScanner.get_large_transactions(
            symbol,
            min_value=threshold,
            timeframe=timeframe
        )

        whale_activity = []
        for tx in transactions:
            whale_activity.append({
                'symbol': symbol,
                'from': normalize_address(tx['from_address']),
                'to': normalize_address(tx['to_address']),
                'amount': tx['amount'],
                'amount_usd': tx['amount_usd'],
                'hash': tx['hash'],
                'timestamp': tx['timestamp'],
                'type': WhaleTracker.classify_transaction(tx)
            })

        logging.info(f"Обнаружено {len(whale_activity)} китовых транзакций для {symbol}")
        return whale_activity

    @staticmethod
    def classify_transaction(tx):
        """Классификация транзакций китов"""
        exchange_addresses = [
            "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance
            "0x28c6c06298d514Db089934071355E5743bf21d60"  # Binance 2
        ]

        if tx['to_address'] in exchange_addresses:
            return "SELL"
        elif tx['from_address'] in exchange_addresses:
            return "BUY"
        else:
            return "TRANSFER"

    @staticmethod
    def detect_whale_patterns(activity):
        """Выявление паттернов поведения китов"""
        buy_volume = sum(t['amount_usd'] for t in activity if t['type'] == "BUY")
        sell_volume = sum(t['amount_usd'] for t in activity if t['type'] == "SELL")

        return {
            "buy_ratio": buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0,
            "net_flow": buy_volume - sell_volume,
            "whale_index": len(activity) / 10  # Индекс активности китов
        }