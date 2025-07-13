import requests
import logging
import time
from neuro_utils.crypto_tools import normalize_address, calculate_token_hash


class BlockchainScanner:
    API_ENDPOINTS = {
        'ethereum': "https://api.etherscan.io/api",
        'bsc': "https://api.bscscan.com/api",
        'polygon': "https://api.polygonscan.com/api",
        'arbitrum': "https://api.arbiscan.io/api"
    }

    @staticmethod
    def get_token_contract(symbol):
        """Получение контракта токена по символу"""
        # В реальной реализации это будет база данных или API
        token_map = {
            "BTC": "0xbtc_contract",
            "ETH": "0xeth_contract",
            "BNB": "0xbnb_contract",
            "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7"
        }
        return token_map.get(symbol.upper(), "")

    @staticmethod
    def get_large_transactions(symbol, min_value=100000, timeframe="1h"):
        """Получение крупных транзакций за указанный период"""
        chain = BlockchainScanner.detect_chain(symbol)
        if not chain:
            return []

        contract_address = BlockchainScanner.get_token_contract(symbol.split('/')[0])
        if not contract_address:
            logging.error(f"Контракт для {symbol} не найден")
            return []

        params = {
            'module': 'account',
            'action': 'tokentx',
            'contractaddress': contract_address,
            'sort': 'desc',
            'apikey': 'YOUR_API_KEY'
        }

        # Добавление временного фильтра
        if timeframe == "1h":
            params['startblock'] = 0  # Заглушка, реальная реализация будет использовать временные метки
            params['endblock'] = 99999999

        try:
            response = requests.get(
                BlockchainScanner.API_ENDPOINTS[chain],
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            large_txs = []
            current_time = time.time()

            for tx in data.get('result', []):
                value_usd = float(tx.get('value', 0)) / 10 ** int(tx.get('tokenDecimal', 18))

                # Фильтрация по времени
                tx_time = int(tx.get('timeStamp', current_time))
                if current_time - tx_time > 3600:  # Только последний час
                    continue

                if value_usd >= min_value:
                    large_txs.append({
                        'symbol': symbol,
                        'from_address': normalize_address(tx['from']),
                        'to_address': normalize_address(tx['to']),
                        'amount': float(tx['value']) / 10 ** int(tx['tokenDecimal']),
                        'amount_usd': value_usd,
                        'hash': tx['hash'],
                        'timestamp': tx_time,
                        'blockchain': chain
                    })

            return large_txs
        except Exception as e:
            logging.error(f"Ошибка сканирования блокчейна: {str(e)}")
            return []

    @staticmethod
    def detect_chain(symbol):
        """Определение блокчейна по символу"""
        chains = {
            'ETH': 'ethereum',
            'BTC': 'bitcoin',
            'BNB': 'bsc',
            'MATIC': 'polygon',
            'ARB': 'arbitrum'
        }
        return chains.get(symbol.upper().split('/')[0], 'ethereum')