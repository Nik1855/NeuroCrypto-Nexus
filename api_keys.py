# Централизованное хранилище API ключей
API_KEYS = {
    "twitter": {
        "bearer_token": "YOUR_TWITTER_BEARER_TOKEN"
    },
    "etherscan": {
        "api_key": "YOUR_ETHERSCAN_API_KEY"
    },
    "bscscan": {
        "api_key": "YOUR_BSCSCAN_API_KEY"
    },
    "coingecko": {
        "api_key": "YOUR_COINGECKO_API_KEY"
    },
    "dex_screener": {
        "api_key": "YOUR_DEXSCREENER_API_KEY"
    },
    "binance": {
        "api_key": "YOUR_BINANCE_API_KEY",
        "api_secret": "YOUR_BINANCE_API_SECRET"
    },
    "bybit": {
        "api_key": "YOUR_BYBIT_API_KEY",
        "api_secret": "YOUR_BYBIT_API_SECRET"
    },
    "glassnode": {
        "api_key": "YOUR_GLASSNODE_API_KEY"
    },
    "cryptoquant": {
        "api_key": "YOUR_CRYPTOQUANT_API_KEY"
    },
    "telegram": {
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_id": "YOUR_TELEGRAM_CHAT_ID"
    }
}

API_TEMPLATES = {
    "new_service": {
        "api_key": "",
        "api_secret": "",
        "notes": ""
    }
}

def get_api_key(service, key_type="api_key"):
    """Безопасное получение ключа API"""
    return API_KEYS.get(service, {}).get(key_type, "")

def add_api_key(service, key_data):
    """Добавление новых API ключей"""
    if service not in API_KEYS:
        API_KEYS[service] = {}
    API_KEYS[service].update(key_data)
    print(f"Ключи для {service} успешно добавлены")