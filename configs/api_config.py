API_CONFIG = {
    "blockchain": {
        "etherscan": {
            "base_url": "https://api.etherscan.io/api",
            "required": True,
            "key_param": "apikey",
            "rate_limit": "5/1"  # 5 запросов в секунду
        },
        "bscscan": {
            "base_url": "https://api.bscscan.com/api",
            "required": True,
            "key_param": "apikey",
            "rate_limit": "5/1"
        }
    },
    "social": {
        "twitter": {
            "base_url": "https://api.twitter.com/2",
            "required": True,
            "auth_type": "bearer",
            "endpoints": {
                "search": "/tweets/search/recent",
                "user": "/users/by"
            }
        },
        "reddit": {
            "base_url": "https://oauth.reddit.com",
            "required": False,
            "auth_type": "oauth",
            "scopes": ["read"]
        },
        "telegram": {
            "base_url": "https://api.telegram.org",
            "required": True,
            "auth_type": "bot_token"
        }
    },
    "market": {
        "coingecko": {
            "base_url": "https://api.coingecko.com/api/v3",
            "required": True,
            "rate_limit": "50/60",
            "priority": "high"
        },
        "dex_screener": {
            "base_url": "https://api.dexscreener.com/latest/dex",
            "required": True,
            "rate_limit": "unlimited"
        },
        "coinmarketcap": {
            "base_url": "https://pro-api.coinmarketcap.com/v1",
            "required": False,
            "auth_type": "api_key"
        }
    },
    "trading": {
        "binance": {
            "base_url": "https://api.binance.com",
            "required": False,
            "auth_type": "api_key+secret",
            "permissions": ["spot", "margin"]
        },
        "bybit": {
            "base_url": "https://api.bybit.com",
            "required": False,
            "auth_type": "api_key+secret",
            "permissions": ["spot", "linear"]
        },
        "kucoin": {
            "base_url": "https://api.kucoin.com",
            "required": False,
            "auth_type": "api_key+secret"
        }
    },
    "premium": {
        "glassnode": {
            "base_url": "https://api.glassnode.com/v1",
            "required": False,
            "auth_type": "api_key",
            "tiers": ["advanced"]
        },
        "cryptoquant": {
            "base_url": "https://api.cryptoquant.com/v1",
            "required": False,
            "auth_type": "api_key",
            "tiers": ["professional"]
        },
        "messari": {
            "base_url": "https://data.messari.io/api/v1",
            "required": False,
            "auth_type": "api_key"
        }
    }
}