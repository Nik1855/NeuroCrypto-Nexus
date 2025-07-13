import hashlib
import logging
import pandas as pd
import numpy as np


class CryptoTools:
    @staticmethod
    def normalize_address(address):
        """Нормализация крипто-адресов"""
        if address.startswith("0x"):
            return address.lower()
        return "0x" + address.lower()

    @staticmethod
    def calculate_token_hash(symbol):
        """Создание уникального хэша для токена"""
        return hashlib.sha256(symbol.encode()).hexdigest()

    @staticmethod
    def safe_divide(a, b):
        """Безопасное деление с обработкой нулей"""
        return a / b if b != 0 else 0

    @staticmethod
    def detect_anomalies(prices, threshold=3.0):
        """Обнаружение аномалий в ценовых данных"""
        mean = np.mean(prices)
        std = np.std(prices)
        return [p for p in prices if abs(p - mean) > threshold * std]

    @staticmethod
    def calculate_correlation(prices1, prices2):
        """Расчет корреляции между двумя активами"""
        series1 = pd.Series(prices1)
        series2 = pd.Series(prices2)
        return series1.corr(series2)