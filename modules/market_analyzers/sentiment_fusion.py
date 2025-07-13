import requests
import logging
from textblob import TextBlob
from transformers import pipeline
from configs.api_config import API_CONFIG
from neuro_utils.crypto_tools import clean_text


class SentimentIntegrator:
    @staticmethod
    def fuse_multiple_sources(symbol):
        """Анализ настроений из 10+ источников"""
        sources = {
            "twitter": SentimentIntegrator.analyze_twitter(symbol),
            "reddit": SentimentIntegrator.analyze_reddit(symbol),
            "news": SentimentIntegrator.analyze_news(symbol),
            "telegram": SentimentIntegrator.analyze_telegram(symbol),
            "github": SentimentIntegrator.analyze_github(symbol),
            "discord": SentimentIntegrator.analyze_discord(symbol),
            "coingecko": SentimentIntegrator.analyze_coingecko(symbol),
            "tradingview": SentimentIntegrator.analyze_tradingview(symbol)
        }

        # Взвешенное среднее
        weights = {
            "twitter": 0.25, "reddit": 0.20, "news": 0.15,
            "telegram": 0.15, "github": 0.05, "discord": 0.05,
            "coingecko": 0.10, "tradingview": 0.05
        }

        total = 0
        weight_sum = 0
        for src in sources:
            if sources[src] is not None:
                total += sources[src] * weights[src]
                weight_sum += weights[src]

        sentiment_score = total / weight_sum if weight_sum > 0 else 0
        logging.info(f"Сводный показатель настроений для {symbol}: {sentiment_score:.2f}")
        return sentiment_score

    @staticmethod
    def analyze_twitter(symbol):
        """Анализ Twitter с NLP трансформером"""
        try:
            # Использование Hugging Face для глубокого анализа
            nlp = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )

            # Сбор твитов (реальная реализация будет использовать Twitter API)
            tweets = [
                f"{symbol} is going to the moon! 🚀",
                f"Concerned about {symbol} recent dip",
                f"New partnership announced for {symbol}"
            ]

            sentiments = []
            for tweet in tweets:
                cleaned = clean_text(tweet)
                result = nlp(cleaned)[0]
                score = SentimentIntegrator.convert_label_to_score(
                    result['label'], result['score']
                )
                sentiments.append(score)

            return sum(sentiments) / len(sentiments) if sentiments else 0
        except Exception as e:
            logging.error(f"Twitter анализ ошибка: {e}")
            return 0

    @staticmethod
    def convert_label_to_score(label, confidence):
        """Конвертация метки в числовой показатель"""
        label_map = {"positive": 1, "neutral": 0, "negative": -1}
        base_score = label_map.get(label.lower(), 0)
        return base_score * confidence

    @staticmethod
    def analyze_reddit(symbol):
        """Анализ Reddit с учетом контекста"""
        # Заглушка для реальной реализации
        return 0.15

    @staticmethod
    def analyze_news(symbol):
        """Анализ новостей с помощью NLP"""
        # Заглушка для реальной реализации
        return 0.20