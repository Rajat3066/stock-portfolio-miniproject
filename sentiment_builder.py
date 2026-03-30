from news_finnhub import get_daily_sentiment
from sentiment import compute_sentiment


def build_sentiment_series(ticker, dates):
    sentiment_dict = {}

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        news = get_daily_sentiment(ticker, date_str, date_str)

        if not isinstance(news, list):
            sentiment_dict[date_str] = 0
            continue

        sentiment = compute_sentiment(news)
        sentiment_dict[date_str] = sentiment

    return sentiment_dict
