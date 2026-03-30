import requests

API_KEY = "d738i9pr01qjjol24jjgd738i9pr01qjjol24jk0"


def get_daily_sentiment(ticker, start_date, end_date):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start_date,
        "to": end_date,
        "token": API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        print(f"News fetch error for {ticker} on {start_date}: {e}")
        return []
