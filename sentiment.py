from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def compute_sentiment(news_list):
    scores = []

    if not isinstance(news_list, list):
        return 0

    for article in news_list:
        if not isinstance(article, dict):
            continue
        headline = article.get("headline", "")
        if headline:
            score = sia.polarity_scores(headline)["compound"]
            scores.append(score)

    if len(scores) == 0:
        return 0

    return sum(scores) / len(scores)
