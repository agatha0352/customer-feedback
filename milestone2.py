import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return sentiment, polarity


if __name__ == "__main__":
    df = pd.read_csv("Milestone1_Cleaned_Feedback.csv")
    
    df[["sentiment", "confidence_score"]] = df["clean_feedback"].apply(
        lambda x: pd.Series(get_sentiment(x))
    )
    
    df.to_csv("Milestone2_Sentiment_Result_new.csv", index=False)
    
    print("Milestone 2 completed successfully")
    
    sentiment_counts = df["sentiment"].value_counts()
    
    plt.figure(figsize=(8, 5))
    colours = ["green", "red", "gray"]
    sentiment_counts.plot(kind="bar", color=colours)
    
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Sentiment Distribution", fontsize=14)
    plt.xticks(rotation=0)
    
    for i, count in enumerate(sentiment_counts):
        plt.text(i, count + 20, str(count), ha="center", fontsize=10)
    
    plt.savefig("sentiment_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Bar chart done")
    print(df[["clean_feedback", "sentiment", "confidence_score"]].head())
