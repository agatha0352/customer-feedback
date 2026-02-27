

import pandas as pd
from collections import Counter
import re
def extract_keywords(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    words = text.split()
    return words

if __name__ == "__main__":


    df = pd.read_csv("Milestone2_Sentiment_Result_new.csv")

    all_words = []
    df["clean_feedback"].apply(lambda x: all_words.extend(extract_keywords(x)))


    keyword_freq = Counter(all_words)

    
    keywords_df = pd.DataFrame(
        keyword_freq.items(),
        columns=["Keyword", "Frequency"]
    ).sort_values(by="Frequency", ascending=False)

    
    keywords_df.to_csv("Milestone3_Keyword_Insights.csv", index=False)
    print(keywords_df.head(10))

    print("Keyword extraction completed successfully!")
