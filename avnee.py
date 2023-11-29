# Import necessary libraries
import numpy as np 
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as ex

# Import data file
data = pd.read_csv('vaccination_tweets.csv')
data.head(5)

# Preprocess data
# Remove Twitter handles
data.text = data.text.apply(lambda x:re.sub('@[^\s]+','',x))

# Remove hashtags
data.text = data.text.apply(lambda x:re.sub(r'\B#\S+','',x))

# Remove URLs
data.text = data.text.apply(lambda x:re.sub(r"http\S+", "", x))

# Remove special characters
data.text = data.text.apply(lambda x:' '.join(re.findall(r'\w+', x)))

# Remove single characters
data.text = data.text.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

# Substitute multiple spaces with a single space
data.text = data.text.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))

# Import SIA from the NLTK Vader library
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Create a new column to store sentiment scores with original text data
data['sentiments']           = data['text'].apply(lambda x: sia.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))
data['Positive Sentiment']   = data['sentiments'].apply(lambda x: x['pos']+1*(10**-6)) 
data['Neutral Sentiment']    = data['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
data['Negative Sentiment']   = data['sentiments'].apply(lambda x: x['neg']+1*(10**-6))

# Remove the 'sentiments' column
data.drop(columns=['sentiments'],inplace=True)

# Generate three subplots representing the distribution of sentiments across the dataset
plt.subplot(2,1,1)
plt.title('Distriubtion Of Sentiments Across The Tweets',fontsize=19,fontweight='bold')
sns.kdeplot(data['Negative Sentiment'],bw=0.1)
sns.kdeplot(data['Positive Sentiment'],bw=0.1)
sns.kdeplot(data['Neutral Sentiment'],bw=0.1)
plt.show()

# Generate a barplot representing total sentiment scores 
plt.figure(figsize=(10, 6))
data[['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment']].sum().plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Total Sentiment Score')
plt.xticks(rotation=0)
plt.show()
