# Sentiment-Analysis 

Description:
This repository contains a Python script for sentiment analysis on a dataset of vaccination-related tweets. The analysis uses the NLTK Vader library and includes data preprocessing, visualisation of sentiment distributions, and a summary bar plot of total sentiment scores.

Libaries used:
The script uses numpy, pandas, re, nltk, matplotlib, seaborn, and plotly

Data Preprocessing:
The script performs the following preprocessing steps on the tweet data:
  Removal of Twitter handles
  Removal of hashtags
  Removal of URLs
  Removal of special characters
  Removal of single characters
  Substitution of multiple spaces with a single space

Sentiment Analysis:
Sentiment analysis is conducted using the NLTK Vader library. The script creates new columns for positive, neutral, and negative sentiment scores, and a small constant is added to each score to avoid division by zero.

Visualisations:
The script generates two types of visualizations:
  Kernel Density Plots: These plots illustrate the distribution of negative, positive, and neutral sentiment scores across the dataset.
  Bar Plot: This plot provides a summary of the total sentiment scores for positive, neutral, and negative sentiments.
