import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


train_data = pd.read_csv('Tweets-train.csv')
##################################################################################
# Select only the 'airline_sentiment' and 'text' columns
selected_columns = ['airline_sentiment', 'text']
train_data = train_data[selected_columns]

# 1. Display only sentiment and text (just a head, otherwise too much data)
print(train_data.head())

##################################################################################
# 2. Randomly select 10 tweets for each sentiment
# Function to check criteria in text


def check_criteria(text, criteria):
    return any(item in text.lower() for item in criteria)


for sentiment in train_data['airline_sentiment'].unique():
    sentiment_tweets = train_data[train_data['airline_sentiment'] == sentiment]
    random_tweets = sentiment_tweets.sample(10, random_state=42)  # Set a random

    print(f"\nRandom Tweets for {sentiment.capitalize()} Sentiment:")
    for index, row in random_tweets.iterrows():
        # Check criteria
        has_reference = check_criteria(row['text'], ['@'])
        has_link = check_criteria(row['text'], ['http', 'https'])
        has_punctuation = re.search(r'[^\w\s]', row['text']) is not None
        has_emoticon = check_criteria(row['text'], [':)', ':(', ':-)', ':-(', ':D', ';)', ':/', ':|'])

        # Display the tweet and its characteristics
        print(f"Tweet: {row['text']}")
        print(f"Contains '@' Reference: {has_reference}")
        print(f"Contains Link: {has_link}")
        print(f"Contains Punctuation: {has_punctuation}")
        print(f"Contains Emoticon: {has_emoticon}")
        print('-' * 50)
##################################################################################
# 3. Function to clean observed tokens from tweet text


def clean_tweet(text):
    # Remove references with '@'
    text = re.sub(r'@\w+', '', text)

    # Remove links
    text = re.sub(r'http\S+|https\S+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove common emoticons
    text = re.sub(r'(:-?\))|(:-?\()|(;-\))|(:/)|(:\|)', '', text)

    return text.strip()


# Cleaning function
train_data['cleaned_text'] = train_data['text'].apply(clean_tweet)

# Display the header
train_data.to_csv('Tweets-train.csv', index=False)
print(train_data['cleaned_text'].head())


##################################################################################
# 4. Function to get the most common words for each sentiment
def most_common_words(sentiment, n=15):
    sentiment_data = train_data[train_data['airline_sentiment'] == sentiment]
    text = ' '.join(sentiment_data['cleaned_text'])

    # Tokenize and count words
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    word_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0])))

    # Get the most common n words
    common_words = word_counts.most_common(n)
    return common_words


# List down the most common 15 words for each sentiment
sentiments = train_data['airline_sentiment'].unique()
for sentiment in sentiments:
    common_words = most_common_words(sentiment)
    print(f"\nMost Common 15 Words for {sentiment.capitalize()} Sentiment:")
    for word, count in common_words:
        print(f"{word}: {count}")


##################################################################################
# 5. Select only the 'airline_sentiment' and 'cleaned_text' columns
selected_columns = ['airline_sentiment', 'cleaned_text']
train_data = train_data[selected_columns]


# Function to remove stopwords and get the most common words for each sentiment
def most_common_words_without_stopwords(sentiment, stop_words, n=15):
    sentiment_data = train_data[train_data['airline_sentiment'] == sentiment]
    text = ' '.join(sentiment_data['cleaned_text'])

    # Tokenize and count words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    word_counts = Counter(dict(zip(feature_names, X.toarray()[0])))

    # Get the most common n words
    common_words = word_counts.most_common(n)

    # Remove most common words (stopwords)
    text_without_common_words = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    return common_words, text_without_common_words


# Define stop words
stop_words = set(stopwords.words('english'))

# Apply stopwords removal and list down the most common 15 words for each sentiment
for sentiment in train_data['airline_sentiment'].unique():
    common_words, train_data[f'{sentiment}_text_without_common_words'] = most_common_words_without_stopwords(sentiment, stop_words)

    print(f"\nMost Common 15 Words (without stopwords) for {sentiment.capitalize()} Sentiment:")
    for word, count in common_words:
        print(f"{word}: {count}")

# Display the first few rows of the resulting DataFrame
print(train_data[['airline_sentiment', 'cleaned_text', 'negative_text_without_common_words', 'neutral_text_without_common_words', 'positive_text_without_common_words']].head())

##################################################################################
# 6. Words to remove
words_to_remove = ['americanair', 'united', 'delta', 'southwestair', 'jetblue', 'virginamerica', 'usairways', 'flight', 'plane']


# Function to remove specified words and get the most common words for each sentiment
def most_common_words_without_specified_words(sentiment, words_to_remove, n=15):
    sentiment_data = train_data[train_data['airline_sentiment'] == sentiment]
    text = ' '.join(sentiment_data['cleaned_text'])

    # Remove specified words
    text_without_specified_words = ' '.join([word for word in text.split() if word.lower() not in words_to_remove])

    # Tokenize and count words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text_without_specified_words])
    feature_names = vectorizer.get_feature_names_out()
    word_counts = Counter(dict(zip(feature_names, X.toarray()[0])))

    # Get the most common n words
    common_words = word_counts.most_common(n)

    return common_words, text_without_specified_words


for sentiment in train_data['airline_sentiment'].unique():
    common_words, train_data[f'{sentiment}_text_without_specified_words'] = most_common_words_without_specified_words(sentiment, words_to_remove)

    print(f"\nMost Common 15 Words (without specified words) for {sentiment.capitalize()} Sentiment:")
    for word, count in common_words:
        print(f"{word}: {count}")

# Display the first few rows of the resulting DataFrame
print(train_data[['airline_sentiment', 'cleaned_text', 'negative_text_without_specified_words', 'neutral_text_without_specified_words', 'positive_text_without_specified_words']].head())
# In my observation text became much cleaner and it's easy to see that sentiments match the actual text (see below)
# neutral	@VirginAmerica Are the hours of operation for the Club at SFO that are posted online current?	Are the hours of operation for the Club at SFO that are posted online current
# negative	@VirginAmerica help, left expensive headphones on flight 89 IAD to LAX today. Seat 2A. No one answering L&amp;F number at LAX!	help left expensive headphones on flight 89 IAD to LAX today Seat 2A No one answering LampF number at LAX
# negative	@VirginAmerica awaiting my return phone call, just would prefer to use your online self-service option :(	awaiting my return phone call just would prefer to use your online selfservice option

##################################################################################
# 7 ... 11. Encode sentiments using LabelEncoder
# Read the training data
train_data = pd.read_csv('Tweets-train.csv')

# Encode sentiments using LabelEncoder
label_encoder = LabelEncoder()
train_data['encoded_sentiment'] = label_encoder.fit_transform(train_data['airline_sentiment'])

# Vectorize the 'text' column using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_data['text'])
y = train_data['encoded_sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Read the test data
test_data = pd.read_csv('Tweets-test.csv')

# Clean the 'text' column in the test data
test_data['cleaned_text'] = test_data['text'].apply(clean_tweet)

# Clean and encode sentiments for test data
test_data['encoded_sentiment'] = label_encoder.transform(test_data['airline_sentiment'])

# Vectorize the 'text' column using the already fitted vectorizer
X_test = vectorizer.transform(test_data['cleaned_text'])

# Predict sentiments for test data
y_pred_test = model.predict(X_test)

# Print and explain the Confusion Matrix
conf_matrix = confusion_matrix(test_data['encoded_sentiment'], y_pred_test)
print("Confusion Matrix:")
print(conf_matrix)

# Compute Accuracy of the model on the test data
accuracy_test = accuracy_score(test_data['encoded_sentiment'], y_pred_test)
print(f"Accuracy on Test Data: {accuracy_test:.2f}")

# Display the Classification Report
classification_report_result = classification_report(test_data['encoded_sentiment'], y_pred_test, target_names=label_encoder.classes_)
print("Classification Report:\n", classification_report_result)


# True Negatives (TN): 2388

# This is the count of instances where the actual class is negative (e.g., 'negative') and the model correctly predicted it as negative.
# False Positives (FP): 75

# This is the count of instances where the actual class is negative, but the model predicted a different class (either neutral or positive).
# False Negatives (FN): 536

# This is the count of instances where the actual class is either neutral or positive, but the model predicted it as negative.
# True Positives (TP): 315
# This is the count of instances where the actual class is either neutral or positive, and the model correctly predicted it as neutral or positive.

# Accuracy on Test Data: 0.72
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.72      0.97      0.82      2508
#      neutral       0.73      0.14      0.23       851
#     positive       0.73      0.52      0.61       641
#
#     accuracy                           0.72      4000
#    macro avg       0.73      0.54      0.56      4000
# weighted avg       0.72      0.72      0.66      4000
