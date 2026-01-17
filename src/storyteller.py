import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset directly from ZIP
zip_path = "data/1000Folk_Story_around_the_Globe.csv.zip"
df = pd.read_csv(zip_path)

# Use only the main story text column
df = df[['full_text']].dropna()

# Basic text cleaning
df['full_text'] = df['full_text'].str.lower()

# Feature Engineering using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['full_text'])

# Dummy labels for unsupervised-style demo
labels = [0] * len(df)

# Train ML model
model = MultinomialNB()
model.fit(X, labels)

# Sample prediction
sample_story = ["A brave girl saves her village using wisdom and courage"]
sample_vector = vectorizer.transform(sample_story)
prediction = model.predict(sample_vector)

# Save output
with open("outputs/sample_output.txt", "w") as f:
    f.write("Smart Cultural Storyteller executed successfully.\n")
    f.write("Sample story processed by ML model.\n")

print("Model run completed. Output saved.")
