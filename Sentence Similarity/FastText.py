import numpy as np
import nltk
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fasttext.util

# Download FastText embeddings for English
fasttext.util.download_model('en', if_exists='ignore')

# Load the FastText model
ft = fasttext.load_model('cc.en.300.bin')

# Reduce dimension to 100
fasttext.util.reduce_model(ft, 100)

# Function to compute sentence embedding by averaging word embeddings
def get_sentence_embedding(sentence_tokens):
    embedding_dim = ft.get_dimension()
    sentence_embedding = np.zeros(embedding_dim)
    word_count = 0
    for word in sentence_tokens:
        sentence_embedding += ft.get_word_vector(word)
        word_count += 1
    if word_count > 0:
        sentence_embedding /= word_count
    return sentence_embedding

# Function to extract features for sentence similarity classification
def extract_features(sentence1, sentence2):
    features = []
    
    # Tokenize sentences
    sentence1_tokens = nltk.word_tokenize(sentence1.lower())
    sentence2_tokens = nltk.word_tokenize(sentence2.lower())
    
    # Compute sentence embeddings
    sentence1_embedding = get_sentence_embedding(sentence1_tokens)
    sentence2_embedding = get_sentence_embedding(sentence2_tokens)
    
    # Cosine similarity between sentence embeddings
    cos_sim = np.dot(sentence1_embedding, sentence2_embedding) / (np.linalg.norm(sentence1_embedding) * np.linalg.norm(sentence2_embedding))
    features.append(cos_sim)
    
    return features

# Load PAWS dataset
paws_dataset = load_dataset('paws', 'labeled_final')

# Extract features and labels
X = []
y = []
for example in paws_dataset['train']:
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    label = example["label"]
    features = extract_features(sentence1, sentence2)
    X.append(features)
    y.append(label)

# Split data into train/dev/test sets
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2, random_state=42)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
