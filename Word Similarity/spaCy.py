import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Load word similarity dataset
test_data_path = "/Users/gunika/Documents/Precog/SimLex-999.txt"
test_data = pd.read_csv(test_data_path, sep='\t')

# Function to obtain word embeddings for a word
def word_embeddings(word):
    return nlp(word).vector.reshape(1, -1)

# Tokenize words and obtain word embeddings
word_embeddings_list = []
for _, row in test_data.iterrows():
    word1 = row['word1']
    word2 = row['word2']

    # Obtain word embeddings for the words
    embedding1 = word_embeddings(word1)
    embedding2 = word_embeddings(word2)

    # Compute cosine similarity between word embeddings
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    word_embeddings_list.append(similarity_score)

# Evaluate using Spearman's rank correlation coefficient
true_scores = test_data['SimLex999'].values
correlation = pd.Series(word_embeddings_list).corr(pd.Series(true_scores), method='spearman')

print("Spearman's Rank Correlation Coefficient using spaCy word embeddings:", correlation)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualization graph
plt.figure(figsize=(8, 6))
sns.regplot(x=true_scores, y=word_embeddings_list, scatter_kws={"alpha":0.5})
plt.title("True vs. Predicted Similarity Scores")
plt.xlabel("True Similarity Scores")
plt.ylabel("Predicted Similarity Scores")
plt.grid(True)
plt.show()

# Distribution graph
plt.figure(figsize=(8, 6))
sns.histplot(word_embeddings_list, bins=20, kde=True)
plt.title("Distribution of Predicted Similarity Scores")
plt.xlabel("Predicted Similarity Scores")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
