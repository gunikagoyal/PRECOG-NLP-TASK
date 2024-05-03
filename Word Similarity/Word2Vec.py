from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset

print("Loading the dataset...")
# Load a shard of the dataset
dataset = load_dataset("sedthh/gutenberg_english", split="train[:10%]")  # Load the first 50% of the training split

# Dataset is loaded
print("Dataset loaded successfully.")
print("Number of rows in the loaded dataset:", len(dataset))

# Extract sentences from the dataset
print("Extracting sentences from the dataset...")
sentences = []
for i, text in enumerate(dataset["TEXT"], start=1):
    for sentence in text.split("."):
        sentences.append(sentence.split())
    if i % 500 == 0:
        print(f"{i} rows processed. Sentences extracted so far:", len(sentences))

print("Sentences extracted successfully. Total sentences:", len(sentences))

# Train Word2Vec model
print("Training Word2Vec model...")
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
print("Word2Vec model trained successfully.")

# Save the trained Word2Vec model
model.save("word2vec_model")

print("Word2Vec model saved successfully.")

# Load the word similarity dataset
print("Loading the word similarity dataset...")
test_data_path = "/Users/gunika/Documents/Precog/SimLex-999.txt"
test_data = pd.read_csv(test_data_path, sep='\t')
print("Word similarity dataset loaded successfully.")

# Predict similarity scores
print("Predicting similarity scores...")
predicted_scores = []
for idx, row in test_data.iterrows():
    word1 = row['word1']
    word2 = row['word2']

    if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
        similarity_score = cosine_similarity([model.wv[word1]], [model.wv[word2]])[0][0]
    else:
        similarity_score = 0  # Default similarity

    predicted_scores.append(similarity_score)
print("Similarity scores predicted successfully.")

# Evaluate using Spearman's rank correlation coefficient
print("Evaluating using Spearman's rank correlation coefficient...")
true_scores = test_data['SimLex999'].values
correlation = pd.Series(predicted_scores).corr(pd.Series(true_scores), method='spearman')
print("Evaluation completed.")

print("Spearman's Rank Correlation Coefficient:", correlation)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.regplot(x=true_scores, y=predicted_scores, scatter_kws={"alpha":0.5})
plt.title("True vs. Predicted Similarity Scores")
plt.xlabel("True Similarity Scores")
plt.ylabel("Predicted Similarity Scores")
plt.grid(True)
plt.show()

import numpy as np
# Calculate residuals
residuals = np.array(true_scores) - np.array(predicted_scores)

# Plotting the residual plot
plt.figure(figsize=(8, 6))
plt.scatter(true_scores, residuals, alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('True Similarity Scores')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()