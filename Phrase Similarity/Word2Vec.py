import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model
import gensim.downloader as api
from datasets import load_dataset

# Load Word2Vec embeddings
def load_word2vec_embeddings():
    # Load pre-trained Word2Vec embeddings
    word2vec_model = api.load("word2vec-google-news-300")
    return word2vec_model

# Load dataset from Hugging Face
def load_dataset_from_huggingface():
    dataset = load_dataset("PiC/phrase_similarity")
    return pd.DataFrame(dataset['test'])

# Function to convert a phrase to its vector representation using Word2Vec embeddings
def phrase_to_vector(phrase, word_embeddings):
    words = phrase.split()
    vector = np.zeros_like(word_embeddings['word'])
    for word in words:
        if word in word_embeddings:
            vector += word_embeddings[word]
    return vector / len(words)

# Prepare data
def prepare_data(dataset, word_embeddings):
    phrase1 = dataset['phrase1']
    phrase2 = dataset['phrase2']
    similarity_score = dataset['label']
    
    X1 = np.array([phrase_to_vector(phrase, word_embeddings) for phrase in phrase1])
    X2 = np.array([phrase_to_vector(phrase, word_embeddings) for phrase in phrase2])
    y = similarity_score.values
    return X1, X2, y

# Define neural network model
def create_model(input_dim):
    input1 = Input(shape=(input_dim,))
    input2 = Input(shape=(input_dim,))
    
    # Define individual branches for each input
    branch1 = Dense(128, activation='relu')(input1)
    branch2 = Dense(128, activation='relu')(input2)
    
    # Concatenate the outputs of the two branches
    merged = Concatenate()([branch1, branch2])
    
    # Additional layers
    merged = Dropout(0.2)(merged)
    merged = Dense(64, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)
    
    # Define the model
    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function
def main():
    # Load Word2Vec embeddings
    word2vec_embeddings = load_word2vec_embeddings()
    
    # Load dataset from Hugging Face
    dataset = load_dataset_from_huggingface()
    
    # Prepare data
    X1, X2, y = prepare_data(dataset, word2vec_embeddings)
    
    # Train-test split
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
    
    # Define model
    input_dim = X1_train.shape[1]
    model = create_model(input_dim)
    
    # Train model
    model.fit(x=[X1_train, X2_train], y=y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    # Evaluate model
    y_pred = model.predict([X1_test, X2_test])
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()