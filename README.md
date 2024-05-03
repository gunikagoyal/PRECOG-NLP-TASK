# PRECOG-NLP-TASK

# Word, Phrase, and Sentence Similarity Prediction

This repository contains code for predicting similarity scores between words, phrases, and sentences using various methodologies and models.
## Directory Structure

- `notebooks/`: Contains Jupyter notebooks for each prediction task along with results analysis.
  - `word_similarity_prediction/`: Notebooks for word similarity prediction.
    - `Word2Vec.ipynb`
    - `spaCy.ipynb`
    - `GloVe_ELMo.ipynb`
  - `phrase_similarity_prediction/`: Notebooks for phrase similarity prediction.
    - `Glove_LSTM_TF_IDF.ipynb`: Notebook for Siamese Word2Vec model.
    - `Word2Vec.ipynb`: Notebook for Siamese GloVe model.
  - `sentence_similarity_prediction/`: Notebooks for sentence similarity prediction.
    - `FastText.ipynb`: Notebook for FastText with Logistic Regression model.
    - `spaCy.ipynb`: Notebook for spaCy linguistic features.
- `data/`: Data directories for different datasets used in the experiments.
  - `simlex999/`
- `requirements.txt`: List of dependencies needed to run the project.
