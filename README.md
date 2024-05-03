# PRECOG-NLP-TASK

# Word, Phrase, and Sentence Similarity Prediction

This repository contains code for predicting similarity scores between words, phrases, and sentences using various methodologies and models.
## Directory Structure

- `notebooks/`: Contains Jupyter notebooks for each prediction task along with results analysis.
  - `word_similarity_prediction/`: Notebooks for word similarity prediction.
    - `Word2Vec.ipynb`: Notebook for Word2Vec model.
    - `spaCy.ipynb`: Notebook for spaCy model.
    - `ELMo.ipynb`: Notebook for ELMo model.
    - `GloVe.ipynb`: Notebook for GloVe model.
    - `results_analysis.ipynb`: Notebook for analyzing results.
  - `phrase_similarity_prediction/`: Notebooks for phrase similarity prediction.
    - `Siamese_Word2Vec.ipynb`: Notebook for Siamese Word2Vec model.
    - `Siamese_GloVe.ipynb`: Notebook for Siamese GloVe model.
    - `LSTM_Networks.ipynb`: Notebook for LSTM networks.
    - `TF-IDF_Random_Forest.ipynb`: Notebook for TF-IDF with Random Forest model.
    - `results_analysis.ipynb`: Notebook for analyzing results.
  - `sentence_similarity_prediction/`: Notebooks for sentence similarity prediction.
    - `FastText_Logistic_Regression.ipynb`: Notebook for FastText with Logistic Regression model.
    - `spaCy_Linguistic_Features.ipynb`: Notebook for spaCy linguistic features.
    - `BERT_based_Model.ipynb`: Notebook for BERT-based model.
    - `results_analysis.ipynb`: Notebook for analyzing results.
- `data/`: Data directories for different datasets used in the experiments.
  - `gutenberg/`
  - `simlex999/`
  - `pic/`
  - `paws/`
- `models/`: Saved models for different tasks and methodologies.
  - `word2vec/`
  - `glove/`
  - `elmo/`
  - `siamese_word2vec/`
  - `siamese_glove/`
  - `lstm/`
  - `tfidf_rf/`
  - `bert/`
- `requirements.txt`: List of dependencies needed to run the project.
- `LICENSE`: License information for the project.
