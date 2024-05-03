import numpy as np
import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load English language model in spaCy
print("Loading spaCy English language model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy English language model loaded successfully.")

# Function to extract linguistic features (PoS tags and dependency parse) from a sentence
def extract_linguistic_features(sentence):
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc]
    dep_parse = [token.dep_ for token in doc]
    
    # Combine features into a single list
    features = pos_tags + dep_parse
    
    return features

# Load PAWS dataset
print("\nLoading PAWS dataset...")
paws_dataset = load_dataset('paws', 'labeled_final')
print("PAWS dataset loaded successfully.")

# Extract features and labels
print("\nExtracting features and labels...")
X = []
y = []
for example in paws_dataset["train"]:
    sentence1 = example["sentence1"].lower()
    sentence2 = example["sentence2"].lower()
    features1 = extract_linguistic_features(sentence1)
    features2 = extract_linguistic_features(sentence2)
    features = features1 + features2  # Concatenate features for both sentences
    X.append(features)
    y.append(example["label"])
print("Features and labels extracted successfully.")

# Perform one-hot encoding for the linguistic features
print("\nPerforming one-hot encoding for the linguistic features...")
unique_pos_tags = set()
unique_dep_parse = set()
for seq in X:
    unique_pos_tags.update(seq[:len(seq)//2])  # First half contains PoS tags
    unique_dep_parse.update(seq[len(seq)//2:])  # Second half contains dependency parse

# Create dictionaries to map unique values to indices
pos_tag_to_index = {tag: i for i, tag in enumerate(sorted(unique_pos_tags))}
dep_parse_to_index = {parse: i for i, parse in enumerate(sorted(unique_dep_parse))}

# Replace values with indices in the sequences
X_encoded = []
for seq in X:
    pos_tags_encoded = [pos_tag_to_index[tag] for tag in seq[:len(seq)//2]]
    dep_parse_encoded = [dep_parse_to_index[parse] for parse in seq[len(seq)//2:]]
    encoded_seq = pos_tags_encoded + dep_parse_encoded
    X_encoded.append(encoded_seq)
print("One-hot encoding completed successfully.")

# Pad sequences to a fixed length
max_length = max(len(seq) for seq in X_encoded)
X_padded = pad_sequences(X_encoded, maxlen=max_length, padding='post', truncating='post', value=0)

# Convert features and labels to numpy arrays
X_final = np.array(X_padded)
y_final = np.array(y)

# Split data into train/dev/test sets
print("\nSplitting data into train/dev/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
print("Data split successfully.")

# Train classifier
print("\nTraining classifier...")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print("Classifier trained successfully.")

# Evaluate classifier
print("\nEvaluating classifier...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

