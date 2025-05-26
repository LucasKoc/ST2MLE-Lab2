import re
import string

import numpy as np
import pandas as pd
import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

from config import Config


class Text_Preprocessing:
    """
    Class for preprocessing text data exercise.

    For Spacy module, must run `python -m spacy download en_core_web_sm`
    """

    def __init__(self):
        self.model = None
        self.X_test_vectorized = None
        self.X_train_vectorized = None
        self.cleaned_data = None
        self.dataset = None

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        self.categories = [
            "comp.graphics",
            "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware",
            "comp.sys.mac.hardware",
            "comp.windows.x",
        ]

        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def load_dataset(self) -> None:
        """
        Import dataset from scikit-learn's fetch_20newsgroups() method.
        """
        self.dataset = fetch_20newsgroups(
            subset='all',
            categories=self.categories,
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=42,
        )

    def clean_text(self, text: str) -> str:
        """
        Clean the text data by removing stop words and punctuation.
        """
        # Lowercase the text and remove punctuation
        text = text.lower()
        text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)

        # Lemmatize the text using spaCy
        doc = self.nlp(text)
        tokens = [
            token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2
        ]
        return " ".join(tokens)

    def preprocess_texts(self) -> None:
        """
        Preprocess the text data by cleaning each document in the dataset.
        """
        # Show with tqdm progress bar
        cleaned_texts = [self.clean_text(doc) for doc in tqdm(self.dataset.data)]
        self.cleaned_data = cleaned_texts

    def split_data(self) -> None:
        """
        Split into training and testing sets (80/20).
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.cleaned_data,
            self.dataset.target,
            test_size=Config.TEST_SIZE,
            stratify=self.dataset.target,
            random_state=Config.RANDOM_STATE,
        )

        print("------------------------------")
        print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        print("------------------------------")

    def show_sample(self, count=2) -> None:
        """
        Print a few cleaned samples for verification.
        """
        for i in range(count):
            print(f"\n--- Sample {i+1} ---")
            print("Category:", self.categories[self.y_train[i]])
            print(self.X_train[i][:300], "...\n")

    def predict_and_evaluate(self, model) -> None:
        """
        Predict and evaluate the model on the test set.
        """
        if self.X_test_vectorized is None or self.y_test is None:
            raise ValueError("Data not vectorized or labels not available.")

        predictions = model.predict(self.X_test_vectorized)
        accuracy = (predictions == self.y_test).mean()
        print("------------------------------")
        print(f"Model accuracy: {accuracy:.4f}")
        print("------------------------------")

    def naive_bayes_model(self) -> None:
        """
        Train a Naive Bayes model on the training set and evaluate it on the test set.
        """
        if self.X_train_vectorized is None or self.y_train is None:
            raise ValueError("Data not vectorized or labels not available.")

        self.model = MultinomialNB()
        self.model.fit(self.X_train_vectorized, self.y_train)

        self.predict_and_evaluate(self.model)

    def bag_of_words(self) -> None:
        """
        Vectorize the text using Bag of Words.
        - Use CountVectorizer from scikit-learn. Remove stopwords.
        - Visualize the size of the vocabulary (number of unique words).
        """
        if self.cleaned_data is None:
            raise ValueError("Data not preprocessed.")

        vectorizer = CountVectorizer(stop_words='english')
        self.X_train_vectorized = vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = vectorizer.transform(self.X_test)

        print(f"Vocabulary size: {len(vectorizer.vocabulary_)} tokens")

    def tf_idf(self) -> None:
        """
        Vectorize the text using TF-IDF.
        - Use TfidfVectorizer from scikit-learn. Remove stopwords.
        - Compare the average TF-IDF values for the top 10 frequent terms.
        """
        if self.cleaned_data is None:
            raise ValueError("Data not preprocessed.")

        vectorizer = TfidfVectorizer(stop_words='english')
        self.X_train_vectorized = vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = vectorizer.transform(self.X_test)

        # Get the feature names and their average TF-IDF values
        avg_tfidf_scores = np.asarray(self.X_train_vectorized.mean(axis=0)).flatten()
        top_indices = avg_tfidf_scores.argsort()[::-1][:10]

        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        top_scores = avg_tfidf_scores[top_indices]

        print("\nTop 10 words by average TF-IDF score:")
        print(pd.DataFrame({"Term": top_words, "Avg TF-IDF": top_scores}))

