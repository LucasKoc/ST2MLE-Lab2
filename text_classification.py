import re
import string

import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config


class Text_Preprocessing():
    """
    Class for preprocessing text data exercise.

    For Spacy module, must run `python -m spacy download en_core_web_sm`
    """
    def __init__(self):
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
            random_state=42
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
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop and len(token) > 2
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
            random_state=Config.RANDOM_STATE
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

        
