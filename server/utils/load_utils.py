import os
import sys
import pickle

from dotenv import load_dotenv


_ = load_dotenv()


def load_utils():
    MODEL_PATH = os.getenv("MODEL_PATH")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH")
    LABEL_ENC_PATH = os.getenv("LABEL_ENC_PATH")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        f.close()

    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f) 
        f.close()

    with open(LABEL_ENC_PATH, 'rb') as f:
        le = pickle.load(f) 
        f.close()

    return (model, vectorizer, le)
