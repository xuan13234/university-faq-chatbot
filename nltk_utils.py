import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Simple tokenizer: split sentence into lowercase words by spaces.
    Avoids NLTK punkt errors.
    Example:
    "Hello, how are you?" -> ["hello,", "how", "are", "you?"]
    """
    return sentence.lower().split()

def stem(word):
    """
    Stemming = reduce word to its root form.
    Example:
    ["organize", "organizes", "organizing"] -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag    = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0 ]
    """
    # Stem each token
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Initialize bag with 0 for each word in vocabulary
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
