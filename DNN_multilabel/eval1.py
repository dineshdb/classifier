import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf


categories = ['cart', 'explore_items', 'checkout', 'show categories']
# reset underlying graph data
tf.reset_default_graph()


# Build neural network
net = tflearn.input_data(shape=[None, 135])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 4, activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load("DNN_multilabel/trainedModel/model.tflearn")

# initialize the stemmer
stemmer = LancasterStemmer()
words = ['>', 'a', 'about', 'access', 'accompl', 'again', 'al', 'alright', 'am', 'and', 'around', 'at', 'bibshort', 'bik', 'bottl', 'bottom', 'bracket', 'brak', 'buy', 'cag', 'can', 'cap', 'cart', 'categ', 'chain', 'check', 'checkout', 'choos', 'cle', 'cloth', 'complet', 'compon', 'crankset', 'definit', 'derail', 'do', 'don', 'ens', 'fend', 'fin', 'fix', 'footbal', 'for', 'fork', 'fram', 'glov', 'good', 'got', 'handleb', 'hav', 'headset', 'helmet', 'hey', 'hydr', 'i', 'im', 'in', 'interest', 'iphon', 'iâ€™d', 'jersey', 'kind', 'laptop', 'lead', 'let', 'light', 'lik', 'lock', 'look', 'lookup', 'may', 'mayb', 'me', 'mountain', 'mov', 'my', 'myself', 'nee', 'of', 'ok', 'oth', 'out', 'pack', 'panny', 'ped', 'peep', 'pelas', 'pleas', 'plesa', 'produc', 'pump', 'rack', 'ready', 'real', 'rear', 'road', 'saddl', 'search', 'see', 'sel', 'select', 'set', 'shirt', 'sho', 'shop', 'short', 'show', 'so', 'sock', 'som', 'stand', 'stor', 'tel', 'that', 'the', 'think', 'tight', 'tir', 'to', 'top', 'tour', 'tub', 'u', 'vest', 'want', 'what', 'whatï½s', 'wheel', 'wil', 'wish', 'with', 'would', 'yo', 'you', 'youv']
def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


