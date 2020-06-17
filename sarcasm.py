import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# open sarcasm.json file
with open("sarcasm.json", 'r') as f: 
  datastore = json.load(f)

# map through file and store items in each array
sentences = []
labels = []
urls = []
for item in datastore: 
  sentences.append(item['headline'])
  labels.append(item['is_sarcastic'])
  urls.append(item['article_link'])

