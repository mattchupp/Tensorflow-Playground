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


# Sentence data
sentences = [
  'I love my dog',
  'I love my cat',
  'You love my dog!',
  'Do you think my dog is amazing?'
]

# other test data
test_data = [
  'i really love my dog',
  'my dog loves my manatee'
]

# tokenize the words
# if tokenizer finds word that doesn't fit, replaces with <OOV>
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

# test sequence from test_data
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)


# add padding to sequences for sentence length
# can set padding to post with padding='post'
# can also set a max length of sentence with maxlen=5
padded = pad_sequences(sequences)


# print index of words and squence
print(word_index)
print(sequences)
print(padded)

