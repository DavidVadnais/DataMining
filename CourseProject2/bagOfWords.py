from keras.preprocessing.text import Tokenizer
 
text = [
  'There was a man',
  'The man had a dog',
  'The dog and the man walked',
]
# using tokenizer 
model = Tokenizer()
model.fit_on_texts(text)
 
#print keys 
print(f'Key : {list(model.word_index.keys())}')
 
#create bag of words representation 
rep = model.texts_to_matrix(text, mode='count')
print(rep)
