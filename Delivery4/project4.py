import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import json
import os

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD

from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.layers import Embedding, LSTM, Dense, Dropout

df = pd.read_csv('deneme.csv')
label_map = {}
count = 0
for label in df.Mahkeme.unique():
    label_map[label] = count
    count += 1
df['NUM_LABEL'] = df.Mahkeme.map(label_map)
#print(label_map)
print(df.Mahkeme.value_counts())
def clean_text(text):
    alphabetic_only = [word for word in str(text).split() if word.isalpha()]
    lower_case_only = [word.lower() for word in alphabetic_only]
    stopwords_tr = set(stopwords.words("turkish"))
    return [word for word in lower_case_only if word not in stopwords_tr]

df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: " ".join(clean_text(x)))
df = df[['ictihat', 'NUM_LABEL']]
print(df)

X = df.ictihat
Y = df.NUM_LABEL
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=123)
freq_df = X_train.str.split(expand=True).stack().value_counts().reset_index()

freq_df.columns = ['Word', 'Frequency']
train_v_size = len(freq_df)
tokenizer = Tokenizer()
#for i in X_train:pr

# X_train = str(X_train).lower()
#print(X_train.shape)
tokenizer.fit_on_texts(X_train)
#get train sequences
train_seqs = tokenizer.texts_to_sequences(X_train)
train_seqs_max_size = max([len(seq) for seq in train_seqs])
#get test sequences
X_test = str(X_test).lower()
test_seqs = tokenizer.texts_to_sequences(X_test)
test_seqs_max_size = max([len(seq) for seq in test_seqs])
train_padded = tf.keras.utils.pad_sequences(train_seqs, maxlen=train_seqs_max_size, padding="post", truncating="post")
test_padded = tf.keras.utils.pad_sequences(test_seqs, maxlen=train_seqs_max_size, padding="post", truncating="post")
X_train_tokenized = [[word for word in document.split()] for document in X_train]
from gensim.models import Word2Vec, FastText
word_model = Word2Vec(X_train_tokenized, vector_size=100)

#build matrix
embedding_matrix_w2v = np.random.random(((train_v_size) + 1, 100))
for word,i in tokenizer.word_index.items():
    try:
        embedding_matrix_w2v[i] = word_model.wv[word]
    except:
        pass

# create layer
embedding_layer_w2v = Embedding((train_v_size) + 1, output_dim=100,
                            weights=[embedding_matrix_w2v], trainable=True)

ft = FastText(vector_size=300)
ft.build_vocab(X_train_tokenized)
ft.train(tokenizer.word_index, total_examples=ft.corpus_count, epochs=1)

# build matrix
embedding_matrix_ft = np.random.random(((train_v_size) + 1, ft.vector_size))
for word,i in tokenizer.word_index.items():
    try:
        embedding_matrix_ft[i] = ft.wv[word]
    except:
        pass

# create layer
embedding_layer_ft = Embedding((train_v_size) + 1, output_dim=300,
                            weights=[embedding_matrix_ft], trainable=True)

def lstm_model(embeddings, classification=True):
    model = Sequential()
    model.add(embeddings)
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(1, activation="sigmoid"))

    adam_opt = Adam(learning_rate=3e-4)
    if classification:
        model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=["accuracy"])
    else:
        model.compile(loss="mean_squared_error", optimizer=adam_opt, metrics=["mse"])

    return model


def train_model(model, train_padded, test_padded, y_train, y_test):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_padded, y_train, epochs=1,
                        validation_data=(test_padded, y_test), callbacks=[early_stopping])

    return history


def evaluate_model(model, test_padded, y_test):
    results = model.evaluate(test_padded, y_test, batch_size=128)
    return results


model = lstm_model(embedding_layer_w2v)
print(len(train_padded))
print(len(test_padded))
print(len(y_train))
print(len(train_padded))
history = train_model(model, train_padded, test_padded, y_train, y_test)
w2v_lstm_status_history = pd.DataFrame(history.history)
print(w2v_lstm_status_history)
print(evaluate_model(model, test_padded, y_test))
"""model = lstm_model(embedding_layer_ft)
history = train_model(model, train_padded, test_padded, y_train, y_test)
ft_lstm_status_history = pd.DataFrame(history.history)
print(ft_lstm_status_history)"""
"""
path_json_files = "C:/Users/husey/PycharmProjects/pythonProject12/IntToNLP_GRP_9/2021-01/"
ictihat_list = []
mahkeme_list = []

def get_text():
    docs_num = 0

    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:
        
        with open(path_json_files + file_name, encoding="utf8") as json_file:
            dict = json.load(json_file)
            mahkeme_text = dict["Mahkemesi"]
            ictihat_text = dict["ictihat"]
            ictihat_list.append(ictihat_text)
            if mahkeme_text == 'Asliye Ceza Mahkemesi':
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "ASLİYE HUKUK MAHKEMESİ" or mahkeme_text == 'Asliye Hukuk (Aile) Mahkemesi' or mahkeme_text == 'Konya Bölge Adliye Mahkemesi 2. Hukuk Dairesi':
                mahkeme_list.append("Asliye Hukuk Mahkemesi")
            elif mahkeme_text == "Sulh Hukuk Mahkemesi" or mahkeme_text == "Sulh Ceza Mahkemesi" or mahkeme_text == "KADASTRO MAHKEMESİ" or mahkeme_text == "İcra Hukuk Mahkemesi" or mahkeme_text == 'SULH HUKUK MAHKEMESİ' or mahkeme_text == 'TÜKETİCİ MAHKEMESİ':
                mahkeme_list.append("OTHERS")
            elif mahkeme_text == 'Asliye Hukuk (İş) Mahkemesi':
                mahkeme_list.append("İş Mahkemesi")
            elif mahkeme_text == 'TİCARET MAHKEMESİ' or mahkeme_text == "Ticaret Mahkemesi":
                mahkeme_list.append("Ticaret Mahkemesi")
            elif mahkeme_text == 'Aile Mahkemesi' or mahkeme_text == 'Çocuk Ağır Ceza Mahkemesi':
                mahkeme_list.append("Çocuk Mahkemesi")
            elif mahkeme_text == "BÖLGE ADLİYE MAHKEMESİ 1. HUKUK DAİRESİ" or mahkeme_text == "Bölge Adliye Mahkemesi 5. Hukuk Dairesi" or mahkeme_text == 'BÖLGE ADLİYE MAHKEMESİ 1. HUKUK DAİRESİ' or mahkeme_text == 'Bölge Adliye Mahkemesi 10. Hukuk Dairesi' or mahkeme_text == 'İstanbul Bölge Adliye Mahkemesi 10. Hukuk Dairesi' or mahkeme_text == 'Kayseri Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'Ankara Bölge Adliye Mahkemesi 1. Hukuk Dairesi' or mahkeme_text == 'Samsun Bölge Adliye Mahkemesi 4. Hukuk Dairesi' or mahkeme_text == 'Ankara Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'İstanbul Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'İstanbul Bölge Adliye Mahkemesi 38. Hukuk Dairesi' or mahkeme_text == 'İzmir Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'Antalya Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'Ankara Bölge Adliye Mahkemesi 28. Hukuk Dairesi' or mahkeme_text == 'Sakarya Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'Adana Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'Gaziantep Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'Bölge Adliye Mahkemesi 11. Hukuk Dairesi' or mahkeme_text == 'Bursa Bölge Adliye Mahkemesi 2. Hukuk Dairesi' or mahkeme_text == 'İstanbul Bölge Adliye Mahkemesi 11. Hukuk Dairesi' or mahkeme_text == 'Antalya Bölge Adliye Mahkemesi 5. Hukuk Dairesi' or mahkeme_text == 'Bölge Adliye Mahkemesi 4. Hukuk Dairesi' or mahkeme_text == 'İstanbul Bölge Adliye Mahkemesi 1. Hukuk Dairesi' or mahkeme_text == 'BÖLGE ADLİYE MAHKEMESİ 2. HUKUK DAİRESİ' or mahkeme_text == 'İzmir Bölge Adliye Mahkemesi 18. Hukuk Dairesi':
                mahkeme_list.append("Bölge Adliye Mahkemesi")
            elif mahkeme_text == "Ağır Ceza Mahkemesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "Asliye Hukuk Mahkemesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "Ceza Dairesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "İş Mahkemesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "Bölge Adliye Mahkemesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "Çocuk Mahkemesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "Ticaret Mahkemesi":
                mahkeme_list.append(mahkeme_text)
            elif mahkeme_text == "":
                mahkeme_list.append('EMPTY')
            else:
                mahkeme_list.append('OTHERS')
            #print(text)
            #if "Asliye Ceza Mahkemesi" in text or "asliye ceza mahkemesi" in text or "Asliye ceza mahkemesi" in text or "Asliye Ceza mahkemesi" in text:
            #    docs_num += 1

    #print(docs_num)


get_text()
#for i in mahkeme_list:
#    print(i)
#print(len(mahkeme_list))
#print(len(ictihat_list))
df = pd.DataFrame({'ictihat': ictihat_list, 'Mahkeme': mahkeme_list})
#mahkeme_values = pd.DataFrame([ictihat_list, mahkeme_list], columns=['ictihat', 'Mahkeme'])
df.to_csv('deneme.csv', index=False)
"""


