import pandas as pd
import re
import numpy as np
import random
import time
from seqeval.metrics import f1_score,classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def load_x_y(file_path,n_samples):
    x = np.load(f"{file_path}x_data_{n_samples}.npy")
    y = np.load(f"{file_path}y_data_{n_samples}.npy")
    return x,y

def load_word_tag_mappings(file_path,n_samples):
    word_ids = np.load(f"{file_path}word_ids_{n_samples}.npy",
                        allow_pickle=True)
    tag_ids = np.load(f"{file_path}tag_ids_{n_samples}.npy",
                        allow_pickle=True)
    return word_ids.reshape(-1)[0],tag_ids.reshape(-1)[0]

def create_BiLSTM(n_words,n_tags,embedding_size,max_len):
    model = Sequential()
    model.add(Embedding(n_words,embedding_size,input_length=max_len))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
    return model

def train_model(model,x_train,y_train,embedding_size,batch_size=32,epochs=20,val_split = 0.1):
    checkpointer = ModelCheckpoint(f'BiLSTM_NER_Model_Embedding_Size-{embedding_size}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.0001,
                               patience=3,
                               mode='min',
                               verbose=1)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, 
                        batch_size=32, 
                        epochs=epochs, 
                        validation_split=val_split, 
                        verbose=1,
                        callbacks=[early_stop,checkpointer]
                       )
    return history

def get_id_mappings(ids):
    return {str(i[1]):i[0] for i in ids.items()}

def transform_ids_to_tags(preds,tag_ids):
    id_to_tags = get_id_mappings(tag_ids)

    tag_seqs = []
    for seq in preds:
        tag_seqs.append([id_to_tags[str(i)] for i in seq])
    return tag_seqs

def get_f1_score(model,x_test,y_test,tag_ids):
    test_preds = np.argmax(model.predict(x_test),axis=-1)
    true_vals = np.argmax(y_test,axis=-1)
    test_preds = transform_ids_to_tags(test_preds,tag_ids)
    true_vals = transform_ids_to_tags(true_vals,tag_ids)
    print(classification_report(true_vals,test_preds))
    return f1_score(true_vals,test_preds)

def append_model_results(model_f1,n_samples,model_desc,file):
    with open(file,'a') as f:
        results = f"\n{model_f1},{n_samples},{model_desc},{time.ctime()}"
        f.writelines(results)
    print("~~~Results Successfully Saved")
    return

if __name__ == "__main__":
    file_path = "./annotations/"
    n_samples = "2592"
    x,y = load_x_y(file_path,n_samples)
    word_ids,tag_ids = load_word_tag_mappings(file_path,n_samples)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    max_len = x_train.shape[1]
    embedding_size = 50
    n_words = len(word_ids)
    n_tags = len(tag_ids)
    model = create_BiLSTM(n_words,n_tags,embedding_size,max_len)
    epochs = 50
    batch_size = 32
    history = train_model(model,x_train,y_train,embedding_size,batch_size=32,epochs=epochs,val_split = 0.1)
    model_f1 = get_f1_score(model,x_test,y_test,tag_ids)
    print("F1-Score:",model_f1)
    model_desc = f"BiLSTM-EmbedSize-{embedding_size}"
    results_file = "./annotations/model_results.csv"
    append_model_results(model_f1,n_samples,model_desc,results_file)
    print(pd.read_csv(results_file).head())