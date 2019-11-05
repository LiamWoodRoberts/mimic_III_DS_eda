import pandas as pd
import re
import numpy as np
from tensorflow.keras.utils import to_categorical

def read_turks(file):
    '''Load in DataTurks formatted tsv file and output array of each entry'''
    with open(file) as f:
        lines = [i.rstrip().split("\t") for i in f.readlines()]
    return lines

def clean_words(word_ents):
    '''removes quote and comma characters from '''
    new_word_ents = []
    for ents in word_ents:
        word = ents[0]
        if word.find(',') > 0:
            word = word[word.find(',')+1:]
        word = word.replace('"','')
        ents[0] = word
        new_word_ents.append(ents)
    return new_word_ents

def create_seqs(word_ents):
    '''Formats word entities as sequences'''
    seqs = []
    seq = []
    for ents in word_ents:
        if len(ents)>1:
            seq.append(ents)
        else:
            seqs.append(seq)
            seq=[]
    return seqs

def add_iob_scheme(word_ents):
    '''adds IOB scheme to tags'''
    new_ents = []
    for i in range(0,len(word_ents)):
        if word_ents[i][1] == "O":
            tag = word_ents[i][1]
        else:
            if not i:
                tag = "B-"+word_ents[i][1]
            else:
                if (word_ents[i][1] != word_ents[i-1][1]):
                    tag = "B-"+word_ents[i][1]
                else:
                    tag = "I-"+word_ents[i][1]

        new_ents.append([word_ents[i][0],tag])
    return new_ents

def pad_seq(seq,max_len):
    '''pads or truncates a sequence to a specified length'''
    padded_seq = seq+[["<PAD>","O"]]*max_len
    return padded_seq[:max_len]
    
def pad_sequences(sequences,max_len=None):
    '''pads or truncates a list of sequences to a specified length'''
    if max_len == None:
        max_len = max(len(seq) for seq in sequences)
    return [pad_seq(seq,max_len) for seq in sequences]

def get_word_ids(sentances,tag=False):
    ''''Enumerates each unique word or tag in the dataset and creates 
        a mapping for each word/tag number pair'''

    words = []
    for sentance in sentances:
        words += list([word[tag] for word in sentance])
    word_dict = {word:i for i,word in enumerate(set(words))}
    return word_dict

def words_to_ids(sentances,word_ids,tag_ids):
    '''Maps each sequence to its word/tag id representation'''
    vector = []
    for sentance in sentances:
        vector.append(list([[word_ids[w[0]],tag_ids[w[1]]] for w in sentance]))
    return np.array(vector)

def create_x_y(matrix,n_tags):
    '''Formats sequences as features(x) and labels(y)'''
    x = []
    y = []
    for sequences in matrix:
        xi = [i[0] for i in sequences]
        yi = [i[1] for i in sequences]
        x.append(xi)
        y.append(yi)
    y = np.array([to_categorical(i,n_tags) for i in y])
    return np.array(x),y

def save_ids(word_ids,tag_ids,output_path,n_samples):
    np.save(f"{output_path}word_ids_{n_samples}.npy",word_ids)
    np.save(f"{output_path}tag_ids_{n_samples}.npy",tag_ids)
    print('-'*40)
    print("~~~Word/Tag Ids Successfully Saved~~~")
    return

def save_vals(x,y,file_path = "annotations/"):
    '''Saves x and y values to a specified location'''
    np.save(f"{file_path}x_data_{len(x)}.npy",x)
    np.save(f"{file_path}y_data_{len(y)}.npy",y)
    print("-"*40)
    print("~~~Values Successfully Saved~~~")
    print("-"*40)
    print("X-shape:",x.shape)
    print("Sample:")
    print(x[0][:5])
    print('-'*40)
    print("Y-shape:",y.shape)
    print("Sample:")
    print(y[0][:5])
    print("-"*40)
    return

def create_dataset(turks_file,output_filepath):
    word_ents = read_turks(turks_file)
    new_ents = clean_words(word_ents)
    seqs = create_seqs(new_ents)
    iob_tag_seqs = [add_iob_scheme(ents) for ents in seqs]
    padded_seqs = pad_sequences(iob_tag_seqs,max_len=50)
    word_ids = get_word_ids(padded_seqs)
    tag_ids = get_word_ids(padded_seqs,tag=True)
    vectors = words_to_ids(padded_seqs,word_ids,tag_ids)
    n_tags = len(tag_ids)
    x,y = create_x_y(vectors,n_tags)
    save_ids(word_ids,tag_ids,output_filepath,len(x))
    save_vals(x,y,output_filepath)
    return

if __name__ == "__main__":
    turks_file = "./turks_data/Medical NER Dataset 2600.tsv"
    output_filepath = "./annotations/"
    create_dataset(turks_file,output_filepath)
