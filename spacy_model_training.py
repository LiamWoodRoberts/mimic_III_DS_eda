import spacy
import random
import logging
import json
import time

# Taken From: https://dataturks.com/help/dataturks-ner-json-to-spacy-train.php

def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    ''' 
        Source: https://dataturks.com/help/dataturks-ner-json-to-spacy-train.php
        Load Dataturks Annotated Data File and Returns "Spacy Formatted" Text
    '''

    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))

            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

def train_spacy(TRAIN_DATA,epochs=10):
    ''' 
        Source: https://dataturks.com/help/dataturks-ner-json-to-spacy-train.php
        Accepts "Spacy Formatted" Training data and returns a trained Spacy NER model
    '''
    nlp = spacy.blank('en')  # create blank Language class
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    optimizer = nlp.begin_training()
    for itn in range(epochs):
        print(f"Starting iteration: {itn}..." )
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.2,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)
    with open('data/model_losses.csv','a') as fd:
        fd.write(f"{losses['ner']},{time.ctime()}")
    return nlp

if __name__ == "__main__":
    data = convert_dataturks_to_spacy("./data/annotated_data/MimicNER_0-20.json")
    nlp = train_spacy(data,epochs=20)
    text = "Review of systems is negative for the following  Fevers, chills, nausea, vomiting, night sweats, change in weight, gastrointestinal complaints, neurologic changes, rashes, palpitations, orthopnea"
    doc = nlp(text)
    print ("Entities= " + str(["" + str(ent.text) + "->" + str(ent.label_) for ent in doc.ents]))