from section_parse import run
import re
import json
import pandas as pd

def clean_text(text):
    bad_chars = [":","*"]
    space_chars = ["[","]","(",")","\n"]
    for c in bad_chars:
        text = text.replace(c,"")
    for c in space_chars:
        text = text.replace(c," ")
    return text

def sections_to_sentances(sections):
    sentances = []
    for section in sections:
        section = clean_text(section)
        sentances += [i.lstrip() for i in re.split("\. ",section) if len(i)>0]
    return sentances

def get_sentances(sections,start,samples,min_len=3):
    sections = medication_sections[start:start+n]
    seqs = sections_to_sentances(sections)
    seqs = [i for i in seqs if len(i)>min_len]
    return seqs

def write_sentances_to_txt(filename,seqs):
    with open(filename,"w") as f:
        f.writelines(f"{line}\n" for line in seqs)
    return print("~~~File Saved~~~")

def create_and_save_df(sections,start,n,foldername="nlp_data"):
    seqs = get_sentances(sections,start,n)
    df = pd.DataFrame()
    df["TEXT"] = seqs
    df.to_csv(f"{foldername}/sentances_{start}-{start+n}.csv",index=False)
    print("~~~File Saved Successfully")
    return df

if __name__ == "__main__":
    # Get Sections for Section Parse Module
    title = "HISTORY OF PRESENT ILLNESS"
    medication_sections = run(title)
    medication_sections = [i for i in medication_sections if i != "NOT FOUND"]

    # Convert Sections --> Sentances for NER
    start = 150
    n = 50
    seqs = get_sentances(medication_sections,start,n)
    
    print("Sentance Examples:\n")
    [print(f"{i} - ",seqs[i],"\n") for i in range(5)]
    
    df = create_and_save_df(medication_sections,start,n,foldername="data")