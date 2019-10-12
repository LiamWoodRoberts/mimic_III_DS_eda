# Mimic III EDA and Model Development


### Summary:
This repo takes a look at developing an NER tool tailored to a medical text environment. Specifically the project looks at NER for discharge summaries within the Mimic III dataset.

This project is succeeded by the mimic_III_api repo which deploys some of the services developed in this repo.

### The Dataset:
"MIMIC is an openly available dataset developed by the MIT Lab for Computational Physiology, comprising deidentified health data associated with ~60,000 intensive care unit admissions. It includes demographics, vital signs, laboratory tests, medications, and more." [1]

For more information you can check out the Mimic III website:

[1] https://mimic.physionet.org/

### File Summaries:

#### Jupyter Notebooks / Walkthrough:
- **1. Medical NER Tagger Walkthrough.ipynb:** An initial walkthrough showing the limitations of other pre-trained models and methods for heuristic tagging.

- **2. Hybrid NER Model.ipynb:** A concise walkthrough of how to develop a hybrid model which combines a pre trained model for transferable entities (dates,values,percents,etc) and a heuristic model for medical specific entities (drugs,symptoms, conditions).

- **3. Encoding Text as Vectors.ipynb:** A notebook covering how data can be formatted for use in sequence-to-sequence models.

- **4. api_testing(ongoing).ipynb:** A notebook for testing api routes for mimic_III_api which succeeded this repo

#### Python Scripts:
- **section_parse.py:** Script for parsing discharge summary sections

- **utils.py:** Script for collecting some information about title sections and popular words in discharge summaries
- **create_sentance_dataset.py:** Parses discharge summaries into sentances for later annotation.

#### Datafiles:
- **entities.csv:** A collection of all medical entities currently being used for tagging, is updated regularily with new tags.
- **condition_entities.npy:** Old collection of known medical conditions.
- **dose_entities.npy:** Old Collection of known dose entities.
- **drug_entities.npy:** Old collection of known drug entities
- **route_entities.npy:** Old collection of known medical routes.
- **unannotated.txt:** Output of create_sentance_dataset.py provides a txt document formatted for docanno annotation applications.
