import pandas as pd
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger


df = pd.read_csv('benchmark.csv', sep=';')


# Loading the spacy model, identifying the entities and replacing them with the entity type
def spacy_anonymize(text):
    spacy_model = spacy.load("en_core_web_sm")
    doc = spacy_model(text)
    anonymized_text = text
    for ent in reversed(doc.ents):  # Reverse to not mess up the offsets
        anonymized_text = anonymized_text[:ent.start_char] + f"<{ent.label_}>" + anonymized_text[ent.end_char:]
    return anonymized_text


# Loading the flair model, identifying the entities and replacing them with the entity type
def flair_anonymize(text):
    flair_model = SequenceTagger.load('ner')
    sentence = Sentence(text)
    flair_model.predict(sentence)
    anonymized_text = text
    entities = sentence.get_spans('ner')
    for ent in reversed(entities):
        start_pos = ent.tokens[0].start_position
        end_pos = ent.tokens[-1].end_position
        anonymized_text = anonymized_text[:start_pos] + f"<{ent.tag}>" + anonymized_text[end_pos:]

    return anonymized_text


df['spacy_anonymized'] = df['text'].apply(spacy_anonymize)
df['flair_anonymized'] = df['text'].apply(flair_anonymize)
df.to_csv('anonymized_dataset.csv', index=False, sep=';')
