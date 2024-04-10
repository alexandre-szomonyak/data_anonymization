import time
import pandas as pd
from script import spacy_anonymize, flair_anonymize
import re
import logging

logging.basicConfig(level=logging.WARNING)

df = pd.read_csv('anonymized_dataset.csv', sep=';')


def count_anonymized_entities(text):
    pattern = re.compile(r'<[^>]+>')
    matches = pattern.findall(text)
    return len(matches)


label_counts = df['label'].apply(count_anonymized_entities).sum()
spacy_counts = df['spacy_anonymized'].apply(count_anonymized_entities).sum()
flair_counts = df['flair_anonymized'].apply(count_anonymized_entities).sum()
spacy_accuracy = spacy_counts / label_counts
flair_accuracy = flair_counts / label_counts


def benchmark_speed(anonymize_function):
    start_time = time.time()
    df['text'].apply(anonymize_function)
    return time.time() - start_time


spacy_speed = benchmark_speed(spacy_anonymize)
flair_speed = benchmark_speed(flair_anonymize)


print(f"Spacy - Accuracy: {spacy_accuracy.mean():.4f}, Execution Time: {spacy_speed:.2f} seconds")
print(f"Flair - Accuracy: {flair_accuracy.mean():.4f}, Execution Time: {flair_speed:.2f} seconds")
