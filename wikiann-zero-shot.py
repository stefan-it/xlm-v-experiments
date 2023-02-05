import numpy as np
import sys

from flair.datasets import NER_MULTI_XTREME
from flair.models import SequenceTagger
from tabulate import tabulate

languages = ["ro", "gu", "pa", "lt", "az",
             "uk", "pl", "qu", "hu", "fi",
             "et", "tr", "kk", "zh", "my",
             "yo", "sw", "th", "ko", "ka",
             "ja", "ru", "bg", "es", "pt",
             "it", "fr", "fa", "ur", "mr",
             "hi", "bn", "el", "de", "en",
             "nl", "af", "te", "ta", "ml",
             "eu", "tl", "ms", "jv", "id",
             "vi", "he", "ar"]

language_mapping = {}

label_name_map = {"DATE": "O"}

for language in languages:
    language_mapping[language] = NER_MULTI_XTREME(languages=language)

model_names = sys.argv[1:]

print("Models:", model_names)

dev_table = []
test_table = []

for model_name in model_names:
    current_model = SequenceTagger.load(model_name)
    
    current_dev_entry = [model_name]
    current_test_entry = [model_name]
    
    for language in languages:
        current_corpus = language_mapping[language]
        
        dev_result = current_model.evaluate(current_corpus.dev, gold_label_type="ner",
                                            mini_batch_size=256).main_score
        test_result = current_model.evaluate(current_corpus.test, gold_label_type="ner",
                                             mini_batch_size=256).main_score
        
        dev_result = round(dev_result * 100, 1)
        test_result = round(test_result * 100, 1)
        current_dev_entry.append(dev_result)
        current_test_entry.append(test_result)
    
    dev_avg = round(np.mean(current_test_entry[1:]), 1)
    test_avg = round(np.mean(current_test_entry[1:]), 1)
    
    current_dev_entry.append(dev_avg)
    current_test_entry.append(test_avg)
    
    dev_table.append(current_dev_entry)
    test_table.append(current_test_entry)

# Calculate mean of language columns per language over all models
last_dev_row = ["Language Avg."]

for index, _ in enumerate(languages):
    dev_scores = []
    for row in dev_table:
        dev_scores.append(row[index + 1])
    last_dev_row.append(round(np.mean(dev_scores), 1))

last_dev_row.append(round(np.mean(last_dev_row[1:]), 1))
dev_table.append(last_dev_row)

# Same for test set
last_test_row = ["Language Avg."]

for index, _ in enumerate(languages):
    test_scores = []
    for row in test_table:
        test_scores.append(row[index + 1])
    last_test_row.append(round(np.mean(test_scores), 1))

last_test_row.append(round(np.mean(last_test_row[1:]), 1))
test_table.append(last_test_row)

# Final tables
headers = ["Model Name"] + languages + ["Avg."]

print("Development Results:")
print(tabulate(dev_table, headers=headers, tablefmt="github"))

print("")

print("Test Results:")
print(tabulate(test_table, headers=headers, tablefmt="github"))

