import json
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from zemberek import TurkishMorphology

# you need to run this if this is first time trying import stopwords.
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

contents = []
path_json_files = "C:/Users/husey/PycharmProjects/pythonProject10/2021-01/"
pos_tagger = TurkishMorphology.create_with_defaults()
results = []


# read json files
def get_text():
    docs_num = 0
    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:

        docs_num += 1
        if docs_num == 10000:
            break

        with open(path_json_files + file_name, encoding="utf8") as json_file:
            dict = json.load(json_file)
            text = dict["ictihat"]
            contents.extend(clean_text(text))


# preprocessing text.
def clean_text(text):
    alphabetic_only = [word for word in text.split() if word.isalpha()]
    lower_case_only = [word.lower() for word in alphabetic_only]
    stopwords_tr = set(stopwords.words("turkish"))
    return [word for word in lower_case_only if word not in stopwords_tr]


# find pos tag of word
def get_word_tag(word):
    tags = pos_tagger.analyze(word.strip())
    if len(tags.analysis_results) > 0:
        return tags.analysis_results[0].item.primary_pos.value
    return ''


# find collocation pos tags.
def collocation_tags(trigram_with_freq):
    collocation_with_tags = []
    for collocation in trigram_with_freq:
        freq_of_collocation = trigram_with_freq[collocation]
        pos_tag_collocation = get_word_tag(collocation[0]) + get_word_tag(collocation[1]) + get_word_tag(collocation[2])
        tuple_collocation = collocation, freq_of_collocation, pos_tag_collocation
        collocation_with_tags.append(tuple_collocation)
    return collocation_with_tags


# filter collocations by pos
def pos_filter_collocations(tagged_collocations):
    pos_tags = ["AdjAdjNoun", "AdjNounNoun", "NounAdjNoun", "NounNounNoun"]
    filtered_collocations = []
    for collocation in tagged_collocations:
        collocation_tag = collocation[2]
        if collocation_tag in pos_tags:
            filtered_collocations.append(collocation)
    return filtered_collocations


# show results
def print_result(filtered_collocations):
    for i in range(10):
        print(filtered_collocations[i])


def create_frequency_list(n_grams_list):
    frequency_list = []
    for i in n_grams_list:
        frequency_list.append(i[1])
    # print(frequency_list)
    return frequency_list


def calculate_mean(frequency_list):
    return np.mean(frequency_list)


def calculate_variance(frequency_list):
    return np.var(frequency_list)


def main():
    get_text()
    trigrams = nltk.trigrams(contents)
    token_freq = nltk.FreqDist(trigrams)
    tagged_collocations = collocation_tags(token_freq)
    filtered_collocations = pos_filter_collocations(tagged_collocations)
    created_frequency_list = create_frequency_list(filtered_collocations)
    mean = calculate_mean(created_frequency_list)
    variance = calculate_variance(created_frequency_list)
    print_result(filtered_collocations)
    print_result(filtered_collocations)
    print(mean, variance)


if __name__ == '__main__':
    main()
