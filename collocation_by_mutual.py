import json
import os
import numpy as np
import nltk
import math
from nltk.corpus import stopwords
from zemberek import TurkishMorphology

# you need to run this if this is first time trying import stopwords.
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

contents = []
path_json_files = "C:/Users/hasan/PycharmProjects/CSE4095S22_GRP_9/2021-01/"
pos_tagger = TurkishMorphology.create_with_defaults()
results = []


# read json files
def get_text():
    #docs_num = 0

    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:

        #docs_num += 1
        #if docs_num == 400:
            #break

        with open(path_json_files + file_name, encoding="utf8") as json_file:
            dict = json.load(json_file)
            text = dict["ictihat"]
            # print(text)
            contents.extend(clean_text(text))
    return contents

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
def collocation_tags(bigram_with_freq):
    collocation_with_tags = []
    for collocation in bigram_with_freq:
        freq_of_collocation = bigram_with_freq[collocation]
        pos_tag_collocation = get_word_tag(collocation[0])
        tuple_collocation = collocation, freq_of_collocation, pos_tag_collocation
        collocation_with_tags.append(tuple_collocation)
    return collocation_with_tags


# filter collocations by pos
def pos_filter_collocations(tagged_collocations):
    pos_tags = ["AdjNoun", "NounNoun"]
    filtered_collocations = []
    for collocation in tagged_collocations:
        collocation_tag = collocation[1]
        if collocation_tag in pos_tags:
            filtered_collocations.append(collocation)
    return filtered_collocations


# show results
def print_result(filtered_collocations):
    for i in range(100):
        print(filtered_collocations[i])


def create_frequency_list(n_grams_list):
    frequency_list = []
    total = 0
    for i in n_grams_list:
        frequency_list.append(i[1])
        total += i[1]
    # print(frequency_list)
    print(f'total: {total}')
    return frequency_list


def calculate_mean(frequency_list):
    return np.mean(frequency_list)


def calculate_variance(frequency_list):
    return np.var(frequency_list)


def calculate_mutual_information_values(word_1, word_2, bigram_words,word_1_counter, word_2_counter, bigram_freq, document_total_size):
    try:
        real_value = (bigram_freq/document_total_size)
        carpim = ((word_1_counter*word_2_counter)/(document_total_size*document_total_size))
        deneme_value = real_value/carpim
        mutual_value = math.log2(deneme_value)
        print(
        f'Word-1: {word_1}, Word-1-Size: {word_1_counter}, Word-2: {word_2}, Word-2-Size: {word_2_counter}, Bigram-Words: {bigram_words}, Bigram-Words-Size: {bigram_freq}, Mutual Information {abs(mutual_value)}')
    except ZeroDivisionError:
        mutual_value = 1



def get_mutual_information_values(tagged_unigram, tagged_bigram, document_total_size):
    loop_stop_counter = 0
    word_1_counter = 0
    word_2_counter = 0
    for i in tagged_bigram:
        loop_stop_counter += 1
        if loop_stop_counter == 201:
            break
        bigram_freq = i[1]
        bigram_words = i[0]
        # print(type(bigram_freq))
        word_1 = i[0][0]
        # print(word_1)
        word_2 = i[0][1]
        for j in tagged_unigram:
            if j[0] == word_1:
                word_1_counter = j[1]
                # print(word_1_counter)
            if j[0] == word_2:
                word_2_counter = j[1]

        calculate_mutual_information_values(word_1, word_2, bigram_words, word_1_counter, word_2_counter, bigram_freq, document_total_size)

def main():
    unigram = get_text()
    bigrams = nltk.bigrams(contents)

    # *********************************
    token_freq = nltk.FreqDist(unigram)
    token_freq_for_bigram = nltk.FreqDist(bigrams)
    # *********************************

    # *********************************
    tagged_collocations = collocation_tags(token_freq)
    tagged_collocations_for_bigram = collocation_tags(token_freq_for_bigram)

    get_mutual_information_values(tagged_collocations, tagged_collocations_for_bigram, len(unigram))
    print(len(unigram))



if __name__ == '__main__':
    main()
