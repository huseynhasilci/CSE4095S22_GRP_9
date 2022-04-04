import json
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
real_data = {}
a = 0
print()

def add_into_real_data(real_data_list):
    for j in real_data_list:
        if j not in real_data.keys():
            real_data[j] = 1
        elif j in real_data.keys():
            real_data[j] += 1


def get_unique_word_number(uniques):
    return len(uniques)


def calculate_variance(variance_list):
    return np.var(variance_list)


def calculate_std(standard_daviation_list):
    return np.std(standard_daviation_list)


def calculate_mean(listed_mean):
    return np.mean(listed_mean)


def calculate_total_document(total_words_list):
    return np.sum(total_words_list)


for i in os.listdir():

    with open(i, encoding="utf8") as json_file:
        if i == 'nlp_deneme.py':
            continue
        else:
            data = json.load(json_file)
            # ihticat
            # ***********************************
            ichicat_text = data['ictihat']
            ichicat_list = ichicat_text.split(' ')
            tokenized_ichicat = word_tokenize(ichicat_text)
            # ***********************************
            add_into_real_data(tokenized_ichicat)
            # mahkemesi
            # ***********************************
            mahkeme_text = data['Mahkemesi']
            mahkeme_list = mahkeme_text.split(' ')
            # ***********************************
            add_into_real_data(mahkeme_list)
            # suc
            # ***********************************
            suc_text = data['Suç']
            suc_list = suc_text.split(' ')
            # ***********************************
            add_into_real_data(mahkeme_list)
            # Dava Türü
            # ***********************************
            dava_turu_text = data['Dava Türü']
            dava_turu_list = suc_text.split(' ')
            # ***********************************
            add_into_real_data(dava_turu_list)

            if data['Dairesi'] not in real_data.keys():
                real_data[data['Dairesi']] = 1
            elif data['Dairesi'] in real_data.keys():
                real_data[data['Dairesi']] += 1


def get_the_constant_values(word_dict):
    sorted_data = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1])}

    for_std_val_list = []
    for i in sorted_data.keys():
        if i == '...' or i == '' or i == ', ':
            continue
        else:
            for_std_val_list.append(sorted_data[i])

    unique_words = get_unique_word_number(for_std_val_list)
    standard_deviation = calculate_std(for_std_val_list)
    mean = calculate_mean(for_std_val_list)
    total_document = calculate_total_document(for_std_val_list)
    variance = calculate_variance(for_std_val_list)
    print(standard_deviation)
    print(mean)
    print(total_document)
    print(variance)
    print(unique_words)


get_the_constant_values(real_data)
print(stopwords.words('turkish'))
