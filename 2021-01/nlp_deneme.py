import json
import os
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
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

def calculate_chi_square_test(word_1, word_2, bigram_words,word_1_counter, word_2_counter, bigram_freq, document_total_size):
    try:
        numerator = (document_total_size)*(((bigram_freq*document_total_size)-(word_1_counter*word_2_counter))**2)
        denominator = (bigram_freq+word_1_counter)*(bigram_freq+word_2_counter)*(word_1_counter+document_total_size)*(word_2_counter+document_total_size)
        chi_square_result = numerator/denominator
        print(f'Word-1: {word_1}, Word-1-Size: {word_1_counter}, Word-2: {word_2}, Word-2-Size: {word_2_counter}, Bigram-Words: {bigram_words}, Bigram-Words-Size: {bigram_freq}, Chi-Square-Test {chi_square_result}')
    except ZeroDivisionError:
        mutual_value = 1


def get_the_constant_values(word_dict):
    word_1_list = [1155, 3076, 2864, 14262, 1085, 7824, 3083, 5035, 4693, 9903]
    word_2_list = [4693, 3183, 3083, 11003, 297, 8001, 3297, 7824, 3491, 8400]
    bigr
    sorted_data = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1],reverse=True)}
    print(sorted_data['vicdani'])
    print(sorted_data['kanının'])
    print(sorted_data['sürecini'])
    print(sorted_data['yansıtan'])
    print(sorted_data['bölge'])
    print(sorted_data['adliye'])
    print(sorted_data['oy'])
    print(sorted_data['birliğiyle'])
    print(sorted_data['tetkik'])
    print(sorted_data['hakimi'])
    print(sorted_data['yürürlüğe'])
    print(sorted_data['giren'])
    print(sorted_data['adliye'])
    print(sorted_data['mahkemesi'])
    print(sorted_data['yayımlanarak'])
    print(sorted_data['yürürlüğe'])
    print(sorted_data['kanının'])
    print(sorted_data['oluştuğu'])
    print(sorted_data['gününde'])
    print(sorted_data['oybirliğiyle'])
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
    #print(standard_deviation)
    #print(mean)
    #print(total_document)
    #print(variance)
    #print(unique_words)

#print(real_data)

get_the_constant_values(real_data)
print(stopwords.words('turkish'))
