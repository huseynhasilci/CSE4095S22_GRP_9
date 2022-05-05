import json
import os
import csv
import xlsxwriter
import numpy as np
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords
from zemberek import TurkishMorphology
from nltk.tokenize import word_tokenize
import string
contents = []
suc_frequency = {}
path_json_files = "C:/Users/husey/PycharmProjects/pythonProject10/2021-01/"
dict_list = []
general_values = {
    'ictihat': [],
    'Suç': [],
    # 'class_list': []
}
fieldnames = ['ictihat', 'Suç']# 'class_list']
combine_classes = [' kasten yaralama', ' tehdit', ' hakaret', ' hırsızlık', ' uyuşturucu madde ticareti yapma',
                   ' mala zarar verme', ' kişiyi hürriyetinden yoksun kılma', ' 5607 sayılı kanuna muhalefet',
                   ' resmi belgede sahtecilik', ' silahla tehdit', ' 5607 sayılı kanuna aykırılık',
                   ' görevi yaptırmamak için direnme', ' kullanmak için uyuşturucu madde bulundurma',
                   ' taksirle yaralama', ' silahlı terör örgütüne üye olma', ' 6136 sayılı yasaya aykırılık',
                   ' nitelikli hırsızlık', ' dolandırıcılık', ' trafik güvenliğini tehlikeye sokma',
                   ' görevi kötüye kullanma']
suc_to_class = {}

def get_text():
    docs_num = 0

    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:
        """
        docs_num += 1
        if docs_num == 100:
            break"""

        with open(path_json_files + file_name, encoding="utf-8-sig") as json_file:
            dict = json.load(json_file)
            text = dict["Suç"]
            ictihat_text = dict["ictihat"]
            happy_lines = []


            tokens = word_tokenize(ictihat_text)
            tokens = [w.lower() for w in tokens]

            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]

            words = [word for word in stripped if word.isalpha()]
            happy_lines.append(words)

            text = text.lower()
            if text == "":
                continue
            split_comma = text.split(',')
            for i in range(len(split_comma)):
                general_values['ictihat'].append(happy_lines)
                general_values['Suç'].append(split_comma[i])
                """if split_comma[i] not in combine_classes:
                    general_values['ictihat'].append(happy_lines)
                    general_values['Suç'].append('Others')"""
                #else:

                #dict_list.append({"ictihat": ictihat_text, 'Suç':split_comma[i]})
                #if split_comma[i] not in suc_frequency:
                #    suc_frequency[split_comma[i]] = 1
                #else:
                #    suc_frequency[split_comma[i]] += 1
            #print(split_comma)

get_text()
dataframe1 = pd.DataFrame(general_values)
with pd.ExcelWriter('../../general_hukums_with_21_classes111222.xlsx', engine='xlsxwriter') as writer:
    dataframe1.to_excel(writer, sheet_name='general_hukum111')

#for i in dict_list:
#    print(i)

#with open('hukums.csv', 'w', encoding="utf-8", newline='') as file:
#    writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
#    writer.writeheader()
#    for i in dict_list:
#        writer.writerow(i)
"""sorted_suc_frequency = dict(sorted(suc_frequency.items(), key=lambda item: item[1], reverse=True))
json_sorted_suc_frequency = json.dumps(sorted_suc_frequency, ensure_ascii=False, indent=4)
json_file = open('before_combine.json', 'w', encoding='utf-8')
json_file.write(json_sorted_suc_frequency )
json_file.close()"""
# print(sorted_suc_frequency)

"""
            if text not in suc_frequency:
                suc_frequency[text] = 1
            elif '5607' in text and ' 5607 sayılı Kanuna muhalefet' not in suc_frequency:
                #print('Burada')
                suc_frequency[' 5607 sayılı Kanuna muhalefet'] = 1

            elif '6136' in text and ' 6136 sayılı Yasaya aykırılık' not in suc_frequency:
                #print('Burada')
                suc_frequency[' 6136 sayılı Yasaya aykırılık'] = 1
"""
