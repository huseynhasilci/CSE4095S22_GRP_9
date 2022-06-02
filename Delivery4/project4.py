import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import json
import os
#df = pd.read_csv('latest_hukums_with_classes_csv_file1.csv')
#print(df['ictihat'])
#counter = 0
#for i in df['ictihat']:
#    if "Asliye Ceza Mahkemesi" in i:
#        print(i)
#        counter += 1

#print(counter)

path_json_files = "C:/Users/husey/PycharmProjects/pythonProject10/2021-01/"



def get_text():
    docs_num = 0

    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:
        """
        docs_num += 1
        if docs_num == 100:
            break
        """
        with open(path_json_files + file_name, encoding="utf8") as json_file:
            dict = json.load(json_file)
            text = dict["ictihat"]
            #print(text)
            if "Asliye Ceza Mahkemesi" in text or "asliye ceza mahkemesi" in text or "Asliye ceza mahkemesi" in text or "Asliye Ceza mahkemesi" in text:
                docs_num += 1

    print(docs_num)


get_text()
