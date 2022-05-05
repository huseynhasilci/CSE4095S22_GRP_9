import csv
import os, json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


path_json_files = "C:/Users/murat/Desktop/Nlp-Project/documents/docs/"
path_csv_file = "C:/Users/murat/Desktop/Nlp_delivery2/"


# convert json to csv files
def get_text():
    #docs_num = 0
    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:
        """
        if docs_num == 1000:
            break
        docs_num += 1
        """
        with open(path_json_files + file_name, encoding="utf8") as json_file:
            data = json.load(json_file)
            if data["Suç"] == "":
                data["Suç"] = "undefined"

            informations = [file_name, data["Suç"].strip().lower(), data["ictihat"]]
            #csv_writer.writerow(informations)


def clean_text(text):
    alphabetic_only = [word for word in text.split() if word.isalpha()]
    lower_case_only = [word.lower() for word in alphabetic_only]
    stopwords_tr = set(stopwords.words("turkish"))
    return [word for word in lower_case_only if word not in stopwords_tr]


def doc2bow():
    # most frequent crimes
    #print(df["Label"].value_counts().head(20))

    bow_transformer = CountVectorizer(analyzer=clean_text).fit(df['Text'].astype("U"))
    document_bow = bow_transformer.transform(df["Text"].astype("U"))

    # frequency of words for each document.
    return document_bow


def bow2tfidf(document_bow):
    tfidf_transformer = TfidfTransformer().fit(document_bow)
    text_tfidf = tfidf_transformer.transform(document_bow)
    # words with scores.
    return text_tfidf



def train_model(text_tfidf, dataframe):
    x_train, x_test, y_train, y_test = train_test_split(text_tfidf, dataframe.NUM_LABEL, random_state=42,
                                                        test_size=0.2)

    model = MultinomialNB(alpha=0.5).fit(text_tfidf, dataframe.NUM_LABEL)

    all_predicts = model.predict(x_test)
    return all_predicts, y_test



if __name__ == '__main__':
    """
    ### Write json files to csv file. ###
    csv_file = open(path_csv_file + 'data.csv', 'w', encoding="UTF8")
    csv_writer = csv.writer(csv_file)
    header = ["File Name", "Label", "Text"]
    csv_writer.writerow(header)
    get_text()
    """
    df = pd.read_csv(path_csv_file + "data.csv")
    # make label numeric value.
    label_map = {}
    count = 0
    for label in df.Label.unique():
        label_map[label] = count
        count += 1
    df['NUM_LABEL'] = df.Label.map(label_map)
    document_bow = doc2bow()
    text_tfidf = bow2tfidf(document_bow)  # find most significant words of crimes
    predictions, y_test = train_model(text_tfidf, df)
    print(accuracy_score(predictions, y_test))



