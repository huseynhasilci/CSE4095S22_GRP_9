import json, os
import math

import nltk
from nltk.corpus import stopwords
from zemberek import TurkishMorphology
from collections import Counter


# you need to run this if this is first time trying import stopwords.
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

contents = []
path_json_files = "documents/docs/"
pos_tagger = TurkishMorphology.create_with_defaults()


# read json files
def get_text():
    # docs_num = 0
    for file_name in [file for file in os.listdir(path_json_files) if file.endswith('.json')]:
        """
        docs_num += 1
        if docs_num == 10:
            break
        """
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


# find t values of collocations
def find_t_value(bigram, bigram_freq, words_with_freq, count_total_words):
    count_word1 = words_with_freq[bigram[0]]
    count_word2 = words_with_freq[bigram[1]]
    expected_mean = (count_word1/count_total_words) * (count_word2/count_total_words)  # multiply probability of two words.
    observed_mean = bigram_freq/count_total_words  # observed probability of bigram.
    if observed_mean == 0:
        t_value = 0
    else:
        t_value = (observed_mean - expected_mean) / math.sqrt(observed_mean/count_total_words)

    return t_value, count_word1, count_word2, bigram_freq, bigram


def get_collocations(bigrams_with_freq):
    words_with_freq = Counter(contents)
    count_total_words = len(contents)
    results_t_test = []
    for bigram in bigrams_with_freq:
        bigram_freq = bigrams_with_freq[bigram]
        results_t_test.append(find_t_value(bigram, bigram_freq, words_with_freq, count_total_words))
    results_t_test.sort(key=lambda x: x[0], reverse=True)  # sort by t value.
    return results_t_test


# show results
def print_result(collocation_by_t_test):
    print("T Value\tCount(W1)\tCount(W2)\tCount(W1-W2)\t(W1-W2)")
    for i in range(100):
        print(collocation_by_t_test[i])


def main():
    get_text()
    bigrams = nltk.bigrams(contents)
    token_freq = nltk.FreqDist(bigrams)
    collocation_by_t_test = get_collocations(token_freq)
    print_result(collocation_by_t_test)


if __name__ == '__main__':
    main()
