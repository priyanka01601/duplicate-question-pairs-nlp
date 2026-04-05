import re
import numpy as np
import pickle
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
from joblib import load
# Load stopwords ONCE (important)


# ------------------ BASIC FEATURES ------------------

def test_common_words(q1, q2):
    w1 = set(q1.lower().strip().split())
    w2 = set(q2.lower().strip().split())
    return len(w1 & w2)


def test_total_words(q1, q2):
    w1 = set(q1.lower().strip().split())
    w2 = set(q2.lower().strip().split())
    return len(w1) + len(w2)


# ------------------ TOKEN FEATURES ------------------

def test_fetch_token_features(q1, q2,STOP_WORDS):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])

    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


# ------------------ LENGTH FEATURES ------------------

def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    # Safe longest substring
    try:
        substrs = list(distance.lcsubstrings(q1, q2))
        if len(substrs) > 0:
            length_features[2] = len(substrs[0]) / (min(len(q1), len(q2)) + 1)
    except:
        length_features[2] = 0.0

    return length_features


# ------------------ FUZZY FEATURES ------------------

def test_fetch_fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]


# ------------------ PREPROCESSING ------------------

def preprocess(q):
    q = str(q).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')

    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # HTML removal
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove punctuation
    q = re.sub(r'\W', ' ', q).strip()

    return q


# ------------------ FINAL PIPELINE ------------------

def query_point_creator(q1, q2, cv,STOP_WORDS):
    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))

    common = test_common_words(q1, q2)
    total = test_total_words(q1, q2)

    input_query.append(common)
    input_query.append(total)

    ratio = common / total if total != 0 else 0
    input_query.append(round(ratio, 2))

    input_query.extend(test_fetch_token_features(q1, q2,STOP_WORDS))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, -1), q1_bow, q2_bow))