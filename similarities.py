from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate import bleu_score
import numpy as np
import math


def cosine_similarity_(doc1, doc2):
    tokens1 = np.array(word_tokenize(doc1))
    tokens2 = np.array(word_tokenize(doc2))
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]
    stop_words = set(stopwords.words("english"))
    tokens1 = [token for token in tokens1 if token.lower() not in stop_words]
    tokens2 = [token for token in tokens2 if token.lower() not in stop_words]
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(vector1[0], vector1[1])
    return similarity[0][0]


def get_bleu_score(reference, candidate):
    bleu = bleu_score.sentence_bleu([reference], candidate)
    return bleu


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def get_levenshtein_similarity(doc1, doc2):
    distance = levenshtein_distance(doc1, doc2)
    max_length = max(len(doc1), len(doc2))
    similarity = 1 - (distance / max_length)
    return similarity


def get_jaccard_similarity(doc1, doc2):
    set1 = set(doc1.lower().split())
    set2 = set(doc2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union
    return similarity


def euclidean_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    squared_diff = [(a - b) ** 2 for a, b in zip(vec1, vec2)]
    euclidean_dist = math.sqrt(sum(squared_diff))
    return euclidean_dist


def get_euclidean_similarity(doc1, doc2):
    tokens1 = doc1.lower().split()
    tokens2 = doc2.lower().split()
    vocabulary = list(set(tokens1 + tokens2))
    vec1 = [tokens1.count(word) for word in vocabulary]
    vec2 = [tokens2.count(word) for word in vocabulary]
    return euclidean_distance(vec1, vec2)


def get_dice_coefficient(doc1, doc2):
    set1 = set(doc1.lower().split())
    set2 = set(doc2.lower().split())
    intersection = len(set1.intersection(set2))
    dice_coeff = 2 * intersection / (len(set1) + len(set2))
    return dice_coeff


def get_word_error_rate(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
	total_words = len(ref_words)
	wer = (substitutions + deletions + insertions) / total_words
	return wer