import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []
with open('collection.csv', 'r', newline='', encoding='utf-8') as csvfile:
    readerite = csv.reader(csvfile)
    for i, row in enumerate(readerite):
        if i > 0:  # skip header
            documents.append(row[0])

print("Original documents:")
for idx, d in enumerate(documents, start=1):
    print(f"d{idx}: {d}")

stemmer = PorterStemmer()

def stem_tokenizer(text: str):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [stemmer.stem(t) for t in tokens]

stop_words = ["i", "she", "her", "they", "their", "and", "or", "a", "an", "the"]

vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=stem_tokenizer,
    stop_words=stop_words,
    ngram_range=(1, 2),
    binary=True,
)

doc_matrix = vectorizer.fit_transform(documents)
print("\nVocabulary:", vectorizer.get_feature_names_out().tolist())

query = "I love dogs"
query_vector = vectorizer.transform([query])

doc_vectors = doc_matrix.toarray().tolist()
query_vector = query_vector.toarray().tolist()[0]

scores = []
for dv in doc_vectors:
    scores.append(sum(d_i * q_i for d_i, q_i in zip(dv, query_vector)))

print("\nScores:")
for i, s in enumerate(scores, start=1):
    print(f"d{i}: {s}")

ranking = sorted(
    [(i + 1, scores[i], documents[i]) for i in range(len(documents))],
    key=lambda x: (-x[1], x[0])
)

print("\nRanking (doc_id, score, document):")
for doc_id, score, text in ranking:
    print(f"d{doc_id}\t{score}\t{text}")

