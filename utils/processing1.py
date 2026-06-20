import pandas as pd
import re
import spacy
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# ---- Load NLP resources safely ----
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
    )

try:
    stop_words = set(stopwords.words("english"))
except Exception as e:
    raise RuntimeError(
        "NLTK stopwords not found. Run: nltk.download('stopwords')"
    )


def clean_text(text: str) -> str:
    """
    Cleans raw article text.
    Must remain identical to training preprocessing.
    """
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_features_from_text(article_text: str, tfidf_vectorizer):
    """
    Feature extraction for extractive summarization.

    Replicates training-time feature engineering:
    - TextRank score
    - TF-IDF similarity with full article
    - Sentence position
    - NER count
    - Stopword ratio
    """

    if not article_text or len(article_text.strip()) < 200:
        raise ValueError("Article text is too short for extractive summarization.")

    cleaned_article = clean_text(article_text)
    sentences = sent_tokenize(cleaned_article)

    if len(sentences) < 2:
        raise ValueError("At least two sentences are required for extractive summarization.")

    total_sentences = len(sentences)

    # TF-IDF representations
    tfidf_article = tfidf_vectorizer.transform([" ".join(sentences)])
    tfidf_sentences = tfidf_vectorizer.transform(sentences)

    # Similarity matrix for TextRank
    sim_matrix = cosine_similarity(tfidf_sentences)
    nx_graph = nx.from_numpy_array(sim_matrix)

    try:
        textrank_scores = nx.pagerank(nx_graph, max_iter=200)
    except Exception:
        # Fallback for numerical instability
        textrank_scores = {i: 1.0 for i in range(total_sentences)}

    feature_rows = []

    for i, sent in enumerate(sentences):
        words = word_tokenize(sent)
        word_count = len(words)

        doc = nlp(sent)

        features = {
            "sentence_index": i,
            "normalized_position": i / total_sentences,
            "sentence_length_words": word_count,
            "tfidf_sim_article": cosine_similarity(
                tfidf_sentences[i], tfidf_article
            )[0][0],
            "ner_count": len(doc.ents),
            "has_digit": int(any(char.isdigit() for char in sent)),
            "stop_ratio": (
                sum(1 for w in words if w.lower() in stop_words) / word_count
                if word_count > 0 else 0
            ),
            "textrank_score": textrank_scores.get(i, 0),
            "sentence_text": sent
        }

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
