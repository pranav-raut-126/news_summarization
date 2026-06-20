
import streamlit as st
import pandas as pd
import textstat
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

#extractive Feature 1: importance score predicted for each sentences
def get_sentence_importance_table(df_features):
    """
    Returns clean dataframe for Streamlit display.
    """

    import pandas as pd

    # Create sentence index manually
    df_features = df_features.copy()
    df_features["sentence_index"] = df_features.index

    # Select required columns safely
    table_df = df_features[
        ["sentence_index", "sentence_text", "importance_score"]
    ].copy()

    # Sort by importance
    table_df = table_df.sort_values(
        by="importance_score",
        ascending=False
    )

    # Rename columns
    table_df.rename(columns={
        "sentence_index": "Sentence Index",
        "sentence_text": "Sentence",
        "importance_score": "Importance Score"
    }, inplace=True)

    # Round scores for clean look
    table_df["Importance Score"] = table_df["Importance Score"].round(4)

    return table_df

#extractive feature 2 :Keywords

lemmatizer = WordNetLemmatizer()

def extract_keywords(text, top_n=5):

    # Step 1: Extract named entities
    doc = nlp(text)
    entity_words = set(
        ent.text.lower()
        for ent in doc.ents
    )

    words = word_tokenize(text.lower())

    filtered = [
        lemmatizer.lemmatize(w)
        for w in words
        if w.isalpha()
        and w not in stop_words
        and len(w) > 3
        and w not in entity_words   # 🔥 REMOVE NAMED ENTITIES
    ]

    word_freq = Counter(filtered)

    return [word for word, _ in word_freq.most_common(top_n)]
#extractive feature 3: Named entities( only person and places)
import re

def extract_named_entities(text):
    doc = nlp(text)

    def normalize(name):
        """Remove punctuation, lowercase, strip whitespace."""
        return re.sub(r'[^\w\s]', '', name).strip().lower()

    def strip_possessive(name):
        """Remove possessive 's from end of string (handles curly and straight apostrophes)."""
        return re.sub(r"[\u2019']s$", "", name.strip(), flags=re.IGNORECASE).strip()

    def is_valid_name(name):
        """Filter out spaCy mistakes like 'Jaishankar calls' where a lowercase verb sneaks in."""
        words = name.split()
        # A valid person name should have all words capitalized
        return all(w[0].isupper() for w in words if len(w) > 1)

    # --- Extract locations first (needed to cross-check persons) ---
    raw_locations = list(set(
        ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"
    ))

    # Normalize possessives and deduplicate locations
    # e.g. "Sri Lanka's" → "Sri Lanka"
    normalized_location_map = {}
    for loc in raw_locations:
        clean = strip_possessive(loc)
        clean_lower = clean.lower()
        if clean_lower not in normalized_location_map:
            normalized_location_map[clean_lower] = clean

    deduped_locations = list(normalized_location_map.values())

    # Remove derivative/prefix forms (e.g. "Americans" when "America" exists)
    cleaned_locations = []
    for loc in deduped_locations:
        loc_lower = loc.lower()
        has_shorter_root = any(
            other.lower() != loc_lower
            and loc_lower.startswith(other.lower())
            and abs(len(loc_lower) - len(other.lower())) <= 3
            for other in deduped_locations
        )
        if not has_shorter_root:
            cleaned_locations.append(loc)

    locations = cleaned_locations[:5]
    location_names_lower = set(loc.lower() for loc in cleaned_locations)

    # --- Extract persons ---
    raw_persons = list(set(
        ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"
    ))

    # Step 1: Filter out invalid spaCy extractions (e.g. "Jaishankar calls")
    raw_persons = [p for p in raw_persons if is_valid_name(p)]

    # Step 2: Remove persons that are actually locations (spaCy misclassification)
    raw_persons = [
        p for p in raw_persons
        if normalize(p) not in location_names_lower
    ]

    # Step 3: Deduplicate by normalized form (handles "S. Jaishankar" vs "S Jaishankar")
    seen_normalized = {}
    for name in raw_persons:
        norm = normalize(name)
        if norm not in seen_normalized:
            seen_normalized[norm] = name

    deduped_persons = list(seen_normalized.values())

    # Step 4: Remove short names that are substrings of longer names
    # e.g. remove "Jaishankar" if "S Jaishankar" already exists
    cleaned_persons = []
    for name in deduped_persons:
        norm_name = normalize(name)
        if not any(
            norm_name != normalize(other) and norm_name in normalize(other)
            for other in deduped_persons
        ):
            cleaned_persons.append(name)

    persons = cleaned_persons[:5]

    return persons, locations

#Abstractive feature 1: compression ratio (behaviour indicator)
def compute_compression(original_text, summary_text):
    original_len = len(original_text.split())
    summary_len = len(summary_text.split())

    if original_len == 0:
        return 0, 0, 0

    compression = round((summary_len / original_len) * 100, 2)

    return original_len, summary_len, compression

#Abstractive feature 2 :Readability score ()
def compute_readability(text):
    flesch = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)

    return round(flesch, 2), round(grade, 2)

