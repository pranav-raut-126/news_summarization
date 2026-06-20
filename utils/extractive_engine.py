import joblib
import pandas as pd
from nltk.tokenize import sent_tokenize
from utils.processing1 import extract_features_from_text


# Load components globally for efficiency
MODEL = joblib.load("models/extractive/extractive_lr_model.joblib")
VECTORIZER = joblib.load("models/extractive/tfidf_vectorizer_main.joblib")
METADATA = joblib.load("models/extractive/extractive_lr_metadata.joblib")
FEATURE_COLUMNS = METADATA["feature_columns"]

def get_extractive_summary(text, top_n=3):
    """
    Processes text and returns the top N most important sentences 
    along with their importance scores.
    """
    #if article is to short 
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        # Fallback: return original text as summary
        return text, None

    # 1. Generate features using your training logic
    df_features = extract_features_from_text(text, VECTORIZER)
    
    if df_features.empty:
        return [], pd.DataFrame()

    # 2. Predict using the specific columns your model expects
    X = df_features[FEATURE_COLUMNS].values
    probs = MODEL.predict_proba(X)[:, 1] 
    df_features['importance_score'] = probs
    
    # 3. Get Top N sentences
    top_df = df_features.nlargest(top_n, 'importance_score').sort_index()
    
    summary_list = top_df['sentence_text'].tolist()
    return " ".join(summary_list), df_features