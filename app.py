import streamlit as st
import requests
import random

from utils.scraper import fetch_article_data
from utils.extractive_engine import get_extractive_summary
from utils.abstractive_engine import load_abstractive_model, get_abstractive_summary
from utils.visualizer import (
    get_sentence_importance_table,
    extract_keywords,
    extract_named_entities,
    compute_compression,
    compute_readability
)

# PAGE CONFIG
st.set_page_config(page_title="NewsNuggets", layout="wide")
NEWS_API_KEY = "548b24ccbf77489ea8004775d38b3035"

# LOAD MODELS (ONCE)
@st.cache_resource
def load_models():
    checkpoint = "models/abstractive"
    tokenizer, model = load_abstractive_model(checkpoint)
    return tokenizer, model

ABS_TOK, ABS_MODEL = load_models()

# SESSION STATE INIT
if "articles_today" not in st.session_state:
    st.session_state.articles_today = None

if "article_summaries" not in st.session_state:
    # structure:
    # { url: { "extractive": str, "abstractive": str } }
    st.session_state.article_summaries = {}

if "interactive_result" not in st.session_state:
    st.session_state.interactive_result = None

# SIDEBAR

st.sidebar.title("NewsNuggets AI")
page = st.sidebar.radio("Go to", ["Home - Latest News", "Summarize Your Own"])


# PAGE 1 : HOME - LATEST NEWS
if page == "Home - Latest News":

    st.title("Top Stories Today")
    st.write("Live news summarized by NewsNuggets AI")

    # Fetch + Validate articles
    if st.session_state.articles_today is None:

        with st.spinner("Fetching and validating live news articles..."):

            url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=10&apiKey={NEWS_API_KEY}"
            response = requests.get(url).json()

            raw_articles = response.get("articles", [])
            validated_articles = []

            for art in raw_articles:

                if not art.get("url"):
                    continue

                article_data = fetch_article_data(art["url"])

                if article_data["success"]:
                    text = article_data["text"]

                    # Basic quality control
                    if text and len(text.split()) > 150:
                        art["full_text"] = text
                        validated_articles.append(art)

                # Stop after 3 valid articles
                if len(validated_articles) == 3:
                    break

            if not validated_articles:
                st.warning("No valid news articles could be processed. Please refresh.")
                st.stop()

            st.session_state.articles_today = validated_articles

    # Show only 1 article

    articles = st.session_state.articles_today[:1]

    for idx, art in enumerate(articles):

        url = art["url"]

        # Generate Summaries

        if url not in st.session_state.article_summaries:

            with st.spinner("Generating summaries..."):

                text = art["full_text"]

                # SAFE EXTRACTIVE
                try:
                    extractive_summary, _ = get_extractive_summary(text, top_n=3)
                    if not extractive_summary.strip():
                        extractive_summary = "Extractive summary could not be generated."
                except Exception:
                    extractive_summary = "Extractive summary could not be generated."

                # SAFE ABSTRACTIVE
                try:
                    abstractive_summary = get_abstractive_summary(
                        text, ABS_TOK, ABS_MODEL,length_category="Long"
                    )
                except Exception:
                    abstractive_summary = "Abstractive summary could not be generated."

                st.session_state.article_summaries[url] = {
                    "extractive": extractive_summary,
                    "abstractive": abstractive_summary,
                }

        summaries = st.session_state.article_summaries[url]


        # UI RENDERING
        with st.container(border=True):

            col1, col2 = st.columns([1, 3])

            with col1:
                img_url = art.get("urlToImage") or "https://via.placeholder.com/400x250?text=NewsNuggets"
                st.image(img_url)

            with col2:
                st.subheader(art["title"])
                st.caption(f"Source: {art['source']['name']} | [Read Original]({url})")

                mode = st.toggle("Switch to Extractive View", key=f"toggle_{idx}")

                if mode:
                    st.info("**Extractive Summary (Importance-based)**")
                    st.write(summaries.get("extractive", "Not available."))
                else:
                    st.success("**Abstractive Summary (Human-like)**")
                    st.write(summaries.get("abstractive", "Not available."))


# PAGE 2 : INTERACTIVE SUMMARIZER
if page == "Summarize Your Own":

    st.title("Interactive Summarizer")
    st.write("Enter an article URL to get summaries.")

    article_url = st.text_input(
        "Article URL",
        placeholder="Paste your URL here"
    )

    model_choice = st.selectbox(
        "Select Summarization Model",
        ["  ", "Extractive (Importance-based)", "Abstractive (Human-like)"]
    )

    # show settings after model is selected 

    if model_choice == "Extractive (Importance-based)":

        with st.container(border=True):

            st.subheader("Model Settings")

            num_sentences = st.slider(
                "Number of sentences",
                min_value=1,
                max_value=10,
                value=3
            )

            st.markdown("### Features")

            show_scores = st.checkbox("Sentence Probability Scores")
            show_entities = st.checkbox("Named Entities")
            show_keywords = st.checkbox("Keywords")
            

    elif model_choice == "Abstractive (Human-like)":

        with st.container(border=True):

            st.subheader("Model Settings")

            length_category = st.selectbox(
                "Select summary length",
                ["  ","Short", "Medium", "Long"])
            
            show_quality = st.checkbox(" Quality Indicators")

    else:
        # If model not selected → stop here
        st.stop()

    # PROCESS BUTTON

    if st.button("Process Article"):

        if not article_url:
            st.warning("Please enter a valid URL.")
            st.stop()

        with st.spinner("Fetching and processing article..."):

            article_data = fetch_article_data(article_url)

            if not article_data["success"]:
                st.error("Failed to fetch article content.")
                st.stop()

            text = article_data["text"]

            # EXTRACTIVE MODEL
            if model_choice == "Extractive (Importance-based)":
                summary, df_features = get_extractive_summary(text, top_n=num_sentences)
                
                st.success("### Extractive Summary")
                st.write(summary)

                if show_scores:
                    st.markdown("### Sentence Probability Scores")

                    score_table = get_sentence_importance_table(df_features)

                    st.dataframe(
                        score_table,
                        height=220,
                        use_container_width=True)

                if show_entities:
                    st.markdown("### Named Entities")
                    persons, locations = extract_named_entities(text)

                    if persons:
                        st.markdown(f"**Persons:** {', '.join(persons)}")
                    else:
                        st.write("No persons found.")   
                    
                    if locations:
                        st.markdown(f"**Locations:** {', '.join(locations)}") 
                    else:
                        st.write("No locations found.")

                if show_keywords:
                    st.markdown("### Keywords")
                    keywords = extract_keywords(text, top_n=5)
                    if keywords:
                        st.markdown(f"{', '.join(keywords)}")
                    else:
                        st.write("No keywords found.")


            # ABSTRACTIVE MODEL
            elif model_choice == "Abstractive (Human-like)":

                summary = get_abstractive_summary(
                    text,
                    ABS_TOK,
                    ABS_MODEL,
                    length_category=length_category
                )

                st.success("### Abstractive Summary")
                st.write(summary)

                # -------- QUALITY INDICATORS --------

                if show_quality:
                    st.markdown("### Quality Indicators")
                    # Get values from your functions
                    original_len, summary_len, compression = compute_compression(text, summary)
                    flesch, grade = compute_readability(summary)
                    
                    #length Analysis
                    st.markdown("### Length Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Word Count", original_len)

                    with col2:
                        st.metric("Summary Word Count", summary_len)

                    with col3:
                        st.metric("Compression (%)", f"{compression}%")

                    #Readability Analysis
                    st.markdown("### Readability Analysis")
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("Flesch Reading Ease", flesch)

                    with col5:
                        st.metric("Estimated Grade Level", grade)
                    
                    st.markdown(
    """
**Readability Interpretation**

- Higher Flesch Reading Ease score indicates easier readability.
- Grade Level estimates the education level required to understand the text.
- Ideally, summaries should maintain content while improving readability.
"""
)
