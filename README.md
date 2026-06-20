# NewsNuggets: Hybrid News Summarization System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F3B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![NewsAPI](https://img.shields.io/badge/NewsAPI-000000?style=for-the-badge&logo=rss&logoColor=white)

**An end-to-end AI-powered news summarization system combining a custom Extractive ML model, Transformer-based Abstractive summarization, and real-time news retrieval through an interactive web application.**

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Datasets Used](#-datasets-used)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Feature Engineering & Label Generation](#-feature-engineering--label-generation)
- [Models](#models)
- [Model Performances](#model-performances)
- [Future Enhancements](#future-enhancements)
- [How to Run Locally](#️-how-to-run-locally)
- [Tech Stack](#️-tech-stack)
- [Author](#author)

---

## 🧠 Project Overview

**NewsNuggets** is an end-to-end AI-powered news summarization system that transforms lengthy news articles into concise and meaningful summaries. The project covers the complete NLP pipeline—from data preprocessing and feature engineering to model development and deployment as an interactive web application.

The project consists of **three major components:**

| Component | File | Description |
|-----|------|-------------|
| 📝**Extractive Summarizer** | `extractive_engine.py` | Custom ML model that ranks important sentences using handcrafted features to generate extractive summaries. |
| 🤖**Abstractive Summarizer** | `abstractive_engine.py` | Transformer-based model that generates fluent, human-like summaries by understanding article context. |
| 🖥️**Streamlit Web App** | `app.py` | Interactive GUI for summarizing real-time news articles using either the Extractive or Abstractive model. |

---

## ✨ Features

📝 Extractive Summarizer
- Supervised Machine Learning model for extractive text summarization.
- Identifies and ranks the most important sentences in a news article.
- Generates concise summaries while preserving the original content and context.
- Built using handcrafted NLP features for sentence importance prediction.

🤖 Abstractive Summarizer
- Transformer-based model fine-tuned for news summarization.
- Generates fluent, human-like summaries by understanding article context.
- Preserves the key information and overall meaning of the original article.

🌐 Streamlit Web Application
- Interactive and user-friendly interface built with Streamlit.
- Fetches real-time news articles using NewsAPI.
- Supports both Extractive and Abstractive summarization modes.
- Displays generated summaries in a clean and intuitive layout.

---

## 📁 Project Structure

```
NEWSNUGGETS/
|
├── data/                            # Datasets used in feature extraction and model training
|    ├── feature.csv
|    ├── final_dataset.csv
|    └── s_split.csv
|
├── models/                          # Trained models and saved artifacts
|    ├── abstractive/
|    |    ├── config.json
|    |    ├── generation_config.json   
|    |    ├── model.safetensors
|    |    ├── special_tokens_map.json
|    |    ├── spiece.model
|    |    └── tokenizer_config.json
|    └── extractive
|         ├── metadata.joblib
|         ├── model.joblib
|         └── tfidf_vectorizer.joblib
|     
├── notebooks/                       # Development and experimentation notebooks
|    ├── Abstractive_model.ipynb      
|    ├── Extractive_model.ipynb      
|    └── Feature_extraction.ipynb    
|  
├── utils/                           # Core application modules
|    ├── abstractive_engine.py
|    ├── extractive_engine.py
|    ├── processing1.py
|    ├── scraper.py
|    └── visualizer.py
|
├── app.py                           # Streamlit application
|
└── requirements.txt                 # Project dependencies
```

---

## 📦 Datasets Used

The project uses a subset of the **CNN/DailyMail** dataset, with each model utilizing the data differently based on its learning objective.

| Dataset | Used for | Description |
|-----|------|-------------|
|CNN/DailyMail Dataset (Subset)|Extractive Summarizer|A subset of the dataset was preprocessed and transformed into a sentence-level dataset through feature extraction and sentence labeling. The resulting dataset was used to train the supervised extractive summarization model.|
|CNN/DailyMail Dataset |Abstractive Summarizer |The original article-summary pairs were used to fine-tune the Transformer-based abstractive summarization model for generating fluent and context-aware summaries. |

---

## 🧹 Data Cleaning & Preprocessing

**Notebook:** `Feature_extraction.ipynb`

To ensure high-quality model training, the CNN/DailyMail dataset underwent several preprocessing steps before being used for both extractive and abstractive summarization.

### Preprocessing Steps

- Removed duplicate and incomplete records.
- Removed HTML tags and normalized whitespace from articles and summaries.
- Performed sentence segmentation on news articles.
- Applied case folding and stopword removal for consistent text processing.
- Converted text into TF-IDF vectors for feature extraction in the Extractive pipeline.
- Applied sequence truncation to fixed input (`512`) and target (`128`) token lengths for the Abstractive model.

---

## 🧬 Feature Engineering & Label Generation

**Notebook:** `Feature_extraction.ipynb`

### Extracted Features
- **Sentence Index** – Represents the original position of a sentence within the article.
- **Normalized Position** – Captures the relative position of a sentence on a scale from 0 to 1.
- **Sentence Length** – Counts the total number of words in each sentence.
- **TF-IDF Similarity to Article** – Measures how well a sentence represents the overall article.
- **TF-IDF Similarity to Summary** – Measures the similarity between a sentence and the reference summary.
- **Named Entity Count (NER)** – Counts entities such as people, organizations, and locations.
- **Has Digit** – Indicates whether a sentence contains numerical information.
- **Stopword Ratio** – Calculates the proportion of stopwords in a sentence.
- **TextRank Score** – Assigns an importance score based on sentence connectivity within the article.

### Label Generation
- Generated binary labels (`1 = Important`, `0 = Not Important`) for each sentence using similarity with the reference summary.
- The labeled feature dataset was used to train the supervised Extractive Summarization model.

---

## Models

### Extractive model 
**Notebook:** `Extractive_model.ipynb`

### Train / Validation / Test Split
The dataset was split at the article level to prevent sentences from the same article appearing in multiple splits.
```
70% Training | 15% Validation | 15% Testing  (stratified)
```
### Extractive Summarization Workflow 
The extractive summarizer is formulated as a supervised sentence-level classification task. Each sentence is represented using engineered features and assigned an importance probability by a Logistic Regression model. Sentences are then ranked based on their predicted probabilities, and the highest-ranked sentences are selected to construct the final summary.

**Primary Model**
- Logistic Regression (`recall: 0.68`)
 
**Baseline Methods**
- Lead-k
- TextRank

The performance of the Logistic Regression model was evaluated against these baseline summarization techniques to measure the effectiveness of the proposed feature-driven supervised approach.

### Abstractive Model

PEGASUS (Transformer)
```
Model: google/pegasus-arxiv (HuggingFace)
Fine-tuned on: CNN Daily Mail Dataset (10k pairs)
Max Input: 512 | Max Summary: 128
Epochs: 3 | Learning Rate: 1e-5 | Effective Batch Size: 8
Best Model: Selected using ROUGE-L
```

---

## Model Performances

### Extractive Model Evaluation

Classification Report for Important Class
| Metric | Score |
|--------|------:|
| Precision | 0.32 |
| Recall | 0.68 |
| F1-Score | 0.44 |

ROUGE Evaluation

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:------|---------:|---------:|---------:|
| **Logistic Regression (Proposed)** | **0.3478** | **0.1386** | **0.2286** |
| Lead-3 | 0.2922 | 0.1169 | 0.1869 |
| TextRank | 0.2700 | 0.0863 | 0.1707 |


### Abstractive Model Evaluation

| Metric | Score |
|:-------|------:|
| Evaluation Loss | 4.2292 |
| ROUGE-1 | 29.40 |
| ROUGE-2 | 10.24 |
| ROUGE-L | 20.34 |
| METEOR | 21.99 |
| BERTScore | **85.32** |

### Generated Summary Quality Indicators
- **Compression Ratio** – Measures how much the original article has been condensed into the generated summary while retaining key information.
- **Flesch Reading Ease** – Indicates how easy the generated summary is to read; higher scores represent easier readability.
- **Grade Level Score** – Estimates the minimum education level required for a reader to understand the generated summary.

---

## Future Enhancements

- **Dataset Scaling** – Train the models on a larger portion of the CNN/DailyMail dataset to improve generalization and summary quality as greater computational resources become available.
- **Enhanced Extractive Modeling** – Explore and compare additional supervised machine learning algorithms to further improve sentence importance prediction.
- **Multilingual Summarization** – Extend the system to support summarization of news articles in multiple languages.
- **Cloud Deployment** – Deploy the application on cloud platforms to enable public access, scalability, and easier sharing with users.
---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/pranav-raut-126/news_summarization.git
cd news_summarization
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Download Models 

 📥 [Download from here ](https://drive.google.com/drive/folders/1zumzSTHcU1h0yRN-IAIJUap8Hvh2vKlj?usp=drive_link)

 After downloading, place them in the structure as shown in [Project Structure](#-project-structure)

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

### 5. Open in Browser

The application will be available at:

```
http://localhost:8500
```

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| **Language** | Python 3.10+ |
| **Web App** | Streamlit |
| **ML** | Scikit-learn |
| **NLP / Transformers** | HuggingFace Transformers (PEGASUS/ArXiv) |
| **Text Processing** | NLTK (stopwords, lemmatizer, tokenizer), spaCy |
| **Vectorization** | TF-IDF (Scikit-learn) |
| **Visualization** | Matplotlib, Seaborn |
| **Data** | Pandas, NumPy |
| **News API** | NewsAPI.org |
| **Model Saving** | Joblib, HuggingFace safetensors |

---

## Author

**Pranav Raut**

B.Sc. Data Science | CGPA: 9.4/10

Bunts Sangha's S.M. Shetty College of Science, Commerce and Management Studies, Mumbai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranavraut126/)

---

<div align="center">

**If you found this project useful, please give it a star!** 

</div>
