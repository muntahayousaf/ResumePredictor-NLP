🧠 Resume Category Prediction using NLP & Machine Learning

## 📘 Overview
This project is a **Resume Category Prediction App** built using **Natural Language Processing (NLP)** and **Machine Learning**.  
It automatically predicts a candidate’s **job category** (e.g., *Data Science, Java Developer, Web Designing, HR,* etc.) from their resume text.  
The model was trained with **TF-IDF vectorization**, achieved **98% accuracy**, and was deployed using **Streamlit** for real-time predictions.

---

## 🎯 Objectives
- Automate resume classification using text-based machine learning.
- Perform text cleaning and NLP preprocessing.
- Apply vectorization using TF-IDF.
- Train and evaluate ML models with high accuracy.
- Deploy the trained model as an interactive Streamlit app.

---

## ⚙️ Project Workflow

### Step 1: Data Cleaning & Preprocessing
Used **Regular Expressions (re)** and **NLTK** for advanced text preprocessing:
- Removed URLs, hashtags, mentions, and punctuation.
- Removed numbers and non-ASCII characters.
- Converted text to lowercase.
- Tokenized and removed stopwords.
- Prepared the cleaned text for vectorization.

### Step 2: Feature Extraction
- Applied **TF-IDF Vectorization** to transform text into numerical format.
- Saved the vectorizer (`tfidf.pkl`) for model deployment.

### Step 3: Model Training
- Tested multiple ML models.
- Finalized the best-performing model with **98% accuracy**.
- Saved the trained model as `clf.pkl`.

### Step 4: Deployment
- Built a **Streamlit** web app where users can upload resumes (`.txt` or `.pdf`).
- The app predicts and displays the most suitable **job category** instantly.

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, NLTK, re |
| **Vectorization** | TF-IDF |
| **Model** | KNeighborsClassifier |
| **Deployment** | Streamlit |
| **Serialization** | Pickle |

---

### Run Streamlit App

```bash
streamlit run app2.py
