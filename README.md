
# **Health Analyzer 🩺**

*Streamlit-Powered Health Risk Prediction App*

[![Live Demo](https://img.shields.io/badge/Demo-Live_App-green)](https://healthanalyzer-ml.streamlit.app/)


---

## 📌 **Overview**

**Smart Health Analyzer** is a **Streamlit-based** machine learning app that predicts potential health risks (like heart disease or diabetes) based on user input. It provides end-to-end ML capabilities, including:

* ✅ Data Cleaning & Feature Engineering
* ✅ Exploratory Data Analysis
* ✅ Supervised Learning & Model Evaluation
* ✅ Hyperparameter Tuning
* ✅ Real-time Prediction & SHAP Explainability
* ✅ Deployed via Streamlit Community Cloud

---

## 🚀 **Features**

* 🔍 **Predicts multiple health conditions** from patient data
* 📊 **Interactive visualizations** (probability & SHAP plots)
* 📱 **Mobile-responsive UI**, accessible from any device
* 🔧 **Modular design** to add more diseases/models easily

---

## 🛠 **Tech Stack**

| Layer                | Tools/Technologies            |
| -------------------- | ----------------------------- |
| **Machine Learning** | Python, Scikit-learn, XGBoost |
| **Explainability**   | SHAP, Matplotlib              |
| **Web Framework**    | Streamlit                     |
| **Deployment**       | Streamlit Community Cloud     |

---

## ⚙️ **Setup Instructions**

### 1. Clone the Repository

```bash
git clone https://github.com/Kumarpal613/health_analyzer.git
cd health_analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

📍 Opens at: `http://localhost:8501`

---

## 📂 **Project Structure**

```bash
smart_health_analyzer/
├── app/
│   ├── streamlit_app.py       # Streamlit UI logic
│   └── assets/                # UI assets (images, icons)
├── models/
│   └── heart_model.pkl        # Trained heart disease model
├── notebooks/
│   └── data_cleaning_and_model_training.ipynb   #  Data wrangling ,Model training & tuning
├── requirements.txt
└── README.md
```

---

## 🌐 **Deploy on Streamlit**

🔗 **Live App**: [https://healthanalyzer-ml-streamlit-app.streamlit.app](https://healthanalyzer-ml.streamlit.app/)

---

## 🎯 **Note**

* No Flask, HTML, or CSS — **pure Python + Streamlit**
* Focused only on **supervised learning** (no unsupervised methods used)

