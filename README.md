# 📱 SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **TF-IDF feature extraction** and a **Linear Support Vector Machine (SVM)** classifier.

## 📌 Project Overview

SMS spam is a common form of unwanted communication that can contain advertisements, scams, phishing attempts, or other unsolicited content.

This project builds a text classification pipeline to automatically determine whether an SMS message is:

* 🟢 **Ham** — legitimate/non-spam message
* 🔴 **Spam** — unwanted or fraudulent message

The project uses Natural Language Processing (NLP) techniques to preprocess the SMS text, convert it into numerical features using **TF-IDF**, and classify the messages using **Linear SVM**.

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**, containing **5,572 SMS messages**.

Dataset source:

[Kaggle — SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Dataset Statistics

| Category          | Count |
| ----------------- | ----: |
| Total Messages    | 5,572 |
| Ham               | 4,825 |
| Spam              |   747 |
| Missing Values    |     0 |
| Duplicate Records |   403 |

The original dataset columns `v1` and `v2` are renamed to:

* `label`
* `message`

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical operations
* **Matplotlib** — visualization
* **Seaborn** — statistical visualization
* **NLTK** — natural language processing
* **WordCloud** — text visualization
* **Scikit-learn** — machine learning

## 🔬 Methodology

The project follows these major steps:

```text
SMS Dataset
     ↓
Data Loading
     ↓
Data Inspection & Cleaning
     ↓
Exploratory Data Analysis
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Linear SVM Classification
     ↓
Model Evaluation
```

### 1. Data Loading

The `spam.csv` dataset is loaded using Pandas and the relevant columns are selected.

### 2. Data Inspection

The dataset is analyzed for:

* Number of records
* Data types
* Missing values
* Duplicate messages
* Class distribution

### 3. Exploratory Data Analysis

EDA is performed to understand the distribution and characteristics of spam and ham messages.

The notebook also uses visualizations such as:

* Class distribution plots
* Word clouds
* Text-related visualizations

### 4. Text Preprocessing

Natural Language Processing techniques are applied to prepare the SMS messages for classification.

The project uses:

* Regular expressions
* String processing
* Stopword removal
* WordNet lemmatization

### 5. TF-IDF Feature Extraction

The cleaned SMS messages are converted into numerical feature vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

TF-IDF helps represent words according to their importance within the collection of messages.

### 6. Linear SVM Classification

A **Linear Support Vector Classifier (`LinearSVC`)** is used to distinguish between spam and ham messages.

The project uses a Scikit-learn pipeline to combine the TF-IDF vectorizer and classifier.

### 7. Model Evaluation

The classifier is evaluated using multiple metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve / AUC

Cross-validation and hyperparameter tuning are also included in the notebook.

## 📈 Model

**Classifier:** Linear Support Vector Machine (`LinearSVC`)

**Feature Extraction:** TF-IDF

**Pipeline:**

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("svm", LinearSVC(...))
])
```

## 📁 Project Structure

```text
SMS-Spam-Classification/
│
├── SMS_Spam_SVM_raj.ipynb
├── spam.csv
└── README.md
```

> **Note:** If `spam.csv` is not included in the repository, download it from the dataset source and place it in the project directory before running the notebook.

## 🚀 How to Run

### Option 1 — Google Colab

1. Open `SMS_Spam_SVM_raj.ipynb` in Google Colab.
2. Install the required libraries.
3. Upload `spam.csv` when prompted.
4. Run the notebook cells sequentially.

### Option 2 — Local Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/your-username/SMS-Spam-Classification.git
cd SMS-Spam-Classification
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
SMS_Spam_SVM_raj.ipynb
```

and run the cells.

## 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Text classification
* Natural Language Processing
* Data cleaning
* Exploratory Data Analysis
* TF-IDF vectorization
* Support Vector Machines
* Machine learning pipelines
* Hyperparameter tuning
* Cross-validation
* Classification metrics
* Confusion matrix analysis
* ROC/AUC evaluation

## 🔮 Future Improvements

Possible improvements include:

* Comparing SVM with Logistic Regression, Naive Bayes, Random Forest, and other classifiers
* Testing different NLP preprocessing techniques
* Handling class imbalance
* Using word embeddings such as Word2Vec or GloVe
* Experimenting with transformer-based models such as BERT
* Deploying the trained model as a web application or API
* Adding real-time SMS classification

## 👨‍💻 Author

**Suprio Chattapadhya Raj**

B.Sc. in Information and Communication Technology (ICT)
Bangladesh University of Professionals

---

⭐ If you found this project useful, consider giving the repository a star!
