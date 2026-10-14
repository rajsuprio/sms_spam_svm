# 📱 SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as either Spam or Ham using a complete text-classification pipeline built with NLP preprocessing and a Linear Support Vector Machine (SVM).

This repository includes data exploration, text cleaning, feature engineering, TF-IDF vectorization, hyperparameter tuning, model validation, and a reusable prediction function for classifying new SMS messages.

## ✨ Features Included

This updated project includes the following capabilities:

- Loading and inspecting the SMS Spam Collection dataset
- Checking for missing values and duplicate records
- Exploratory data analysis (EDA)
- Class distribution analysis
- Spam vs ham word cloud visualization
- Advanced text preprocessing using:
  - lowercase conversion
  - URL and email removal
  - digit removal
  - punctuation removal
  - stopword removal
  - WordNet lemmatization
- Feature engineering for:
  - character count
  - word count
  - number of digits
  - number of punctuation marks
- TF-IDF vectorization with n-grams
- Linear SVM classification using `LinearSVC`
- Stratified train/test splitting
- Hyperparameter tuning with `GridSearchCV`
- 5-fold cross-validation
- Evaluation metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - classification report
  - confusion matrix
- A prediction function for real SMS inputs

## 📊 Dataset

The project uses the SMS Spam Collection Dataset from Kaggle.

Dataset source:

[Kaggle — SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Dataset statistics

| Category | Count |
| --- | ---: |
| Total messages | 5,572 |
| Ham | 4,825 |
| Spam | 747 |
| Missing values | 0 |
| Duplicate records | 403 |

The original dataset columns `v1` and `v2` are renamed to:

- `label`
- `message`

The labels are encoded as:

- `ham -> 0`
- `spam -> 1`

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- WordCloud
- Scikit-learn

## 🔬 Methodology

The notebook follows this workflow:

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
Feature Engineering
  ↓
TF-IDF Vectorization
  ↓
Train/Test Split
  ↓
Linear SVM Classification
  ↓
Hyperparameter Tuning
  ↓
Cross-Validation
  ↓
Model Evaluation
  ↓
Prediction on New SMS
```

### 1. Data loading and inspection

The dataset is loaded from `spam.csv` and examined for:

- dataset shape
- data types
- null/missing values
- duplicate rows
- class distribution

### 2. Exploratory data analysis

The notebook analyzes spam and ham message patterns using:

- label count plots
- word count histograms
- word clouds for spam and ham text
- character and word feature distributions
- punctuation and digit distribution plots

### 3. Text preprocessing

A custom `clean_text()` function is used to transform each SMS message by:

- converting to lowercase
- removing URLs and email addresses
- removing digits
- stripping punctuation
- removing stopwords
- applying WordNet lemmatization

### 4. Feature engineering

Additional engineered features are created before modeling:

- `char_count`
- `word_count`
- `num_digits`
- `num_punctuation`

These features help characterize spam-like patterns in the message text.

### 5. TF-IDF feature extraction

The cleaned SMS text is converted into numerical vectors using TF-IDF with:

- `max_features=5000`
- `ngram_range=(1, 2)`

This captures both individual words and common word pairs.

### 6. Linear SVM classification

The model uses a `Pipeline` combining TF-IDF and `LinearSVC`:

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('svm', LinearSVC())
])
```

### 7. Hyperparameter tuning

The model is optimized using `GridSearchCV` over:

```python
{'svm__C': [0.1, 1, 10]}
```

### 8. Cross-validation

The selected model is validated using 5-fold cross-validation.

### 9. Prediction function

The project includes a reusable prediction helper:

```python
def predict_sms(text):
    pred = model.predict([clean_text(text)])[0]
    return "Spam" if pred == 1 else "Ham"
```

Example usage:

```python
print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
# Spam

print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))
# Ham
```

## 📈 Model Performance

The model demonstrates strong classification performance on the test set.

### Example evaluation metrics

| Metric | Score |
| --- | ---: |
| Accuracy | 0.9836 |
| Precision | 0.9831 |
| Recall | 0.8855 |
| F1-score | 0.9317 |

### Classification report

```text
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       903
           1       0.98      0.89      0.93       131

    accuracy                           0.98      1034
   macro avg       0.98      0.94      0.96      1034
weighted avg       0.98      0.98      0.98      1034
```

### Hyperparameter tuning result

```text
Best Parameters: {'svm__C': 1}
Best CV Score: 0.9787182587666263
```

### Cross-validation result

```text
Mean Accuracy: 0.9783316887022269
```

## 📁 Project Structure

```text
sms_spam_svm/
├── SMS_Spam_SVM_raj.ipynb
├── spam.csv
├── README.md
└── .gitignore
```

> Note: The dataset file (`spam.csv`) is required to run the notebook. Download it from Kaggle and place it in the project folder before executing the notebook.

## 🚀 How to Run

### Option 1 — Google Colab

1. Open `SMS_Spam_SVM_raj.ipynb` in Google Colab.
2. Install any required dependencies if prompted.
3. Upload the `spam.csv` dataset.
4. Run the notebook cells in order.

### Option 2 — Local Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/rajsuprio/sms_spam_svm.git
cd sms_spam_svm
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
```

Start Jupyter:

```bash
jupyter notebook
```

Open the notebook:

```text
SMS_Spam_SVM_raj.ipynb
```

Then run all cells sequentially.

## 💡 Learning Outcomes

This project demonstrates:

- text classification
- natural language processing
- data cleaning
- exploratory data analysis
- TF-IDF vectorization
- feature engineering
- support vector machines
- model tuning
- cross-validation
- classification metrics
- confusion matrix analysis
- working with SMS spam datasets

## 🔮 Future Improvements

Possible improvements include:

- comparing SVM with Logistic Regression, Naive Bayes, and Random Forest
- testing alternative preprocessing strategies
- addressing class imbalance with class weighting or resampling
- experimenting with Word2Vec, GloVe, or transformer-based models
- deploying the model as a web app or API
- extending the project into a real-time SMS filtering system

## 👨‍💻 Author

**Suprio Chattapadhya Raj**

B.Sc. in Information and Communication Technology (ICT)
Bangladesh University of Professionals

---

⭐ If you found this project useful, consider starring the repository.
