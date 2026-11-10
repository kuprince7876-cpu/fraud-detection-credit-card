# fraud-detection-credit-card

# 💳 Fraud Detection in Credit Card Transactions

Identify fraudulent credit card transactions using **anomaly detection** (Isolation Forest, Local Outlier Factor) and a supervised **XGBoost** classifier, with a **Streamlit** dashboard for live predictions.

## Objective
Detect rare fraudulent transactions (0.17% of the data) in a highly imbalanced dataset.

## Tools
Python, Pandas, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib/Seaborn, Streamlit

## Dataset
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - 284,807 transactions, 492 frauds.
Download `creditcard.csv` into the `data/` folder (see `data/README.md`).

## Workflow
1. Load the Kaggle credit card dataset
2. Preprocess: remove duplicates, scale `Time` and `Amount`, stratified 80/20 split
3. Balance the **training set only** with SMOTE
4. Apply **Isolation Forest** and **Local Outlier Factor** (unsupervised anomaly detection)
5. Train **XGBoost** classifier on the balanced data
6. Evaluate with confusion matrix, classification report and **ROC curve**
7. Dashboard for input/prediction (Streamlit) - ready to deploy as a web app

## Project structure
```
fraud-detection-credit-card/
├── fraud_detection.ipynb   # Jupyter notebook (full workflow)
├── train.py                # same workflow as a script, saves model + plots
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/                   # put creditcard.csv here
├── models/                 # saved model/scaler (created by train.py)
└── images/                 # confusion matrix, ROC curve, etc. (created by train.py)
```

## How to run
```bash
git clone https://github.com/<your-username>/fraud-detection-credit-card.git
cd fraud-detection-credit-card
pip install -r requirements.txt

# 1. add data/creditcard.csv, then train
python train.py          # or open fraud_detection.ipynb

# 2. launch the dashboard
streamlit run app.py
```

## Results
After running `train.py`, the outputs are saved in `images/`:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | XGBoost confusion matrix |
| `roc_curve.png` | ROC comparison: XGBoost vs Isolation Forest vs LOF |
| `feature_importance.png` | Top 15 XGBoost features |
| `class_distribution.png` | Normal vs fraud counts |

Add your own screenshots and the final scores (precision, recall, ROC-AUC) here after training.

![Confusion Matrix](images/confusion_matrix.png)
![ROC Curve](images/roc_curve.png)

## Deploy as a web app (free)
1. Commit `models/xgb_model.joblib`, `models/scaler.joblib`, `models/feature_names.joblib` and `data/sample_*.csv` (remove `models/` rules from `.gitignore` if you add any).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub repo, and select `app.py`.

## Author
Prince Kumar - BCA (AIML), Lamrin Tech Skills University
