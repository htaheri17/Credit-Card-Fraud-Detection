# IEEE-CIS Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)

A machine learning solution for detecting fraudulent online transactions using the IEEE-CIS Fraud Detection dataset from Kaggle.

## 📊 Project Overview

This project aims to predict the probability that an online transaction is fraudulent. The challenge involves working with a highly imbalanced dataset (~3.5% fraud rate) and over 400 features including transaction details, device information, and anonymized variables.

**Competition Link:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

### 🏆 Results

- **Public Leaderboard Score:** 0.9303 AUC
- **Private Leaderboard Score:** 0.8991 AUC
- **Validation Score:** 0.9538 AUC

## 🎯 Problem Statement

Credit card fraud detection is a critical challenge in the financial industry. This project uses machine learning to identify fraudulent transactions in real-time, helping to:
- Reduce financial losses for businesses and consumers
- Improve customer trust and security
- Minimize false positives that inconvenience legitimate customers
- Detect sophisticated fraud patterns

## 📁 Dataset

The dataset consists of two main tables:
- **Transaction Data:** Contains transaction details like amount, product code, and card information
- **Identity Data:** Contains device and network information (not available for all transactions)

**Key Statistics:**
- 590,540 training samples
- 506,691 test samples
- 434 features (after merging transaction and identity data)
- Highly imbalanced: 3.5% fraud rate
- Mix of numerical and categorical features

**Feature Categories:**
- Transaction features: TransactionAmt, ProductCD, card1-6, addr1-2
- Identity features: DeviceType, DeviceInfo, id_01-38
- Anonymized features: V1-339, C1-14, D1-15, M1-9

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/htaheri17/ieee-fraud-detection.git
cd ieee-fraud-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the data:
   - Go to [Kaggle Competition Data Page](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
   - Download all CSV files
   - Place them in the `data/` folder

## 🚀 Usage

### Running the Analysis

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebook/01_EDA_and_baseline.ipynb`

3. Run all cells to:
   - Load and explore the data
   - Perform exploratory data analysis
   - Engineer features
   - Train the baseline LightGBM model
   - Generate predictions and submission file

### Project Structure

```
ieee-fraud-detection/
├── data/                          # Raw data files (not tracked in git)
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── notebook/                      # Jupyter notebooks
│   └── 01_EDA_and_baseline.ipynb # Main analysis notebook
├── submission/                    # Model predictions
│   └── submission.csv
├── images/                        # Screenshots and visualizations
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 📈 Methodology

### 1. Exploratory Data Analysis (EDA)

**Key Findings:**
- **Target Distribution:** 96.5% legitimate vs 3.5% fraudulent transactions
- **Transaction Amounts:** 
  - Legitimate: Mean $134.51, Max $31,937
  - Fraudulent: Mean $149.24, Max $5,191
- **Card Patterns:** Very common cards (used 100-1000 times) have highest fraud rate at 3.92%
- **Missing Values:** Significant missing data in identity features (expected, as not all transactions have identity info)

### 2. Feature Engineering

Created additional features to improve model performance:

**Engineered Features:**
- `TransactionAmt_log`: Log-transformed transaction amount to handle skewness
- `TransactionAmt_decimal`: Decimal portion of amount (captures cent patterns)
- `P_email_is_null`: Binary indicator for missing purchaser email
- `R_email_is_null`: Binary indicator for missing recipient email
- `Transaction_hour`: Hour of day extracted from TransactionDT (0-23)
- `Transaction_day`: Day number from start of data collection period

**Total Features Used:** 398 features (common between train and test sets)

### 3. Model Development

**Algorithm:** LightGBM (Gradient Boosting Decision Tree)

**Why LightGBM?**
- Efficient handling of large datasets
- Native support for categorical features
- Excellent performance on tabular data
- Fast training speed
- Built-in handling of missing values

**Model Configuration:**
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

**Training Strategy:**
- 80/20 train-validation split
- Stratified sampling to maintain fraud ratio
- Early stopping with 50 rounds patience
- 1000 boosting rounds maximum

### 4. Model Performance

| Metric | Score |
|--------|-------|
| Training AUC | 0.9846 |
| Validation AUC | 0.9538 |
| Public Test AUC | 0.9303 |
| Private Test AUC | 0.8991 |

**Performance Analysis:**
- Strong validation score indicates good learning
- Gap between validation (0.9538) and public test (0.9303) suggests some overfitting
- Model generalizes well overall with 93% AUC on unseen data

## 📊 Key Insights

1. **Transaction Patterns:** Fraudulent transactions show distinct amount patterns with higher averages but lower maximums
2. **Card Reuse:** Cards used very frequently (100-1000 times) have the highest fraud rates, suggesting stolen card reuse
3. **Product Vulnerability:** Different product codes show varying fraud susceptibility
4. **Temporal Patterns:** Transaction timing features contribute to fraud detection
5. **Feature Importance:** A combination of transaction details, card information, and V-columns are most predictive

## 🔮 Future Improvements

### Short-term Enhancements
- [ ] Implement K-Fold Cross-Validation (5-fold) for more robust evaluation
- [ ] Handle class imbalance with focal loss or class weights
- [ ] Hyperparameter tuning using Optuna or Bayesian optimization
- [ ] Feature selection to remove low-importance features

### Advanced Feature Engineering
- [ ] Aggregation features: group statistics by card, address, email domain
- [ ] Create unique client ID from card1 + addr1 + D1 combination
- [ ] Velocity features: transaction frequency per card/email
- [ ] Time-based features: day of week, time since last transaction
- [ ] Target encoding for high-cardinality categorical features

### Model Improvements
- [ ] Ensemble multiple models (LightGBM + XGBoost + CatBoost)
- [ ] Stack models with meta-learner
- [ ] Try neural network architectures (TabNet, AutoInt)
- [ ] Implement pseudo-labeling for semi-supervised learning

### Analysis & Deployment
- [ ] Error analysis: identify systematic prediction failures
- [ ] SHAP values for model interpretability
- [ ] Create production-ready pipeline
- [ ] Build API for real-time fraud scoring

## 🛡️ Model Interpretability

Understanding model predictions is crucial for:
- Debugging and improving the model
- Building trust with stakeholders
- Ensuring regulatory compliance
- Detecting potential biases

Future work will include SHAP (SHapley Additive exPlanations) analysis to explain individual predictions.

## 📚 Learning Resources

**Recommended Reading:**
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Imbalanced Learning Techniques](https://imbalanced-learn.org/)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

**Relevant Papers:**
- "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., 2017)
- "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for hosting the competition
- [Vesta Corporation](https://www.vesta.io/) for providing the dataset
- IEEE Computational Intelligence Society for organizing the competition
- The Kaggle community for valuable discussions and insights

## 👤 Author

**Your Name**
- GitHub: [@htaheri17](https://github.com/htaheri17)
- Kaggle: [htaheri17](https://kaggle.com/htaheri17)
- Email: hussain.taheri@ufl.edu

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue on this repository
- Reach out via email

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Last Updated: January 2026*
