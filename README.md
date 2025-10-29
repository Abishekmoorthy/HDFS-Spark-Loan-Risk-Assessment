# Peer-to-Peer Loan Outcome Analysis using Big Data Analytics

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-HDFS-yellow?style=flat&logo=apache&logoColor=white)](https://hadoop.apache.org/)

## 📖 Overview

This project implements a comprehensive big data analytics framework for predicting loan defaults and analyzing lending patterns on Peer-to-Peer (P2P) platforms. Built on the Apache Spark ecosystem, it processes large-scale financial data from the Lending Club platform to provide actionable insights for risk assessment and investment decision-making.

The framework addresses critical challenges in P2P lending by:
- **Predicting loan outcomes** with high accuracy using machine learning models
- **Generating business intelligence insights** through exploratory data analysis
- **Handling massive datasets** efficiently using distributed computing

---

## 🎯 Problem Statement

Peer-to-Peer lending platforms generate enormous volumes of transactional data, making it challenging for investors to assess borrower default risk accurately. Traditional analytics tools struggle with the scale and complexity of this data, leading to suboptimal lending decisions and increased financial risk.

This project addresses these challenges by leveraging Apache Spark's distributed computing capabilities to:
- Process 2.26 million loan records efficiently
- Build predictive models for loan default classification
- Extract meaningful patterns from historical lending data
- Provide data-driven recommendations for risk management

---

## 📊 Dataset

**Source**: [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

**File**: `accepted_2007_to_2018Q4.csv`

**Size**: 2.26 million loan records spanning 2007-2018

**Features**: 23 carefully selected attributes including:
- Loan characteristics (amount, interest rate, term)
- Borrower information (employment, income, credit grade)
- Loan status and performance metrics

---

## 🏗️ Architecture & Technology Stack

### Core Technologies
| Component | Technology |
|-----------|-----------|
| Big Data Framework | Apache Spark (PySpark) |
| Distributed Storage | Hadoop HDFS |
| Machine Learning | PySpark MLlib |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Query Engine | Spark SQL |

### Machine Learning Models
- **Logistic Regression**: Baseline binary classifier
- **Random Forest Classifier**: Ensemble learning for robust predictions
- **Multilayer Perceptron**: Deep learning approach for complex patterns

### Feature Engineering Pipeline
- **StringIndexer**: Converts categorical variables to numerical indices
- **OneHotEncoder**: Creates binary vectors for categorical features
- **VectorAssembler**: Combines features into single vector
- **MinMaxScaler**: Normalizes numerical features to [0,1] range

---

## 🔄 Methodology

### 1. Data Ingestion
- Dataset loaded directly from HDFS into Spark DataFrame
- Efficient distributed processing for large-scale data
- Path: `hdfs://localhost:9000/user/hduser/input/`

### 2. Data Preprocessing
- **Missing Value Treatment**: Rows with null values removed using `.na.drop()`
- **Target Engineering**: Multi-class `loan_status` converted to binary target
  - `0`: Fully Paid
  - `2`: Charged Off / Default
  - Other statuses filtered out for focused analysis

### 3. Exploratory Data Analysis (EDA)
Comprehensive visualizations to understand data characteristics:
- **Class Distribution**: Balance analysis between loan outcomes
- **Feature Distributions**: Histograms for loan amount, interest rate, and annual income
- **Correlation Analysis**: Heatmaps to identify feature relationships
- **Risk Patterns**: Boxplots revealing differences between paid and defaulted loans

### 4. Feature Engineering
Custom ML pipeline transforming raw data into model-ready features:
- Categorical encoding for `grade`, `term`, `home_ownership`
- Numerical scaling for consistent feature ranges
- Sparse vector creation for efficient computation

### 5. Model Training & Evaluation
- **Data Split**: Training, validation, and test sets
- **Metrics**: Accuracy and F1 Score for comprehensive evaluation
- **Comparison**: Performance analysis across multiple algorithms

### 6. Business Intelligence
Spark SQL queries extracting actionable insights:
- Geographic lending patterns by state
- Default rates by borrower demographics
- Interest rate analysis by loan purpose
- Credit grade and verification status correlations

---

## 📈 Key Findings

### Model Performance
- Successfully trained three classification models with strong predictive capabilities
- **Random Forest** and **Multilayer Perceptron** demonstrated superior performance
- Models effectively distinguish between high-risk and low-risk borrowers

### Data Insights
- **Class Imbalance**: Significantly more fully paid loans than charged off loans
- **Interest Rate Impact**: Defaulted loans show higher average interest rates
- **Credit Grade Correlation**: Lower grades (F, G) strongly associated with defaults
- **Home Ownership**: Patterns reveal varying default rates across ownership types

### Business Implications
- Data-driven risk assessment can reduce investor exposure
- Credit grades serve as reliable default predictors
- Interest rate pricing reflects underlying risk levels
- Geographic and demographic factors influence loan performance

---

## 🚀 Getting Started

### Prerequisites

**System Requirements**:
- Apache Hadoop cluster (HDFS enabled)
- Apache Spark 3.x
- Python 3.7 or higher
- Jupyter Notebook

**Python Libraries**:
```bash
pip install pyspark pandas numpy matplotlib seaborn
```

### Installation & Setup

**Step 1: Download Dataset**
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

**Step 2: Upload to HDFS**
```bash
hdfs dfs -mkdir -p /user/hduser/input/
hdfs dfs -put accepted_2007_to_2018Q4.csv /user/hduser/input/
```

**Step 3: Verify HDFS Upload**
```bash
hdfs dfs -ls /user/hduser/input/
```

**Step 4: Configure Spark Session**
Ensure the notebook points to your HDFS path:
```python
df = spark.read.csv("hdfs://localhost:9000/user/hduser/input/accepted_2007_to_2018Q4.csv", 
                    header=True, inferSchema=True)
```

### Running the Project

1. **Start Hadoop Services**
```bash
start-dfs.sh
start-yarn.sh
```

2. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

3. **Execute Analysis**
- Open `Bigdata-proj-LoanOutcomeAnalysis_final (1).ipynb`
- Run cells sequentially from top to bottom
- Monitor Spark UI at `http://localhost:4040` for job progress

---

## 📁 Project Structure

```
├── Bigdata-proj-LoanOutcomeAnalysis_final (1).ipynb  # Main analysis notebook
├── README.md                                          # Project documentation
├── data/                                              # Dataset directory (local)
│   └── accepted_2007_to_2018Q4.csv
├── models/                                            # Saved models (optional)
└── visualizations/                                    # Generated plots
```

---

## 🔍 Sample Outputs

### Model Evaluation Metrics
```
Logistic Regression
├── Training Accuracy: XX.XX%
├── Validation Accuracy: XX.XX%
└── Test F1 Score: X.XXX

Random Forest Classifier
├── Training Accuracy: XX.XX%
├── Validation Accuracy: XX.XX%
└── Test F1 Score: X.XXX
```

### Business Intelligence Queries
- Top 5 states by total loan amount
- Default rates by home ownership status
- Average interest rates by loan purpose
- Loan distribution by credit grade

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Model performance improvements
- Additional feature engineering techniques
- New business intelligence queries
- Documentation enhancements

---

## 📄 License

This project is available for educational and research purposes.

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

## 🙏 Acknowledgments

- **Lending Club** for providing the comprehensive loan dataset
- **Apache Spark Community** for the robust big data framework
- **Kaggle** for hosting and maintaining the dataset

---

## 📚 References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

**Built with ❤️ using Apache Spark and PySpark**
