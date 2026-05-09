# IrisFlowerClassification
# Iris Flower Classification using Machine Learning

## Project Overview
This project uses Machine Learning to classify Iris flowers into three different species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

The model is trained using the Logistic Regression algorithm with the Iris dataset.

---

## Dataset
Dataset used:
https://www.kaggle.com/datasets/saurabh00007/iriscsv

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Features
- Data preprocessing
- Data visualization
- Pairplot visualization
- Correlation heatmap
- Model training
- Model evaluation
- Confusion matrix
- Accuracy score
- Sample prediction
- Model saving

---

## Project Structure

```text
IrisFlower/
│
├── iris.csv
├── iris_classification.py
├── requirements.txt
├── README.md
└── model.pkl
```

---

## Installation

Install all required libraries using:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## Run the Project

Run the Python file using:

```bash
python iris_classification.py
```

---

## Output
The project will generate:
- Dataset information
- Pairplot graph
- Correlation heatmap
- Accuracy score
- Classification report
- Confusion matrix
- Predicted flower species

It will also create:

```text
model.pkl
```

which stores the trained Machine Learning model.

---

## Machine Learning Algorithm Used
- Logistic Regression

---

## Sample Prediction

Input:

```python
[5.1, 3.5, 1.4, 0.2]
```

Predicted Output:

```text
Iris-setosa
```

---

## Accuracy
Expected model accuracy:

```text
95% - 100%
```

---

## Author
Chaitali Kar

---

## Conclusion
This project demonstrates how Machine Learning can be used to classify flower species based on their measurements using Python and Scikit-learn.
