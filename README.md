# Project Overview

This project predicts whether a student is At-Risk or Not At-Risk using a machine learning model.

## Required Libraries

pandas, numpy, scikit-learn, matplotlib, seaborn, flask, flask-cors, joblib

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask flask-cors joblib
```

## Dataset

Download dataset here:
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

After downloading the dataset, create a folder named "Data" containing the dataset CSV file and then place the "Data" folder inside the "source_code" folder

## How to Run

1. Run the training script:

```bash
python models.py
```

2. Run the backend:

```bash
python app.py
```

3. Open:

```text
http://127.0.0.1:5000
```

## Notes

- Run `models.py` first and note close each figure so the script can run.
- `model.pkl` will be generated automatically.
- The dataset is not included in this submission folder.

