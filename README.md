# Student Performance Risk Prediction

## What This Project Does

This project predicts whether a student is At-Risk or Not At-Risk using a machine learning model and a small web interface.

## Required Libraries

pandas, numpy, scikit-learn, matplotlib, seaborn, flask, flask-cors, joblib

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask flask-cors joblib
```

## Project Files

- `models.py` trains the model and generates `model.pkl`
- `app.py` starts the Flask backend
- `index.html`, `script.js`, and `styles.css` are the frontend files
- `Data/` contains the dataset files used by the project

## How To Run

1. Open a terminal in the repository root.
2. If `model.pkl` is missing, run:

```bash
python models.py
```

3. Start the app:

```bash
python app.py
```

4. Open your browser at:

```text
http://127.0.0.1:5000
```

## Notes

- Run `models.py` first if `model.pkl` does not exist.
- Then run `app.py`.
- The app depends on the dataset files in `Data/` and the generated `model.pkl`.
# Project Overview

This project predicts whether a student is At-Risk or Not At-Risk using a machine learning model.

## Required Libraries

pandas, numpy, scikit-learn, matplotlib, seaborn, flask, flask-cors, joblib

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask flask-cors joblib
```

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


