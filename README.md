# Heart Disease Prediction Using Machine Learning

A learning/demo web application built with Python, Flask, HTML, CSS, Bootstrap, SQLite, pandas, and scikit-learn Random Forest.

> **Important medical disclaimer:** This is an educational project, not a medical device. It must not be used to diagnose, treat, or rule out heart disease. The UCI Cleveland dataset is small and historical and does not establish clinical validity. A healthcare professional must interpret health information.

## Features

- User registration and login (passwords are hashed)
- Patient registration with full name, date of birth, gender, phone number, email, and generated patient ID
- Assessment form with the 10 requested fields: age, sex, chest pain type, resting BP, cholesterol, maximum heart rate, fasting blood sugar, exercise angina, Oldpeak, and ST slope
- Random Forest training script and model artifact
- Prediction history and generated text reports
- SQLite database with seven tables: `users`, `patients`, `assessments`, `model_versions`, `predictions`, `reports`, `audit_logs`
- Basic audit logging and ownership checks for user records

## Requirements

- Python 3.10 or newer recommended
- VS Code
- Internet connection for the first dataset download

## Run in VS Code (Windows PowerShell)

1. Extract this ZIP and open the `Heart_Disease_Prediction_Project_Updated` folder in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt instead:

   ```bat
   .venv\Scripts\activate.bat
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Download the public UCI Cleveland dataset and train the model:

   ```powershell
   python train_model.py
   ```

   This downloads `processed.cleveland.data` into `data/`, trains a Random Forest pipeline, and saves the model in `model/`. The dataset is not included in this ZIP; the training script retrieves it from the UCI Machine Learning Repository. If download fails, check internet access and retry.

6. Start the web application:

   ```powershell
   python app.py
   ```

7. Open `http://127.0.0.1:5000` in your browser. Register an account, add a patient, then create an assessment.

## Dataset and target

The script uses the processed Cleveland subset of the UCI Heart Disease dataset. The original `num` target is mapped to a binary target: `0` means no disease recorded (`num == 0`); `1` means disease recorded (`num > 0`). This mapping is only for reproducing a common educational classification task.

## Feature values in the form

- `sex`: 0 = female, 1 = male (dataset encoding)
- `cp`: 1–4 (dataset chest-pain categories)
- `fbs`: 0 = no, 1 = yes
- `exang`: 0 = no, 1 = yes
- `slope`: 1 = upsloping, 2 = flat, 3 = downsloping
- `oldpeak`: numeric ST-depression value

Patient registration stores full name, date of birth, gender, phone number, and email. Patient IDs are generated automatically. Existing SQLite databases are migrated to add the new patient fields without deleting their records.

These are dataset codes, not standalone medical interpretations. Do not use the form as a clinical assessment tool.

## Database tables

1. `users` — accounts and password hashes
2. `patients` — patient/demo records created by a user
3. `assessments` — feature values supplied for an assessment
4. `model_versions` — model metadata
5. `predictions` — one prediction per assessment in this implementation
6. `reports` — generated report text
7. `audit_logs` — recorded actions

The app uses SQLite by default so it runs locally without a separate database server. You can migrate the schema to MySQL later if your course submission requires MySQL.

## Before publishing to GitHub

- Do not commit real patient information, passwords, `.env` files, database files, trained model artifacts, or downloaded datasets unless you have checked the dataset's terms and your repository needs them.
- The `.gitignore` file excludes local database/model/dataset files by default.
- Change the development secret key and turn off Flask debug mode before any deployment.
- This starter is not production-ready: add CSRF protection, rate limiting, stronger deployment configuration, and a security review before any public deployment.

## Data source

UCI Machine Learning Repository, Heart Disease dataset:
https://archive.ics.uci.edu/dataset/45/heart+disease
