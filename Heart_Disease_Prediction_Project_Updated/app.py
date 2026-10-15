import json
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "instance" / "heart_disease.db"
MODEL_PATH = BASE_DIR / "model" / "heart_disease_model.joblib"

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-development-key-before-deployment"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "thalach", "fbs", "exang", "oldpeak", "slope"
]
FEATURE_LABELS = {
    "age": "Age (years)", "sex": "Sex", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)", "chol": "Cholesterol (mg/dL)",
    "thalach": "Maximum heart rate (bpm)", "fbs": "Fasting blood sugar",
    "exang": "Exercise angina", "oldpeak": "Oldpeak", "slope": "ST slope"
}


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            age INTEGER,
            sex INTEGER,
            contact TEXT,
            email TEXT,
            date_of_birth TEXT,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users(user_id)
        );
        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            input_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        CREATE TABLE IF NOT EXISTS model_versions (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL UNIQUE,
            algorithm TEXT NOT NULL,
            dataset_name TEXT,
            trained_at TEXT NOT NULL,
            metrics_json TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL UNIQUE,
            model_id INTEGER NOT NULL,
            predicted_class INTEGER NOT NULL,
            probability REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id),
            FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
        );
        CREATE TABLE IF NOT EXISTS reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            generated_by INTEGER NOT NULL,
            report_text TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
            FOREIGN KEY (generated_by) REFERENCES users(user_id)
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id INTEGER,
            details TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """)
        # Migrate databases created by earlier versions without deleting existing records.
        patient_columns = {row["name"] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
        if "email" not in patient_columns:
            conn.execute("ALTER TABLE patients ADD COLUMN email TEXT")
        if "date_of_birth" not in patient_columns:
            conn.execute("ALTER TABLE patients ADD COLUMN date_of_birth TEXT")
        conn.execute("""
            INSERT OR IGNORE INTO model_versions
            (model_name, version, algorithm, dataset_name, trained_at, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Heart Disease Screening Model", "1.0", "Random Forest",
              "UCI Cleveland Heart Disease", datetime.now().isoformat(timespec="seconds"),
              json.dumps({"note": "See training_metrics.json after running train_model.py"})))


def log_action(user_id, action, entity_type=None, entity_id=None, details=None):
    with get_db() as conn:
        # A stale browser session can refer to a user deleted from the database.
        # Store NULL in the optional audit user_id rather than triggering a foreign-key error.
        if user_id is not None:
            exists = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if exists is None:
                user_id = None
        conn.execute("""
            INSERT INTO audit_logs (user_id, action, entity_type, entity_id, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, action, entity_type, entity_id, details,
              datetime.now().isoformat(timespec="seconds")))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_globals():
    return {"feature_labels": FEATURE_LABELS}


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not full_name or not email or len(password) < 8:
            flash("Enter your name, a valid email, and a password of at least 8 characters.", "danger")
            return render_template("register.html")
        try:
            with get_db() as conn:
                cur = conn.execute(
                    "INSERT INTO users (full_name, email, password_hash, role, created_at) VALUES (?, ?, ?, 'user', ?)",
                    (full_name, email, generate_password_hash(password),
                     datetime.now().isoformat(timespec="seconds"))
                )
                user_id = cur.lastrowid
            session.clear()
            session["user_id"] = user_id
            session["full_name"] = full_name
            log_action(user_id, "REGISTER", "USERS", user_id)
            return redirect(url_for("dashboard"))
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "danger")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["user_id"]
            session["full_name"] = user["full_name"]
            log_action(user["user_id"], "LOGIN", "USERS", user["user_id"])
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    user_id = session.get("user_id")
    if user_id:
        log_action(user_id, "LOGOUT", "USERS", user_id)
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    uid = session["user_id"]
    with get_db() as conn:
        patients = conn.execute(
            "SELECT * FROM patients WHERE created_by = ? ORDER BY patient_id DESC", (uid,)
        ).fetchall()
        history = conn.execute("""
            SELECT pr.prediction_id, pr.predicted_class, pr.probability, pr.created_at,
                   a.assessment_id, p.patient_id, p.full_name AS patient_name
            FROM predictions pr
            JOIN assessments a ON a.assessment_id = pr.assessment_id
            JOIN patients p ON p.patient_id = a.patient_id
            WHERE p.created_by = ?
            ORDER BY pr.prediction_id DESC LIMIT 10
        """, (uid,)).fetchall()
    return render_template("dashboard.html", patients=patients, history=history)


@app.route("/patients", methods=["GET", "POST"])
@login_required
def patients():
    # Do not allow a stale login session to insert records for a nonexistent user.
    with get_db() as conn:
        current_user = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (session.get("user_id"),)
        ).fetchone()
    if current_user is None:
        session.clear()
        flash("Your session is no longer valid. Please log in again.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("full_name", "").strip()
        dob = request.form.get("date_of_birth", "").strip()
        sex_text = request.form.get("sex", "").strip()
        phone = request.form.get("contact", "").strip()
        email = request.form.get("email", "").strip().lower()
        if not name or not dob or not sex_text or not phone or not email:
            flash("Please complete all patient fields.", "danger")
        else:
            try:
                from datetime import date
                birth_date = date.fromisoformat(dob)
                today = date.today()
                if birth_date > today:
                    raise ValueError
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                sex = int(sex_text)
                if not 1 <= age <= 120 or sex not in (0, 1):
                    raise ValueError
                with get_db() as conn:
                    cur = conn.execute("""
                        INSERT INTO patients (full_name, age, sex, contact, email, date_of_birth, created_by, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (name, age, sex, phone, email, dob, session["user_id"],
                          datetime.now().isoformat(timespec="seconds")))
                    patient_id = cur.lastrowid
                log_action(session["user_id"], "CREATE", "PATIENTS", patient_id)
                flash(f"Patient registered successfully. Patient ID: PT-{datetime.now().year}-{patient_id:04d}", "success")
                return redirect(url_for("patients"))
            except ValueError:
                flash("Enter a valid date of birth and gender.", "danger")
            except sqlite3.IntegrityError:
                flash("Patient could not be saved because a related record is invalid. Please log in again.", "danger")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE created_by = ? ORDER BY patient_id DESC",
            (session["user_id"],)
        ).fetchall()
    return render_template("patients.html", patients=rows, today=datetime.now().date().isoformat())


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    with get_db() as conn:
        patient_rows = conn.execute(
            "SELECT * FROM patients WHERE created_by = ? ORDER BY full_name",
            (session["user_id"],)
        ).fetchall()
    if request.method == "POST":
        if not MODEL_PATH.exists():
            flash("The trained model is missing. Run `python train_model.py` first.", "danger")
            return redirect(url_for("predict"))
        try:
            patient_id = int(request.form["patient_id"])
            patient = next((p for p in patient_rows if p["patient_id"] == patient_id), None)
            if patient is None:
                flash("Select a patient you created.", "danger")
                return redirect(url_for("predict"))
            values = {}
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                if raw == "":
                    raise ValueError(f"Please enter {FEATURE_LABELS[feature]}.")
                values[feature] = float(raw)
            if not (1 <= values["age"] <= 120):
                raise ValueError("Age must be between 1 and 120.")
            if values["sex"] not in (0, 1) or values["fbs"] not in (0, 1) or values["exang"] not in (0, 1):
                raise ValueError("Sex, fasting blood sugar, and exercise-induced angina must be 0 or 1.")
            if values["cp"] not in (1, 2, 3, 4):
                raise ValueError("Chest pain type must be 1, 2, 3, or 4.")
            if values["slope"] not in (1, 2, 3):
                raise ValueError("ST slope must be 1, 2, or 3.")
            if any(v < 0 for v in values.values()):
                raise ValueError("Values cannot be negative.")
            model = joblib.load(MODEL_PATH)
            frame = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
            predicted = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][1]) if hasattr(model, "predict_proba") else None
            now = datetime.now().isoformat(timespec="seconds")
            with get_db() as conn:
                cur = conn.execute(
                    "INSERT INTO assessments (patient_id, input_data, created_at) VALUES (?, ?, ?)",
                    (patient_id, json.dumps(values), now)
                )
                assessment_id = cur.lastrowid
                model_row = conn.execute("SELECT model_id FROM model_versions WHERE version='1.0'").fetchone()
                model_id = model_row["model_id"]
                cur = conn.execute("""
                    INSERT INTO predictions (assessment_id, model_id, predicted_class, probability, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (assessment_id, model_id, predicted, probability, now))
                prediction_id = cur.lastrowid
                label = "Higher likelihood in this model's classification" if predicted else "Lower likelihood in this model's classification"
                probability_text = f"{probability:.1%}" if probability is not None else "Not available"
                report = (
                    f"Patient: {patient['full_name']}\\nAssessment ID: {assessment_id}\\n"
                    f"Model result: {label}\\n"
                    f"Model probability for positive class: {probability_text}\\n"
                    "This is an educational screening output, not a medical diagnosis. "
                    "A qualified healthcare professional must interpret results with clinical context."
                )
                conn.execute(
                    "INSERT INTO reports (prediction_id, generated_by, report_text, generated_at) VALUES (?, ?, ?, ?)",
                    (prediction_id, session["user_id"], report, now)
                )
            log_action(session["user_id"], "CREATE", "PREDICTIONS", prediction_id)
            return redirect(url_for("result", prediction_id=prediction_id))
        except (ValueError, KeyError) as exc:
            flash(str(exc) or "Please check all entered values.", "danger")
        except Exception as exc:
            app.logger.exception("Prediction failed")
            flash("Prediction could not be completed. Check the model and inputs, then try again.", "danger")
    return render_template("predict.html", patients=patient_rows, features=FEATURES)


@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    with get_db() as conn:
        row = conn.execute("""
            SELECT pr.*, p.full_name AS patient_name, p.created_by,
                   a.input_data, r.report_id, r.report_text, r.generated_at AS report_generated_at
            FROM predictions pr
            JOIN assessments a ON a.assessment_id = pr.assessment_id
            JOIN patients p ON p.patient_id = a.patient_id
            LEFT JOIN reports r ON r.prediction_id = pr.prediction_id
            WHERE pr.prediction_id = ?
        """, (prediction_id,)).fetchone()
    if not row or row["created_by"] != session["user_id"]:
        flash("Prediction not found.", "danger")
        return redirect(url_for("dashboard"))
    log_action(session["user_id"], "VIEW", "PREDICTIONS", prediction_id)
    return render_template("result.html", result=row, input_data=json.loads(row["input_data"]))


@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT pr.*, p.full_name AS patient_name
            FROM predictions pr
            JOIN assessments a ON a.assessment_id = pr.assessment_id
            JOIN patients p ON p.patient_id = a.patient_id
            WHERE p.created_by = ? ORDER BY pr.prediction_id DESC
        """, (session["user_id"],)).fetchall()
    return render_template("history.html", history=rows)


@app.route("/report/<int:report_id>")
@login_required
def report(report_id):
    with get_db() as conn:
        row = conn.execute("""
            SELECT r.*, pr.prediction_id, p.created_by, p.full_name AS patient_name
            FROM reports r
            JOIN predictions pr ON pr.prediction_id = r.prediction_id
            JOIN assessments a ON a.assessment_id = pr.assessment_id
            JOIN patients p ON p.patient_id = a.patient_id
            WHERE r.report_id = ?
        """, (report_id,)).fetchone()
    if not row or row["created_by"] != session["user_id"]:
        flash("Report not found.", "danger")
        return redirect(url_for("dashboard"))
    return render_template("report.html", report=row)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
