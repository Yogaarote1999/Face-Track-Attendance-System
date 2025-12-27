import os
import io
import sqlite3
import pickle
import time
from datetime import datetime
from pathlib import Path
from openpyxl import Workbook
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    jsonify, send_file, flash
)

import pandas as pd
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import cv2
import calendar   # for monthly attendance calendar

# ---------------- CONFIG (YOUR CONFIGURATION) ----------------
APP_ROOT = Path(__file__).parent.resolve()
UPLOAD_FOLDER = APP_ROOT / "static" / "uploads"
FACES_FOLDER = APP_ROOT / "faces"
CSV_FILE = APP_ROOT / "attendance_log.csv"
DB_FILE = APP_ROOT / "users.db"
MODEL_FILE = APP_ROOT / "attendance_model.pkl"
FACENET_FILE = APP_ROOT / "facenet_model.pth"
EMBED_FILE = APP_ROOT / "embeddings.pkl"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

ADMIN_EMAIL = "admin@gmail.com"   # <<---- YOUR ADMIN EMAIL

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.60))
FLASK_SECRET = os.environ.get("FLASK_SECRET", "change_this_secret_in_prod")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET

# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        embedding BLOB
    )
    """)

    # Attendance table
    c.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        username TEXT,
        email TEXT,
        image TEXT,
        date TEXT,
        time TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Ensure embedding column exists (for old DBs)
    try:
        c.execute("ALTER TABLE users ADD COLUMN embedding BLOB")
    except sqlite3.OperationalError:
        # column already exists, ignore
        pass

    conn.commit()
    conn.close()

def ensure_admin_user():
    """
    Ensure that ADMIN_EMAIL exists as a user.
    If not present, create it with password 'admin123'.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email=?", (ADMIN_EMAIL,))
    row = c.fetchone()
    if not row:
        pwd_hash = generate_password_hash("admin123")
        c.execute(
            "INSERT INTO users (username, email, password) VALUES (?,?,?)",
            ("ADMIN", ADMIN_EMAIL, pwd_hash)
        )
        print(f"[INFO] Created admin user {ADMIN_EMAIL} with password 'admin123'")
    conn.commit()
    conn.close()

init_db()
ensure_admin_user()

# ---------------- FACENET + MTCNN ----------------
try:
    facenet = InceptionResnetV1(pretrained=None).eval().to(DEVICE)
except Exception as e:
    raise RuntimeError("Failed to instantiate InceptionResnetV1: " + str(e))

if not Path(FACENET_FILE).exists():
    raise FileNotFoundError(f"Place facenet_model.pth at {FACENET_FILE}")

raw_state = torch.load(str(FACENET_FILE), map_location=DEVICE)
if isinstance(raw_state, dict) and 'state_dict' in raw_state and isinstance(raw_state['state_dict'], dict):
    raw_state = raw_state['state_dict']

model_state = facenet.state_dict()
filtered_state = {}
for k, v in raw_state.items():
    if k in model_state:
        try:
            if hasattr(v, "size") and v.size() == model_state[k].size():
                filtered_state[k] = v
        except Exception:
            filtered_state[k] = v
facenet.load_state_dict(filtered_state, strict=False)
facenet.eval()

mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE, keep_all=False)

# ---------------- EMBEDDINGS FILE (BACKUP) ----------------
emb_lock = threading.Lock()

def load_embeddings_file():
    if EMBED_FILE.exists():
        try:
            with open(EMBED_FILE, "rb") as f:
                d = pickle.load(f)
            # ensure numpy arrays
            for k, v in list(d.items()):
                d[k] = np.asarray(v, dtype=np.float32)
            return d
        except Exception as e:
            print("Failed to load embeddings.pkl:", e)
            return {}
    return {}

def save_embeddings_file(d):
    # convert to list for stable pickling
    safe = {k: np.asarray(v, dtype=np.float32).tolist() for k, v in d.items()}
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(safe, f)

embeddings_cache = load_embeddings_file()
print("Loaded embeddings from file:", len(embeddings_cache), "users")

# ---------------- STATE FOR RATE LIMIT ----------------
mark_lock = threading.Lock()
next_allowed_mark = {}           # email -> timestamp allowed next
MIN_INTERVAL_AFTER_MARK = 3      # seconds cooldown after success

# ---------------- UTILS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_image_from_bytes(image_bytes, username="unknown"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{username}_{timestamp}.jpg")
    path = UPLOAD_FOLDER / filename
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))[:, :, ::-1]
    cv2.imwrite(str(path), img)
    return filename, str(path)


def embedding_from_bytes(image_bytes):
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # detect & crop face
    face = None
    try:
        face = mtcnn(pil)  # returns tensor (3,160,160) or None
    except Exception as e:
        print("MTCNN error:", e)
        face = None

    if face is None:
        raise ValueError("No face detected")

    tensor = face.unsqueeze(0).to(DEVICE).float()  # (1,3,160,160)
    with torch.no_grad():
        emb = facenet(tensor).cpu().numpy()[0].astype(np.float32)
    return emb

def get_user_by_email(email):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username, email, embedding FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row

def store_user_embedding(email, emb: np.ndarray):
    # store in DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET embedding=? WHERE email=?", (pickle.dumps(emb), email))
    conn.commit()
    conn.close()

    # also update backup file
    with emb_lock:
        embeddings_cache[email] = emb
        save_embeddings_file(embeddings_cache)

def append_attendance_db(user_id, username, email, image_filename, date_str, time_str):
    with mark_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT INTO attendance (user_id, username, email, image, date, time) VALUES (?,?,?,?,?,?)",
            (user_id, username, email, image_filename, date_str, time_str)
        )
        conn.commit()
        conn.close()

def append_csv_record(username, email, image_filename, date_str, time_str):

            return

def already_marked_today_email(email, date_str):
    """Check in DB if this email already has attendance today."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM attendance WHERE email=? AND date=?", (email, date_str))
    row = c.fetchone()
    conn.close()
    return row and row[0] > 0

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    # go to login by default
    return redirect(url_for("login"))

# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        face_file = request.files.get("face_image")

        if not username or not email or not password or not face_file:
            return render_template("register.html", error="All fields are required including face image.")

        # prevent registering admin email again
        if email == ADMIN_EMAIL:
            return render_template("register.html", error="This email is reserved for admin.")

        hashed = generate_password_hash(password)

        # create user row
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                (username, email, hashed)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Email already registered")
        conn.close()

        # save raw face image under faces/email
        user_face_dir = FACES_FOLDER / email
        user_face_dir.mkdir(parents=True, exist_ok=True)
        face_bytes = face_file.read()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{ts}.jpg"
        img_path = user_face_dir / img_name
        with open(img_path, "wb") as f:
            f.write(face_bytes)

        # embedding generation
        try:
            emb = embedding_from_bytes(face_bytes)
            store_user_embedding(email, emb)
            print("Stored embedding for:", email)
        except Exception as e:
            print("Failed to create embedding during registration:", e)
            return render_template(
                "register.html",
                error="Could not detect your face clearly. Please try again with better lighting and straight face."
            )

        return redirect(url_for("login"))

    return render_template("register.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, username, password FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()

        if row and check_password_hash(row[2], password):
            session["user_id"] = int(row[0])
            session["username"] = row[1]
            session["email"] = email
            session["is_admin"] = (email == ADMIN_EMAIL)

            if session["is_admin"]:
                return redirect(url_for("admin"))   # ← AUTO OPEN ADMIN PANEL

            return redirect(url_for("dashboard"))   # ← Students go to dashboard

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")
# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    username = session.get("username")
    after_attendance = request.args.get("after_attendance")
    is_admin = session.get("is_admin", False)

    # For students: show auto logout msg after attendance
    if after_attendance and not is_admin:
        return render_template("dashboard.html", username=username, auto_logout=True, is_admin=is_admin)

    # For admin and students: same template, but admin will not show camera (handled in HTML)
    return render_template("dashboard.html", username=username, is_admin=is_admin)

# ---------- CAPTURE (ATTENDANCE) ----------
@app.route("/capture", methods=["POST"])
def capture():
    if "user_id" not in session:
        return jsonify({"status": "not_logged_in"})

    # Admin should not mark attendance via camera
    if session.get("is_admin"):
        return jsonify({"status": "admin_no_capture"})

    user_id = session["user_id"]
    expected_email = session.get("email")
    expected_username = session.get("username")

    file = request.files.get("face_image")
    if not file:
        return jsonify({"status": "no_file"})

    img_bytes = file.read()

    # convert to embedding
    try:
        emb = embedding_from_bytes(img_bytes)
    except:
        return jsonify({"status": "no_face"})

    # get stored embedding
    user_row = get_user_by_email(expected_email)
    if not user_row:
        return jsonify({"status": "user_not_registered"})

    stored_blob = user_row[3]
    stored_emb = pickle.loads(stored_blob) if stored_blob else None

    if stored_emb is None and expected_email in embeddings_cache:
        stored_emb = np.asarray(embeddings_cache[expected_email], dtype=np.float32)

    if stored_emb is None:
        return jsonify({"status": "user_not_registered"})

    # ================= FACE MATCH FIRST =================
    sim = float(np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-10))
    print(f"[DEBUG] similarity for {expected_email}: {sim:.4f}")

    if sim < SIMILARITY_THRESHOLD:
        return jsonify({
            "status": "face_mismatch",
            "redirect": "/logout"
        })
    # =====================================================

    # if face is matched, then check duplicate attendance
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if already_marked_today_email(expected_email, date_str):
        return jsonify({
            "status": "already",
            "username": expected_username,
            "email": expected_email,
            "date": date_str,
            "time": time_str,
            "redirect": "/logout"
        })

    # save attendance
    filename, saved_path = save_image_from_bytes(img_bytes, expected_username or expected_email)
    append_attendance_db(user_id, expected_username, expected_email, filename, date_str, time_str)
    append_csv_record(expected_username, expected_email, filename, date_str, time_str)

    return jsonify({
        "status": "marked",
        "username": expected_username,
        "email": expected_email,
        "date": date_str,
        "time": time_str,
        "redirect": "/logout"
    })

# ---------- ADMIN DASHBOARD: LIST STUDENTS + COUNTS ----------
@app.route("/admin")
def admin():
    # 1) User must be logged in
    if "user_id" not in session:
        return redirect(url_for("login"))

    # 2) Must be admin, not student
    if not session.get("is_admin"):
        return redirect(url_for("dashboard"))

    # If both checks pass → allow admin panel
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT username, email, image, date, time FROM attendance", conn)
    conn.close()
    attendance = df.to_dict(orient="records")
    return render_template("admin.html", attendance=attendance)

@app.route("/students")
def students():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        return redirect(url_for("dashboard"))

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT username, email FROM users", conn)
    conn.close()

    students = df.to_dict(orient="records")
    return render_template("students.html", students=students)

@app.route("/download_all_attendance")
def download_all_attendance():
    if not session.get("is_admin"):
        return "Access Denied", 403

    conn = sqlite3.connect(DB_FILE)

    today = datetime.now()
    month = today.month
    year = today.year
    total_days = calendar.monthrange(year, month)[1]

    df_users = pd.read_sql_query("SELECT id, username, email FROM users", conn)
    df_att = pd.read_sql_query("SELECT user_id, date, time FROM attendance", conn)

    final_rows = []

    for _, user in df_users.iterrows():
        user_id = user["id"]

        # Get attendance for this month
        user_att = df_att[
            (df_att["user_id"] == user_id) &
            (df_att["date"].str.startswith(f"{year}-{month:02}"))
        ]

        present_days = user_att["date"].nunique()
        total_work = total_days
        absent = total_work - present_days

        # Attendance present rows
        if len(user_att) > 0:
            for _, log in user_att.iterrows():
                final_rows.append([
                    user["username"],
                    user["email"],
                    log["date"],
                    log["time"],
                    total_work,
                    present_days,
                    absent
                ])
        else:
            # No attendance entry
            final_rows.append([
                user["username"],
                user["email"],
                "-",
                "-",
                total_work,
                present_days,
                absent
            ])

    file_path = f"ALL_ATTENDANCE_{year}_{month}.xlsx"
    df = pd.DataFrame(
        final_rows,
        columns=[
            "Username",
            "Email",
            "Date",
            "Time",
            "Total Working Days",
            "Present Days",
            "Absent Days"
        ]
    )
    df.to_excel(file_path, index=False)

    conn.close()
    return send_file(file_path, as_attachment=True)

# ---------- DELETE STUDENT ----------
@app.route("/delete_student/<email>", methods=["POST"])
def delete_student(email):
    if not session.get("is_admin"):
        return jsonify({"status": "error", "message": "Access denied"})

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # find user_id
        c.execute("SELECT id FROM users WHERE email=?", (email,))
        row = c.fetchone()
        user_id = row[0] if row else None

        if user_id:
            c.execute("DELETE FROM attendance WHERE user_id=?", (user_id,))

        c.execute("DELETE FROM users WHERE email=?", (email,))
        conn.commit()
        conn.close()

        # also clean from CSV
        if CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE)
            df = df[df["email"] != email]
            df.to_csv(CSV_FILE, index=False)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ---------- EDIT STUDENT (username + email + password + face) ----------
@app.route("/edit_student/<email>", methods=["GET", "POST"])
def edit_student(email):
    if not session.get("is_admin"):
        return "Access Denied", 403

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    if request.method == "POST":
        new_name = request.form.get("username").strip()
        new_email = request.form.get("email").strip().lower()
        new_password = request.form.get("password", "")
        new_face_file = request.files.get("face_image")

        # update username + email
        if new_name and new_email:
            c.execute("UPDATE users SET username=?, email=? WHERE email=?", (new_name, new_email, email))

        # update password if provided
        if new_password:
            pwd_hash = generate_password_hash(new_password)
            c.execute("UPDATE users SET password=? WHERE email=?", (pwd_hash, new_email))

        # update embedding if new face uploaded
        if new_face_file and new_face_file.filename:
            face_bytes = new_face_file.read()

            # save updated face image in faces folder
            user_face_dir = FACES_FOLDER / new_email
            user_face_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"updated_{ts}.jpg"
            img_path = user_face_dir / img_name
            with open(img_path, "wb") as f:
                f.write(face_bytes)

            try:
                new_emb = embedding_from_bytes(face_bytes)
                store_user_embedding(new_email, new_emb)
                print("Updated embedding for:", new_email)
            except Exception as e:
                print("Error updating embedding:", e)

        conn.commit()
        conn.close()
        return redirect(url_for("admin"))

    c.execute("SELECT username, email FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "User not found", 404

    user_data = {"username": row[0], "email": row[1]}
    return render_template("edit_student.html", user=user_data)

# ---------- ATTENDANCE REPORT (LIST ENTRIES) ----------
@app.route("/attendance_report/<email>")
def attendance_report(email):
    if not session.get("is_admin"):
        return "Access Denied", 403

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT date, time, image FROM attendance WHERE email=? ORDER BY date, time",
        conn,
        params=(email,)
    )
    conn.close()

    records = df.to_dict(orient="records")
    return render_template("attendance_report.html", email=email, records=records)

# ---------- ATTENDANCE COUNT ----------
@app.route("/attendance_count/<email>")
def attendance_count(email):
    if not session.get("is_admin"):
        return "Access Denied", 403

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM attendance WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    count = row[0] if row else 0
    return render_template("attendance_count.html", email=email, count=count)

# ---------- MONTHLY CALENDAR (PRESENT / ABSENT) ----------
@app.route("/attendance_calendar/<email>", methods=["GET", "POST"])
def attendance_calendar(email):
    if not session.get("is_admin"):
        return "Access Denied", 403

    today = datetime.now()
    selected_year = int(request.form.get("year", today.year))
    selected_month = int(request.form.get("month", today.month))

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT date, time FROM attendance WHERE email=?",
        conn,
        params=(email,)
    )
    conn.close()

    if df.empty:
        calendar_data = []
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df_month = df[df["date"].dt.month == selected_month]
        present_days = set(df_month["date"].dt.day.tolist())
        total_days = calendar.monthrange(selected_year, selected_month)[1]

        calendar_data = []
        for day in range(1, total_days + 1):
            if day in present_days:
                time_val = df_month[df_month["date"].dt.day == day]["time"].iloc[0]
                calendar_data.append({"day": day, "status": "Present", "time": time_val})
            else:
                calendar_data.append({"day": day, "status": "Absent", "time": "-"})
    return render_template(
        "attendance_calendar.html",
        email=email,
        year=selected_year,
        month=selected_month,
        data=calendar_data
    )

# ---------- EXISTING ATTENDANCE CSV HELPERS (LISTENING FOR JS) ----------
@app.route("/delete_record/<int:record_index>", methods=["POST"])
def delete_record(record_index):
    try:
        if not CSV_FILE.exists():
            return jsonify({"status": "error", "message": "No records"})

        df = pd.read_csv(CSV_FILE)
        if record_index < 0 or record_index >= len(df):
            return jsonify({"status": "error", "message": "Invalid record index"})

        row = df.iloc[record_index]
        username = row.get("username", "")
        email = row.get("email", "")
        image = row.get("image", "")
        date_str = row.get("date", "")
        time_str = row.get("time", "")

        df = df.drop(df.index[record_index]).reset_index(drop=True)
        df.to_csv(CSV_FILE, index=False)

        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("""
                DELETE FROM attendance
                WHERE username=? AND email=? AND image=? AND date=? AND time=?
            """, (username, email, image, date_str, time_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print("DB delete error:", e)

        return jsonify({"status": "success"})
    except Exception as e:
        print("delete_record error:", e)
        return jsonify({"status": "error", "message": str(e)})
    
@app.route("/attendance_data")
def attendance_data():
    if "user_id" not in session:
        return jsonify([])
    if not CSV_FILE.exists():
        return jsonify([])
    df = pd.read_csv(CSV_FILE)
    for col in ["username", "email", "image", "date", "time"]:
        if col not in df.columns:
            df[col] = ""
    return jsonify(df.to_dict(orient="records"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
