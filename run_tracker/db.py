import sqlite3
import pandas as pd

DB_PATH = "run_log.db"

# ---------------------------------------------
# Connection helper
# ---------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------------------------
# Initialize database + tables
# ---------------------------------------------
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Runs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            run_type TEXT,
            distance REAL,
            duration_minutes REAL,
            avg_pace TEXT,
            splits TEXT,
            avg_hr INTEGER,
            max_hr INTEGER,
            hr_by_segment TEXT,
            cadence INTEGER,
            elevation_gain INTEGER,
            effort INTEGER,
            terrain TEXT,
            weather TEXT,
            how_felt TEXT,
            pain TEXT,
            sleep_hours REAL,
            stress INTEGER,
            nutrition_notes TEXT,
            vo2max REAL,
            hrv INTEGER,
            shoe_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Shoes table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT,
            start_date TEXT,
            retired INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

# ---------------------------------------------
# CRUD: Fetch
# ---------------------------------------------
def fetch_runs():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY date ASC", conn)
    conn.close()
    return df

def fetch_shoes(include_retired=False):
    conn = get_conn()
    if include_retired:
        df = pd.read_sql_query("SELECT * FROM shoes ORDER BY id DESC", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM shoes WHERE retired = 0 ORDER BY id DESC", conn)
    conn.close()
    return df

# ---------------------------------------------
# CRUD: Insert
# ---------------------------------------------
def insert_run(data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (
            date, run_type, distance, duration_minutes, avg_pace,
            splits, avg_hr, max_hr, hr_by_segment, cadence,
            elevation_gain, effort, terrain, weather, how_felt, pain,
            sleep_hours, stress, nutrition_notes, vo2max, hrv, shoe_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["date"],
        data["run_type"],
        data["distance"],
        data["duration_minutes"],
        data["avg_pace"],
        data["splits"],
        data["avg_hr"],
        data["max_hr"],
        data["hr_by_segment"],
        data["cadence"],
        data["elevation_gain"],
        data["effort"],
        data["terrain"],
        data["weather"],
        data["how_felt"],
        data["pain"],
        data["sleep_hours"],
        data["stress"],
        data["nutrition_notes"],
        data["vo2max"],
        data["hrv"],
        data["shoe_id"],
    ))
    conn.commit()
    conn.close()

# ---------------------------------------------
# CRUD: Update & Delete
# ---------------------------------------------
def update_run(run_id, data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE runs SET 
            date=?, run_type=?, distance=?, duration_minutes=?, avg_pace=?,
            splits=?, avg_hr=?, max_hr=?, hr_by_segment=?, cadence=?,
            elevation_gain=?, effort=?, terrain=?, weather=?, how_felt=?,
            pain=?, sleep_hours=?, stress=?, nutrition_notes=?, vo2max=?,
            hrv=?, shoe_id=? 
        WHERE id=?
    """, (
        data["date"], data["run_type"], data["distance"], data["duration_minutes"],
        data["avg_pace"], data["splits"], data["avg_hr"], data["max_hr"],
        data["hr_by_segment"], data["cadence"], data["elevation_gain"], 
        data["effort"], data["terrain"], data["weather"], data["how_felt"],
        data["pain"], data["sleep_hours"], data["stress"], data["nutrition_notes"],
        data["vo2max"], data["hrv"], data["shoe_id"], run_id
    ))
    conn.commit()
    conn.close()

def delete_run(run_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()
    conn.close()

# ---------------------------------------------
# Shoes insert / retire
# ---------------------------------------------
def insert_shoe(name, brand, start_date):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO shoes (name, brand, start_date, retired)
        VALUES (?, ?, ?, 0)
    """, (name, brand, start_date))
    conn.commit()
    conn.close()

def retire_shoe(shoe_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE shoes SET retired=1 WHERE id=?", (shoe_id,))
    conn.commit()
    conn.close()
