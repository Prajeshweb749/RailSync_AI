"""
migrations/001_add_journey_instance_and_prediction_log.py
==========================================================
Migration:
1. Adds journey_instance_id to historical_runs and indexes it for rapid resolution.
2. Creates prediction_log table to continuously log live predictions and score them against actuals.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def run_migration():
    if not DB_URL:
        print("[!] DATABASE_URL not found.")
        return False

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("[*] 1. Adding journey_instance_id column to historical_runs if not exists...")
        cur.execute("""
            ALTER TABLE historical_runs 
            ADD COLUMN IF NOT EXISTS journey_instance_id VARCHAR(64);
        """)

        print("[*] 2. Populating journey_instance_id for historical runs...")
        cur.execute("""
            UPDATE historical_runs 
            SET journey_instance_id = train_number || '_' || run_date::text 
            WHERE journey_instance_id IS NULL;
        """)

        print("[*] 3. Creating index on historical_runs(journey_instance_id)...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_runs_journey_instance_id 
            ON historical_runs (journey_instance_id);
        """)

        print("[*] 4. Creating prediction_log table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                log_id SERIAL PRIMARY KEY,
                prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                journey_instance_id VARCHAR(64),
                train_number VARCHAR(16) NOT NULL,
                journey_start_date DATE NOT NULL,
                current_station VARCHAR(16),
                current_delay_min DOUBLE PRECISION,
                target_station VARCHAR(16) NOT NULL,
                scheduled_arrival TIMESTAMP WITHOUT TIME ZONE,
                predicted_arrival TIMESTAMP WITH TIME ZONE,
                predicted_delay_min DOUBLE PRECISION,
                actual_arrival TIMESTAMP WITHOUT TIME ZONE,
                actual_delay_min DOUBLE PRECISION,
                error_min DOUBLE PRECISION,
                model_source VARCHAR(32) DEFAULT 'xgboost_hybrid'
            );
        """)

        print("[*] 5. Creating indexes on prediction_log...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_log_train_date 
            ON prediction_log (train_number, journey_start_date);

            CREATE INDEX IF NOT EXISTS idx_pred_log_journey_instance 
            ON prediction_log (journey_instance_id);
        """)

        conn.commit()
        print("[+] Migration completed successfully!")
        return True

    except Exception as e:
        conn.rollback()
        print(f"[!] Migration failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_migration()
