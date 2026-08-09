SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    screening_id TEXT PRIMARY KEY,
    case_id TEXT,
    created_at TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    quality_score REAL,
    quality_issues TEXT NOT NULL DEFAULT '[]',
    predicted_grade INTEGER,
    predicted_label TEXT,
    confidence REAL,
    referable_probability REAL,
    high_risk_probability REAL,
    priority TEXT NOT NULL,
    manual_review INTEGER NOT NULL,
    review_reasons TEXT NOT NULL DEFAULT '[]',
    model_version TEXT NOT NULL,
    processing_time_ms REAL NOT NULL,
    report_path TEXT,
    simulated INTEGER NOT NULL DEFAULT 0,
    probabilities TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_screenings_priority ON screenings(priority);
"""

