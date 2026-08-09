from src.database.connection import initialize
from src.database.repository import ScreeningRepository

def test_database_crud(tmp_path):
    path = str(tmp_path / "db.sqlite3"); initialize(path); repo = ScreeningRepository(path)
    record = repo.insert({
        "screening_id": "abc", "original_filename": "eye.png", "file_hash": "hash",
        "priority": "ROUTINE", "manual_review": False, "model_version": "test",
        "processing_time_ms": 12, "probabilities": [.8,.1,.05,.03,.02],
    })
    assert record["screening_id"] == "abc"
    assert repo.summary()["total_screenings"] == 1
    assert repo.delete("abc")
    assert repo.get("abc") is None

