from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.database.connection import initialize

if __name__ == "__main__":
    config = load_config()
    initialize(config["paths"]["database"])
    print(f"Database initialized: {config['paths']['database']}")

