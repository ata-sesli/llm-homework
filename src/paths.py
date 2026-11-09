"""Where large files live.

Downloaded data, processed splits, trained models, gensim-data vectors and the
Hugging Face cache all go under one storage root: TICKET_ROUTER_STORAGE from the
environment, else from a git-ignored .env file in the repo root, else the repo
root itself. Import this module before gensim.downloader or transformers so their
cache directories are set.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE_VAR = "TICKET_ROUTER_STORAGE"


def resolve_storage(environ, root):
    if environ.get(STORAGE_VAR):
        return Path(environ[STORAGE_VAR])
    dotenv = root / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == STORAGE_VAR and value.strip():
                return Path(value.strip())
    return root


STORAGE = resolve_storage(os.environ, ROOT)
RAW_DIR = STORAGE / "data" / "raw"
PROCESSED_DIR = STORAGE / "data" / "processed"
MODELS_DIR = STORAGE / "models"

os.environ.setdefault("GENSIM_DATA_DIR", str(STORAGE / "gensim-data"))
os.environ.setdefault("HF_HOME", str(STORAGE / "hf-cache"))
