from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
CACHE_DIR = PACKAGE_DIR / "temp"

SUPPORTED_LANGUAGES = {
    'en': {
        'source_file': DATA_DIR / 'corpus_en.docx',
        'cache_file': CACHE_DIR / 'en_processed.txt',
        'display_name': 'English'
    },
    'ro': {
        'source_file': DATA_DIR / 'corpus_ro.docx',
        'cache_file': CACHE_DIR / 'ro_processed.txt',
        'display_name': 'Romanian'
    }
}

DEFAULT_LANGUAGE = 'en'

CACHE_METADATA_FILE = CACHE_DIR / 'metadata.json'

VALIDATE_CACHE_TIMESTAMPS = True
FORCE_REPROCESS_FLAG = False