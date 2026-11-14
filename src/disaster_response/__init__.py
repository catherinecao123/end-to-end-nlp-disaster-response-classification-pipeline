"""End-to-end NLP disaster response classification pipeline.

ETL cleans the Figure Eight disaster message dataset into SQLite,
an sklearn pipeline (TF-IDF + message length + multi-output random
forest) classifies messages into 36 response categories, and a Flask
app serves predictions with dataset visualizations.
"""

__version__ = "2.0.0"

from disaster_response.etl import clean_data, load_data, save_data
from disaster_response.model import build_model, evaluate_model
from disaster_response.text import TextLengthExtractor, tokenize

__all__ = [
    "__version__",
    "build_model",
    "clean_data",
    "evaluate_model",
    "load_data",
    "save_data",
    "TextLengthExtractor",
    "tokenize",
]
