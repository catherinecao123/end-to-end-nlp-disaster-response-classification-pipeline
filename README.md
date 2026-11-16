# End-to-End NLP Disaster Response Classification Pipeline

![CI](https://github.com/catherinecao123/end-to-end-nlp-disaster-response-classification-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)

An end-to-end NLP pipeline that classifies disaster messages into 36 response
categories, so emergency workers can route each message to the right relief
agency. It covers the full loop: **ETL** (CSV → clean SQLite) → **ML**
(TF-IDF + message-length features → multi-output random forest tuned with
grid search) → **serving** (Flask web app with Plotly dashboards).

> A rebuild of a 2021 Udacity Data Scientist Nanodegree project, modernized
> for current library versions with a clean package layout, tests, and CI.
> The original code is preserved in the repository history.

![Web app example](disasterapp_example.png)

## How it works

```text
disaster_messages.csv ─┐
                       ├─► ETL ─► DisasterResponse.db ─► train ─► classifier.pkl ─► Flask app
disaster_categories.csv┘   (SQLite)        (GridSearchCV)      (joblib)      (classify + charts)
```

- **ETL** (`disaster_response.etl`): joins 26,248 messages with their
  category labels, expands the `categories` column (`related-1;request-0;…`)
  into 36 binary columns, clips out-of-range values, drops duplicates, and
  stores the result in SQLite.
- **Text** (`disaster_response.text`): URL placeholder replacement,
  stopword removal, verb lemmatization, plus a custom sklearn transformer
  for message length as an extra feature.
- **Model** (`disaster_response.model`): `CountVectorizer` + `TfidfTransformer`
  combined with the length feature, feeding a `MultiOutputClassifier` of
  random forests — one binary classifier per category — tuned with
  `GridSearchCV` on micro-averaged recall.
- **App** (`disaster_response.app`): Flask app (factory pattern) where you
  type a message and get its predicted categories, plus genre/category
  distribution charts.

## Project structure

```text
├── data/                           # raw CSVs (DisasterResponse.db is generated, not committed)
├── src/disaster_response/          # the package
│   ├── etl.py                      # load / clean / save + `disaster-etl` CLI
│   ├── text.py                     # tokenizer, NLTK setup, TextLengthExtractor
│   ├── model.py                    # pipeline, grids, evaluation, persistence
│   ├── train.py                    # `disaster-train` CLI
│   ├── app.py                      # Flask factory + `disaster-serve` CLI
│   └── templates/                  # Jinja templates
├── tests/                          # pytest suite (ETL, text, model, app)
├── .github/workflows/ci.yml        # tests on Python 3.12
├── pyproject.toml                  # packaging + console scripts
└── requirements.txt                # pinned dependencies
```

## Setup

```bash
git clone https://github.com/catherinecao123/end-to-end-nlp-disaster-response-classification-pipeline.git
cd end-to-end-nlp-disaster-response-classification-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m disaster_response.text   # download NLTK resources (punkt, wordnet, stopwords)
```

## Usage

```bash
# 1. ETL: build the SQLite database (~10 s)
disaster-etl data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

# 2. Train: quick smoke run on a 3,000-row subsample (~1 min)
disaster-train data/DisasterResponse.db models/classifier.pkl --limit 3000

#    Full run: small grid on all 26k rows (~10-15 min), or the full 2x2 grid
disaster-train data/DisasterResponse.db models/classifier.pkl
disaster-train data/DisasterResponse.db models/classifier.pkl --grid full --cv 3

# 3. Serve: open http://127.0.0.1:3001/
disaster-serve --database data/DisasterResponse.db --model models/classifier.pkl
```

The default grid is intentionally small (one configuration, 2-fold CV) so a
training run stays fast; `--grid full` restores the original 2×2 search.
`--limit N` subsamples the training data for smoke runs.

## Results

Full run — small grid, all 26,215 messages, 5,243-message test set
(mean per-label accuracy **0.95**):

| | precision | recall | f1-score |
|---|---:|---:|---:|
| micro avg | 0.82 | 0.53 | 0.64 |
| weighted avg | 0.76 | 0.53 | 0.57 |

Strong categories include `related` (F1 0.89), `earthquake` (0.83),
`weather_related` (0.77), `aid_related` (0.72), `food` (0.65), `request`
(0.62), `floods` (0.61) and `storm` (0.61). Rare categories with few
positive examples (e.g. `offer`, `security`, `child_alone` — which has no
positive examples at all in the dataset) score near zero: a class-imbalance
limitation of the dataset, not a bug. Precision is consistently much higher
than recall — the model rarely mislabels a message, but misses many true
positives on infrequent categories.

## Tests

```bash
pytest -q   # 20 tests: ETL cleaning, tokenizer, model smoke run, Flask app
```

CI runs the suite on Python 3.12 for every push and pull request.

## What changed in the rebuild (vs. the 2021 original)

- Restructured from loose scripts into an installable `src/` package with
  console entry points (`disaster-etl`, `disaster-train`, `disaster-serve`).
- Updated for current libraries: Flask 3.x, scikit-learn 1.x, pandas 3.x,
  SQLAlchemy 2.x, NLTK 3.10 (incl. the new `punkt_tab` resource), Plotly 7.
- Fixed a latent bug: the original grid searched `clf__min_samples_split`,
  which is not a valid parameter of `MultiOutputClassifier`; it is now
  `clf__estimator__min_samples_split`.
- NLTK downloads moved out of import time into an explicit setup step;
  stopwords and the lemmatizer are cached instead of rebuilt per token.
- ETL rewritten vectorized (no chained assignment); category columns are
  derived by name instead of fragile positional slicing (`df.iloc[:, 4:]`).
- Flask app converted to a factory with configurable database/model paths
  (no more hardcoded relative paths or module-level side effects).
- Added a pytest suite, GitHub Actions CI, pinned requirements, and a
  `.gitignore` for generated artifacts.
- Removed generated binaries from the repo: the 99 MB `classifier.pkl` and
  the SQLite database are rebuilt by the pipeline and no longer committed.
  Note the old `classifier.pkl` is not loadable under the new layout (the
  pickled tokenizer moved modules) — retrain with `disaster-train`.

## Dataset

Figure Eight (now Appen) disaster response messages, originally distributed
with the Udacity Data Scientist Nanodegree: real messages sent during
disaster events, labelled across 36 categories.
