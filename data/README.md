# Data

The two CSVs are the Figure Eight disaster-response dataset (26,248 real
messages sent during disaster events, each labelled across 36 categories):

- `disaster_messages.csv` — `id, message, original, genre`
- `disaster_categories.csv` — `id, categories` (`name-0/1` pairs joined by `;`)

`DisasterResponse.db` is **generated** by the ETL step and intentionally not
committed — run this from the repository root to rebuild it:

```bash
disaster-etl data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
