# Air Cargo Payload Inference (ADS-B → Freighter Weight Index)

Infers how many freighters fly Asia → North America, and how heavy they are at
takeoff, from public ADS-B data. Built to test whether China/HK → US freighters
got lighter and/or fewer after the US de minimis changes (2 May 2025; 29 Aug 2025).

* Spec: [`docs/SPEC.md`](docs/SPEC.md)
* Decisions, deviations and early findings: [`docs/DECISIONS.md`](docs/DECISIONS.md)
* Remaining work and open decisions: [GitHub issues](../../issues)

**Status:** MVP pipeline runs end to end. 67 sample days (1st and 15th of each month,
Jan 2024 → Oct 2026) have been ingested on the original dev machine to map coverage; that
data is *not* in the repo (see [Data](#data)). The full-window ingest, validation (Tier 3),
event study wiring and the report are still open — see the issues.

## Setup on a new machine

Requirements:

* Linux or macOS, Python 3.12, [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
* Disk: ~25 MB per ingested day of ADS-B output (~25 GB for the full window), plus ~2 GB of
  reference data. Raw tarballs are streamed, not stored.
* Bandwidth/CPU: each day is a 1.4–4.1 GB download; one worker processes ~6 MB/s (CPU-bound
  JSON parsing), so ~7 min/day/worker. Use `--workers` ≈ number of cores.

```bash
git clone https://github.com/webpug/air-cargo-payload-inference.git
cd air-cargo-payload-inference
uv sync
cp .env.example .env      # then fill in the keys below
set -a; source .env; set +a
make test                 # 29 unit tests, no network needed
```

### Keys and accounts you need to provision

None of these are in the repo. Never commit `.env`.

| Variable / account | Needed for | Required? | How to get it |
|---|---|---|---|
| `GITHUB_TOKEN` | `ingest`: one GitHub API call per day to list the adsb.lol release assets. Without a token GitHub allows 60 calls/hour, so more than ~60 days per hour fails. | **Yes** for more than a handful of days | github.com → Settings → Developer settings → Fine-grained tokens → generate, *Public repositories (read-only)*, no extra permissions. |
| AWS account (`AWS_PROFILE`, `AWS_REGION`, `CARGO_S3_BUCKET`) | Running the full ingest on EC2 and storing output in S3. | Optional — code support is still an open issue | Your own AWS account; use an instance role or `aws configure --profile`, not keys in files. |
| OpenSky account + approved research access (`OPENSKY_USERNAME`) | Optional higher-coverage backup source via Trino. | Optional — not used by the code yet | Register at opensky-network.org, then *My OpenSky → Request Data Access*. Non-commercial research terms. |

Everything else (OurAirports, OpenSky aircraft DB snapshot, IEM METAR, vradarserver routes,
adsb.lol archives) is downloaded anonymously.

## Running

```bash
# 1. Reference data (airports, runways, aircraft DB, callsign routes): ~2 GB, once.
uv run python -m cargo_index refdata

# 2. Smoke test on a single day (~5–10 min, ~25 MB out)
uv run python -m cargo_index ingest --dates 2026-10-28
uv run python -m cargo_index metar
uv run python -m cargo_index identify
uv run python -m cargo_index takeoffs
uv run python -m cargo_index audit
uv run python -m cargo_index index
uv run python -m cargo_index charts

# 3. Whole window from configs/mvp.yaml (window.start → today; ~1,000 days, ~2.9 TB streamed)
make all
# or a sample: uv run python -m cargo_index ingest --workers 8 --dates 2025-04-01 2025-04-15 ...
```

All steps are resumable: `ingest` skips a day that already has a `manifest.json`, and downloads
are cached with a `.sha256.json` sidecar. Pass `--force` to redo. Every parameter lives in
`configs/mvp.yaml`; use `--config` for another config.

## Pipeline

| Step | Module | Output |
|---|---|---|
| refdata | `ingest/refdata.py`, `ingest/routes.py` | `data/ref/{airports,runways,opensky_aircraft,routes}.parquet` |
| metar | `ingest/metar.py` | `data/processed/metar.parquet` |
| ingest | `ingest/adsblol.py` | `data/raw/adsb/adsblol/date=*/{points,aircraft}.parquet`, `manifest.json` |
| identify | `identify/aircraft.py`, `identify/flights.py` | `data/ref/aircraft_registry.parquet`, `data/processed/flights_all.parquet`, `data/ref/review/freighter_review.csv` |
| takeoffs | `takeoff/segment.py`, `takeoff/features.py`, `model/physics.py` | `data/processed/takeoffs_all.parquet`, `data/processed/qc/takeoff_rejects/` |
| audit | `aggregate/coverage.py` | `data/processed/qc/coverage_airport_day.csv` |
| index | `aggregate/index.py` | `data/processed/index/weekly_lane_{counts,weight}.parquet` |
| charts | `viz/charts.py` | `figures/*.png`, `figures/*.html` |

`aggregate/event_study.py` has the DiD estimator (airframe-clustered bootstrap); it is not yet
wired into the CLI.

## Data

`data/raw/`, `data/processed/`, generated reference parquet and `data/ref/review/` are
git-ignored. Rebuild them with the commands above. The repo only tracks code, config, docs,
tests and the PNG charts.

**Publishing rule (from the spec):** do not publish per-tail data for anything that isn't an
all-cargo commercial operator. That is why the generated review list of mixed-operator
airframes is not committed.

## Human-maintained inputs

* `data/ref/freighter_registrations.csv`: curated freighter overrides (they beat every heuristic).
  Columns: `registration,is_freighter,variant,operator,lessor,source,notes`.
* `data/ref/review/freighter_review.csv`: generated by `identify`. It lists mixed-operator
  747/777 airframes the heuristics can't classify. Copy decisions into the curated CSV and re-run
  `identify`.

## Data licences

adsb.lol globe_history: ODbL 1.0 (database) / CC0 (feeder contributions). OurAirports: public
domain. OpenSky aircraft DB / historical DB: non-commercial research terms. IEM ASOS: free, cite IEM.
vradarserver standing-data routes: see that repo's licence.
