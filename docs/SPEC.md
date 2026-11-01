# Handoff spec: Air Cargo Payload Inference MVP (ADS-B → Freighter Weight Index)

**Audience:** coding agent (Python). **Owner:** Baxter Black. **Goal:** produce a defensible, reproducible analysis + Substack-ready charts in ~2–3 weeks of agent effort, with a codebase that can grow into a real data product.

> Original handoff spec, kept verbatim. Implementation decisions and deviations are tracked in `docs/DECISIONS.md`.

---

## 0. One-paragraph objective

Build a pipeline that ingests historical ADS-B tracks for all-cargo freighter departures from major Chinese/HK gateways (and Anchorage as the transpacific waypoint), extracts takeoff-performance features from each departure, converts those into a **relative takeoff-weight index per airframe type** using physics-informed models, and aggregates into a **weekly lane-level capacity-utilization index** for Asia → North America air cargo. The headline deliverable is an **event study** around the end of the US de minimis exemption (China/HK: **2 May 2025**; global: **29 Aug 2025** — *verify both dates*) showing whether freighters out of China got measurably lighter and/or redeployed, validated against official lagging statistics (BTS T-100, HKIA/PVG monthly cargo stats, IATA/WorldACD headlines).

**Non-goals for MVP:** real-time streaming, absolute tonnage precision, customer attribution of ACMI flights, belly cargo on passenger flights, commercial API hardening.

---

## 1. Success criteria (in priority order)

1. **Tier 1 — Flight-count/redeployment index (must ship).** Weekly counts of freighter departures by operator × aircraft type × origin × destination lane, Jan 2024 → present. Charts showing (a) transpacific freighter frequencies over time, (b) where capacity moved after the de minimis events (Asia–Europe, Asia–LatAm, intra-Asia). No weight inference needed. This alone is a publishable artifact.
2. **Tier 2 — Relative weight index (core thesis).** For each departure, a per-type normalized takeoff-weight proxy with uncertainty. Weekly median (and IQR) by lane. Statistically test pre/post change at each event date (difference-in-differences vs. a control lane, e.g., Asia→Europe or intra-US).
3. **Tier 3 — Validation.** Correlation / lead-lag of the index vs. monthly BTS T-100 freight tons on the same O&D pairs and HKIA monthly cargo throughput. Report: does the ADS-B index lead the official number by 4–8 weeks? What's the R²?
4. **Tier 4 (stretch) — Absolute payload estimate.** Convert relative index → tonnes using OpenAP type parameters + fuel-burn estimate from flight distance. Report with wide bands and explicit caveats.
5. **Writeup.** A `REPORT.md` with methodology, figures, caveats, and a "how I'd productize this" section. Figures exported as PNG + interactive HTML.

Everything must be reproducible from a single `make all` / `python -m cargo_index run --config configs/mvp.yaml`.

---

## 2. Data sources (verify availability/terms at implementation time; pick the first that works)

### 2.1 Historical ADS-B (the critical path)

| Source | Resolution | Cost | Notes |
|---|---|---|---|
| **adsb.lol globe_history** (GitHub releases, `adsblol/globe_history_YYYY`) | ~5 s global snapshots, full day tarballs (readsb JSON format) | Free, ODbL-ish | **First choice.** Covers 2023→. Coverage in mainland China is community-fed and patchy; HKG/TPE/ICN/ANC/US are good. Check which days exist. |
| **ADS-B Exchange historical (`readsb-hist`)** | 5 s global | Samples free (1st of month); full archive paid | Use samples to cross-check adsb.lol; buy a month if coverage at PVG/CAN is needed and budget allows. |
| **OpenSky Network Trino/Impala historical DB** | ~1 Hz state vectors | Free for non-commercial research (requires account + access request) | **Best resolution for ground-roll features.** Apply day 1; use the `traffic` library (`traffic.data.opensky.history(...)`). Non-commercial license — fine for the Substack artifact, note for productization. |
| OpenSky REST / adsb.fi / airplanes.live live feeds | 1–5 s | Free | Only useful for forward-collection from now on; set up a cron collector on day 1 so we accumulate 1 Hz data at target airports going forward. |

**Decision rule:** build the pipeline on adsb.lol daily tarballs first (guaranteed availability), abstract the loader so OpenSky 1 Hz data can be swapped in for the takeoff-segment module. Store everything as Parquet partitioned by `date/airport`.

### 2.2 Aircraft reference data
- **OpenSky aircraft database** (CSV, free): `icao24 → registration, typecode, operator, model`.
- Supplement operator/freighter flags with a hand-curated `data/ref/freighter_registrations.csv` (registration, type, variant F/P2F/BCF/BDSF, operator, lessor if known). Seed from Planespotters/rzjets/Wikipedia fleet pages for the operators in §3.1. Treat as a maintained artifact.
- **OurAirports** `airports.csv` + `runways.csv` (free): airport elevation, runway endpoints, headings, lengths, slope.
- **OpenAP** (`pip install openap`): aircraft type parameters (MTOW, OEW, wing area, engine thrust), kinematic performance models (WRAP), fuel-flow models. Use for priors and the Tier 4 absolute estimate.

### 2.3 Weather
- **Iowa Environmental Mesonet ASOS/METAR archive** (free, global, HTTP CSV): hourly/half-hourly wind, temp, altimeter, for each target airport. Needed for density altitude and headwind correction.
- Optional: ERA5 (via `cdsapi`) for winds aloft if climb-gradient features need it. Skip for MVP.

### 2.4 Validation / ground-truth-ish data
- **BTS T-100 International Segment** (free download; monthly; ~3–4 mo lag): freight tons by carrier, origin, dest, aircraft type, departures performed. This is the primary validation target for US-bound lanes. Also gives per-flight average payload = freight_tons / departures — a **direct calibration target for Tier 4**.
- **HKIA monthly cargo throughput** (AA HK website), **Shanghai Airport Authority** monthly PVG cargo stats (investor releases), **Guangzhou Baiyun** stats if findable.
- **US Census FT-920 / USA Trade Online** monthly imports from China by mode (air), value and weight.
- Headline numbers from IATA monthly press releases, WorldACD, Xeneta/CLIVE (public blog posts only) for narrative sanity checks.

---

## 3. Scope definitions

### 3.1 Operators & types in scope (freighter-only)
**Operators (ICAO codes):** CKS Kalitta, GTI Atlas, PAC Polar, CLX Cargolux, ICV Cargolux Italia, CSN China Southern (freighter regs only), CKK China Cargo Airlines, CAO Air China Cargo, CSS SF Airlines, HYT YTO Cargo, CYZ China Postal, CSH Shanghai Airlines Cargo? (verify), CPA Cathay (freighter regs only), KAL Korean (freighter regs), AAR Asiana (freighter regs), CAL China Airlines (freighter regs), EVA (freighter regs), NCA Nippon Cargo, UPS, FDX, WGN Western Global, NCR National, ABW AirBridgeCargo (likely inactive), QTR/UAE/ETD freighters (for control lanes), SQC Singapore Cargo, LCO Lan Cargo, SCX Sun Country cargo (Amazon), ABX/ATN ATI (Amazon). Treat the list as config.

**Types:** B748 (filter out 747-8I pax regs), B744 (freighter regs only), B77L (777F only; exclude 777-200LR pax), B763 (freighter regs only), A332 (A330-200F/P2F only), MD11, B77F if the typecode appears, A333 P2F. Config-driven.

### 3.2 Airports
**Treatment (China/HK gateways):** PVG, CAN, SZX, HKG, plus ZHY/ZHCC Zhengzhou, CTU/TFU if coverage exists.
**Waypoint:** ANC (most transpacific freighters tech-stop; ANC departure weight = payload + second-leg fuel, so it's a useful secondary measurement of the same cargo).
**US destinations:** ORD, LAX, CVG, JFK, DFW, MIA, ATL, SDF, MEM, LCK (Rickenbacker — Cainiao/Atlas e-comm hub), ONT.
**Control lanes:** Asia→Europe (PVG/HKG → LEJ, LUX, AMS, FRA, LGG), intra-US domestic (MEM/SDF departures), Europe→US (LUX/LEJ → ORD/JFK) — lanes not directly hit by de minimis.

### 3.3 Time window
Jan 2024 → latest available. Pre-period for event study: Jan 2024 – Apr 2025. Events: 2 May 2025, 29 Aug 2025. Also mark Lunar New Year windows and Q4 peak for seasonality.

---

## 4. Pipeline design

```
cargo_index/
  configs/mvp.yaml                # airports, operators, types, dates, paths
  data/
    raw/adsb/{source}/{date}/...  # tarballs / parquet
    raw/metar/{icao}/{yyyymm}.csv
    raw/t100/...
    ref/{opensky_aircraft.csv, freighter_registrations.csv, airports.csv, runways.csv}
    processed/
      flights/{date}.parquet      # one row per flight leg
      takeoffs/{date}.parquet     # one row per departure with features
      index/weekly_lane.parquet
  src/cargo_index/
    ingest/   adsblol.py, opensky.py, metar.py, t100.py, refdata.py
    identify/ aircraft.py (hex→reg→type→operator→freighter flag), flights.py (segment tracks into legs; assign origin/dest)
    takeoff/  segment.py (extract ground roll + initial climb), runway.py, features.py, weather.py
    model/    physics.py (V_LOF → relative mass), calibrate.py, absolute.py (OpenAP-based)
    aggregate/ index.py, event_study.py, validate.py
    viz/      charts.py
    cli.py
  notebooks/  01_coverage_audit.ipynb, 02_takeoff_qc.ipynb, 03_results.ipynb
  tests/
  REPORT.md
```

Use **polars or DuckDB** for the heavy lifting (daily global tarballs are large; filter to a bounding box around target airports *during* extraction and never materialize the global set). Use the **`traffic`** library's `Flight`/`Traffic` objects where convenient (it has runway alignment, takeoff/landing detection, and OpenSky integration built in — don't reinvent).

### 4.1 Ingest
- `adsblol.py`: for each date, download tarball, stream-parse readsb JSON traces, keep only hex codes present in the freighter whitelist **OR** any point within 20 nm of a target airport below 10,000 ft (we also want candidates we haven't whitelisted yet). Write Parquet with columns: `hex, ts, lat, lon, baro_alt, geom_alt, gs, track, vrate, on_ground, callsign, squawk, source_type`.
- `opensky.py`: same schema from Trino if access granted; airport bounding-box queries per day.
- `metar.py`: pull METARs for all target airports, parse to `ts, wind_dir, wind_kt, temp_c, dewpoint_c, altimeter_inhg, qnh_hpa`.
- Log a **coverage audit** per airport per day: number of freighter tracks seen, median sample interval, fraction of departures with ≥8 points below 1,500 ft AGL. This feeds `01_coverage_audit.ipynb` — we need to know early whether PVG/CAN are usable or whether we lean on HKG + ANC.

### 4.2 Identify
- Join hex → OpenSky DB → registration/type/operator. Override with curated freighter CSV. Flag `is_freighter`, `variant`, `operator_icao`, `lessor` (if known).
- Segment each hex's day into flight legs (gap > 15 min on ground or no data → new leg). Assign origin/destination airport by nearest airport to first/last point with `on_ground` or alt < 300 ft AGL; tag `tech_stop` if leg ends at ANC and the same hex departs ANC within 6 h. Compute `great_circle_nm`, `block_time_min`.
- Emit `flights/{date}.parquet`. **This completes Tier 1** after aggregation.

### 4.3 Takeoff segment extraction
For each departure from a target airport:
1. Determine **runway**: from the last on-ground points' heading and position vs. `runways.csv` (use `traffic`'s `aligned_on_runway` if available).
2. Define **roll start**: first point on the runway with `gs ≥ 30 kt` and track aligned ± 10° with runway heading.
3. Define **liftoff**: first point where `on_ground == False` (if source provides it) else first point with `baro_alt > field_elev + 50 ft` or `vrate > 300 fpm`, with position still within runway extended centerline.
4. Define **end of initial climb**: first point with `baro_alt ≥ field_elev + 1,500 ft`.
5. Reject if: < 4 points on the roll, > 15 s gap near liftoff, runway unidentified, track deviates from runway heading before liftoff (intersection/early turn ambiguity), or liftoff point implausible (>14,000 ft from threshold).

Record `n_points_roll`, `sample_interval_s`, and a `quality` score; downstream models should weight by quality.

### 4.4 Feature engineering (per departure)
- `gs_lof` — ground speed at liftoff (interpolate between last on-ground and first airborne point, don't just take the first airborne sample).
- `tas_lof = gs_lof + headwind_component` (from METAR nearest in time; headwind = wind_kt · cos(wind_dir − runway_heading)).
- `rho` — air density from METAR temp and pressure altitude (field elevation + QNH correction). Use ISA formulae; `metpy` optional.
- `roll_time_s`, `roll_dist_m` (haversine from roll start to liftoff), `mean_accel = tas_lof / roll_time_s`.
- `climb_grad_pct` — (alt gain)/(ground distance) from liftoff to 1,500 ft AGL; `mean_vrate_fpm` over same window; `tas_climb` mean TAS over same window.
- `runway_length_m`, `runway_slope`, `remaining_runway_at_lof`.
- Context: `type, variant, operator, origin, dest, great_circle_nm, hour_local, dow, week`.
- Flag `likely_derate` (heuristic: long runway + low accel + low climb gradient but also low liftoff speed → inconsistent; leave for model residual analysis).

### 4.5 Physics-informed relative weight model

Core relations (document in `physics.py` docstring):
- Liftoff: `W = ½ ρ V_LOF² S C_L,LOF`  ⇒  for a fixed type (fixed S, and assuming a consistent flap/C_L convention), **`W ∝ ρ · V_LOF²`**.
- Ground roll: `s ∝ W² / (ρ S C_L T_net)` ⇒ `W ∝ sqrt(s · ρ · T_net)`, but T_net is contaminated by derate — treat as secondary.
- Initial climb: `γ ≈ (T − D)/W` ⇒ lower gradient at higher W **and** at higher derate (confounded, same direction). Secondary.

**MVP estimator (per type):**
```
m_rel = ρ · tas_lof²                      # primary
```
then within each type, express as a percentile / z-score vs. the type's long-run distribution (`m_idx = (m_rel − median_type) / IQR_type`). Also fit a robust regression per type:
```
log(m_rel) ~ 1 + climb_grad + mean_accel + runway_length + origin + hour + (quality weights)
```
and inspect residuals — not for prediction, but to understand how much the secondary features and airport fixed effects explain. **Primary output stays the ρ·V² index**, because it's the least confounded by derate.

Report per departure: `m_idx`, `m_rel`, `quality`, plus a bootstrap CI computed from liftoff-point interpolation uncertainty (perturb liftoff time ± half the sample interval and recompute).

**Known confounders (write them down in the report):** flap setting (freighters are fairly consistent per operator/airport, but not guaranteed), assumed-temperature derate (heavier → less derate → biases *against* seeing weight changes, i.e., conservative), intersection departures, wind gusts between METARs, barometric lag, receiver geometry. Empirically check: does `m_idx` for the same airframe on positioning/ferry flights (identifiable: departure with no cargo stop, e.g., ANC → maintenance base, or Cainiao charter repositioning) sit near the bottom of the distribution? That's the sanity test.

### 4.6 Aggregation & event study
- Weekly, by `lane = (origin_region, dest_region)` and by `origin_airport`: `n_departures`, `median m_idx`, `p25/p75`, `n_airframes`, `sum(MTOW-weighted proxy)` as a capacity measure.
- **Event study:** difference-in-differences of weekly median `m_idx` on China/HK → US lanes vs. control lanes, windows ± 12 weeks around 2 May 2025 and 29 Aug 2025. Also do the same on *flight counts* and *capacity (MTOW-sum)*. Include seasonal controls (week-of-year fixed effects from 2024). Report estimates with bootstrap CIs (resample airframes, not departures — airframe effects are correlated).
- **Redeployment:** for each airframe active on China→US lanes in Q1 2025, track its lane mix by month through present. Sankey/stacked area of where capacity went. Separately for Atlas/Kalitta ACMI fleets vs. Chinese carriers.

### 4.7 Validation
- Join weekly index (resampled to monthly) to **T-100** freight tons for matching carrier × origin × dest where carrier files with BTS (Atlas, Kalitta, Polar, UPS, FedEx, China Southern, Cathay, etc. all do). Compute: Pearson/Spearman on levels and on monthly changes; cross-correlation at lags −2…+3 months to show lead; `payload_per_departure_t100 = freight_tons / departures` vs. `median m_idx` scatter per type — **this is the calibration curve for Tier 4.**
- Compare total China/HK → US freighter departures from our data vs. T-100 departures performed → a direct coverage-rate check (should be ≥ 80% for US-bound lanes since US coverage is good).
- HKIA monthly throughput vs. HKG weekly index.

### 4.8 Tier 4 — absolute payload (only after Tier 3)
Per type, from OpenAP: `OEW, MTOW, S`. Assume `C_L,LOF` per type as a free parameter; fit it so that the T-100-derived per-departure payload matches `½ ρ V² S C_L − OEW − fuel_est` on average. `fuel_est` = OpenAP fuel-flow model × route distance (+ reserves, + second leg if tech stop). Report payload with ± bands from the calibration residual. Present as "indicative tonnes" — the relative index remains the headline.

---

## 5. Charts for the writeup (`viz/charts.py`, all exported PNG + HTML)
1. Coverage map: freighter track density around each target airport; table of usable departures per airport per month.
2. Weekly freighter departures by lane, 2024→present, with event-date vertical lines.
3. Capacity redeployment: stacked area of MTOW-weighted departures by destination region for China/HK-origin freighters; Sankey of airframe lane migration.
4. Weight index: weekly median `m_idx` by lane with IQR bands, treatment vs. control, event lines.
5. DiD coefficient plot (counts, capacity, weight index) for each event.
6. Validation: index vs. T-100 tons (dual axis + scatter), lead-lag cross-correlation bar chart.
7. Physics sanity: `m_rel` distribution per type with known ferry flights highlighted; per-airframe time series for 2–3 famous tails.
8. (Tier 4) Indicative tonnes per departure by lane over time.

---

## 6. Implementation order & checkpoints

| Day | Milestone | Check |
|---|---|---|
| 1 | Repo scaffold, config, OpenSky Trino access request submitted, adsb.lol loader works for one day, METAR loader works, ref data downloaded | `pytest` passes on loaders |
| 2–3 | Full-window ingest for target airports (parallelize by date; cache); coverage audit notebook | **Decision point:** are PVG/CAN usable? If < 50% of T-100 departures seen, pivot primary measurement to HKG + ANC departures and note it |
| 4–5 | Identify module; Tier 1 flight-count and redeployment charts | Counts vs. T-100 departures within ±20% for US-bound lanes |
| 6–8 | Takeoff segmentation + QC notebook; feature table | Visual QC of 50 random takeoffs; rejection rate reported |
| 9–10 | Physics index; ferry-flight sanity check; per-type distributions | Ferry flights sit in bottom quartile |
| 11–12 | Aggregation, event study, DiD | CIs reported; results robust to quality-threshold choices |
| 13–14 | Validation vs. T-100 / HKIA; Tier 4 if validation is decent | Lead-lag chart |
| 15 | `REPORT.md`, figures, cleanup, README with reproduction steps | `make all` runs end-to-end from raw |

Ship Tier 1 charts to Baxter at day 5 regardless of Tier 2 progress.

---

## 7. Engineering requirements
- Python 3.11+, `uv` or `poetry`; pinned deps: `polars`, `duckdb`, `pyarrow`, `traffic`, `openap`, `numpy`, `scipy`, `statsmodels`, `plotly`, `matplotlib`, `requests`, `pyyaml`, `pytest`.
- Every raw download cached and checksummed; never re-download. Raw data dir size estimate logged.
- Config-driven: no airport/operator/date literals in code.
- Logging of rejection reasons at every filter stage, written to `processed/qc/`.
- Unit tests for: density calc, headwind projection, runway assignment, liftoff interpolation, m_rel formula, DiD estimator (synthetic data).
- `REPORT.md` must include a **Limitations** section written for a skeptical quant reader (sampling, confounders, license constraints: OpenSky data is non-commercial; adsb.lol/ADS-B Exchange terms — cite).
- Do **not** publish per-tail data for anything that isn't an all-cargo commercial operator (not an issue here, but enforce via the freighter whitelist).

---

## 8. Open questions for the agent to resolve early and report back
1. Actual ADS-B coverage/quality at PVG, CAN, SZX in adsb.lol data (sample interval, fraction of departures with a usable roll). This determines the design.
2. Whether OpenSky grants Trino access within the project window; if yes, migrate the takeoff module to 1 Hz data.
3. Does readsb data expose `on_ground` reliably for these aircraft (depends on transponder/version)? If not, altitude-based liftoff detection is the fallback — measure how much CI widens.
4. T-100 latest available month and which target carriers report.
5. Any prior literature to cite/borrow from — start with Junzi Sun et al. (TU Delft) on ADS-B-based takeoff mass estimation and the OpenAP/WRAP papers; search for anything newer on freighter payload inference.

---

## 9. Definition of done
A reviewer with no context can clone the repo, run one command, and reproduce every figure in `REPORT.md`; the report answers, with numbers and uncertainty: *"Did freighters leaving China for the US get lighter and/or fewer after May/Aug 2025, where did the capacity go, and did the ADS-B signal see it before the official statistics did?"*
