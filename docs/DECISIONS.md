# Decisions, deviations from spec, and findings

Dated log. Newest findings at the bottom of each section.

## Findings (open questions from SPEC §8)

### F1. adsb.lol coverage at mainland Chinese gateways is effectively nil; HKG has improved over time (2026-10-30)
Probe days 2025-03-12 and 2026-10-28, freighter departures per airport (`qc/coverage_airport_day.csv`):

| Airport | Departures (any evidence) | Ground roll observed | Usable takeoffs |
|---|---|---|---|
| PVG / CAN / CGO / CTU | 24–28 / 7–15 / 1–2 / 2 | **0** | 0 |
| SZX | 16–30 (seen climbing ~5,000 ft) | 0 | 0 |
| HKG, 2025-03-12 | 97 | **0** | 0 |
| HKG, 2026-10-28 | 107 | **94** | 36 |
| ANC | ~130 | ~all | 112–200 |
| JFK / ATL / LAX (2026) | 30 / 15 / 55 | ~all | 27 / 14 / 45 |

Mainland departures are only known via the callsign route DB (F4) or because they're seen
climbing out far from the field. A one-snapshot check of the ADS-B Exchange free sample
(2025-03-01) shows ADSBx sees ~10–14 aircraft on the ground at HKG and some at SZX, but
PVG/CAN are still sparse there too.

**Consequence (spec's day-2 decision point):** the weight index's primary measurement is
**ANC departures** (CN/HK → ANC → US through-flights, same cargo + 2nd-leg fuel), plus HKG for
whatever period adsb.lol covers it, plus US/EU departures for control lanes. Tier 1 counts for
mainland origins depend on route-DB inference. A coverage-sampling ingest (1st & 15th of every
month, 2024-01 → now) is running to map exactly when each airport becomes usable.

### F2. readsb `on_ground` cannot be used to detect liftoff
The transponder switches from surface to airborne messages at ~90 kt (median `gs_switch` 89–96 kt
for every type) on a speed rule, while the real liftoff is ~150–170 kt. Taking that switch as
liftoff gave V_LOF ≈ 90 kt and 450 m rolls. Liftoff is now found by fitting a hinge to vertical
rate after the switch (`takeoff/segment.py`). That gives physically sensible values (median TAS at
liftoff: 777F 168 kt, 747-8F 165, 747-400F 156, 767F 157, MD-11F 170; observed roll from 30 kt
1.4–2.0 km). Barometric rate lag biases t_lof slightly late, by about the same amount within a
type, so it mostly cancels in the within-type index. The time to the switch is still useful as
the start of the roll.

### F3. Sample interval at well-covered airports is ~1–2 s, not 5 s
At ANC/JFK/HKG(2026) the median roll interval is 1.0–2.0 s. Per-flight ρV² band from liftoff
timing alone is ~±2.5%. Within-type IQR of ρV² is ~±10%, so the signal-to-noise per flight is usable.

### F4. Callsign route database fills unseen origins
About half of ANC-bound freighter legs are first picked up at cruise west of Alaska, so their
Asian origin is never seen. vradarserver/standing-data routes (e.g. `GTI8904 ZSPD-PANC-KORD`)
fill these, **only when the route contains the endpoint we did observe** (the snapshot has stale
entries, e.g. `CAO1005 ZSPD-EBLG-LEMD` seen flying to ANC). ~230–245 legs/day are filled
this way (`origin_how = route_db`). Caveat: the DB is a current snapshot. TODO: pin historical
versions from the repo's git history for 2024–25 legs.

## Decisions / deviations

* **D1 Stream, don't store, tarballs.** The full archive is ~3 TB (~3–4 GB/day). `ingest` streams
  GitHub release parts straight into `tarfile` and records per-part sha256 + byte counts in each
  day's `manifest.json`. "Never re-download" holds for the filtered output: a day with a manifest
  is skipped. Re-filtering with a wider capture means re-streaming.
* **D2 Capture by type, not all aircraft near airports.** Spec §4.1 keeps any aircraft near a
  target airport below 10k ft. We keep only target types (B744/B748/B77L/B77F/B763/A332/A333/MD11)
  near target airports, plus the whitelist. All points < 10k ft are kept; above that, 1 per 60 s.
  ~1.2–1.4k aircraft and 20–30 MB per day. Freighter status is decided later, so we can re-identify
  without re-downloading.
* **D3 Liftoff = vertical-rate hinge** (see F2). `on_ground` only bounds the roll start.
* **D4 Freighter identification is layered** (`identify/aircraft.py`): curated CSV > model string
  regex > freighter-only type (MD11) or B77L-unless-passenger-operator > all-cargo operator by
  dominant callsign prefix. adsb.lol's `desc` doesn't separate 777F from 777-200LR, and the
  OpenSky DB is incomplete (e.g. ~13 Atlas 747s). Mixed-operator 747/777 airframes go to
  `data/ref/review/freighter_review.csv` for a human.
* **D5 Airports for endpoint assignment = scheduled-service large/medium airports** (plus targets).
  Otherwise military fields next to hubs (VHSK by HKG) capture approaches.
* **D6 Lanes use through-origin/destination** across tech stops at ANC (≤ 6 h turn), so
  PVG→ANC→ORD counts once as CN_HK→US in Tier 1. Tier 2 measures each departure: an ANC
  departure's lane is its *through-origin* region → destination region.
* **D7 Local hour is solar (lon/15)**, not civil time zone. Good enough for a time-of-day control.
* **D8 Event dates verified**: EO 14256 ended de minimis for China/HK effective 2 May 2025;
  EO 14324 suspended it globally effective 29 Aug 2025.

## Next steps
1. Finish the coverage-sampling ingest and plot usable-departures-per-airport by month (Chart 1).
2. Fix `few_roll_points` rejects at KCVG/KSDF/KMEM (ground samples sparse → maybe use
   airborne-mode samples as the roll).
3. Ferry/positioning sanity test (spec §4.5).
4. T-100 ingest (`ingest/t100.py`); coverage-rate check vs BTS departures.
5. Wire the event study into the CLI; full-window ingest once the human decisions below are made.
