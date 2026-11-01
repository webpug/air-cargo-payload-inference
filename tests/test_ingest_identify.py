
import polars as pl

from cargo_index.config import Config
from cargo_index.identify import aircraft, flights
from cargo_index.ingest import adsblol, routes

AIRPORTS = pl.DataFrame({"icao": ["PANC", "KORD"], "lat": [61.174, 41.979], "lon": [-149.998, -87.904],
                         "elev_ft": [152.0, 672.0], "region": ["ANC", "US"]})


def _trace(icao="a1b2c3", t="B744", lat0=61.174, lon0=-149.998):
    rows = []
    for k in range(30):
        alt = "ground" if k < 10 else 200 + 300 * (k - 10)
        det = {"flight": "CKS123  "} if k == 0 else None
        rows.append([k * 5.0, lat0 + 0.002 * k, lon0, alt, 20 + 6 * k, 345.0, 0, 0 if k < 10 else 1500, det,
                     "adsb_icao", None, None, None, None])
    return {"icao": icao, "r": "N700CK", "t": t, "timestamp": 1_741_737_600.0, "trace": rows}


def _filter(whitelist=()):
    return adsblol.TraceFilter(AIRPORTS, {"B744"}, set(whitelist), 20, 10000)


def test_parse_trace_keeps_candidate_near_airport():
    out = adsblol.parse_trace(_trace(), _filter())
    assert out is not None
    assert out["on_ground"][0] is True and out["baro_alt"][0] is None
    assert out["callsign"][-1] == "CKS123"


def test_parse_trace_drops_far_or_wrong_type():
    assert adsblol.parse_trace(_trace(t="A320"), _filter()) is None
    assert adsblol.parse_trace(_trace(lat0=10.0, lon0=10.0), _filter()) is None
    # ... unless whitelisted
    assert adsblol.parse_trace(_trace(lat0=10.0, lon0=10.0), _filter({"a1b2c3"})) is not None


def test_chained_stream_hashes_parts():
    import hashlib
    s = adsblol._ChainedStream([("a", iter([b"abc", b"def"])), ("b", iter([b"xyz"]))])
    assert s.read() == b"abcdefxyz"
    s.close()
    assert s.digests["a"]["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    assert s.digests["b"]["bytes"] == 3


def test_model_freighter_regex():
    for m in ["747-4R7F", "A330-243F", "747-400BCF", "767-3S2F(ER)", "777-FFT", "747 48EFSCD", "MD-11F", "747-87UF"]:
        assert aircraft.model_says_freighter(m), m
    for m in ["747-8B5", "A330 343E", "777-200LR", "767-31A", None]:
        assert not aircraft.model_says_freighter(m), m


def test_region_of_longest_prefix(tmp_path):
    cfg = Config.load("configs/mvp.yaml")
    assert cfg.region_of("ZSPD") == "CN_HK"
    assert cfg.region_of("VHHH") == "CN_HK"
    assert cfg.region_of("VTBS") == "ASIA_OTHER"
    assert cfg.region_of("PANC") == "ANC"
    assert cfg.region_of("PAFA") == "US"
    assert cfg.region_of("KORD") == "US"


def _pts(segments):
    """segments: list of (lat, lon, alt|'g', n, dt) blocks concatenated in time."""
    rows, t = [], 0.0
    for lat, lon, alt, n, dt in segments:
        for k in range(n):
            g = alt == "g"
            rows.append({"ts": t, "lat": lat, "lon": lon, "baro_alt": None if g else alt, "on_ground": g,
                         "flags": 0, "callsign": "CKS123"})
            t += dt
    return pl.DataFrame(rows, schema_overrides={"baro_alt": pl.Int32})


def test_segment_hex_two_legs_with_tech_stop():
    ap = flights.AirportIndex(AIRPORTS)
    cfg_id = {"leg_gap_min": 15, "ground_alt_agl_ft": 300, "endpoint_max_dist_nm": 10}
    pts = _pts([
        (61.174, -149.998, "g", 5, 10), (61.5, -149.0, 20000, 30, 60), (41.979, -87.904, "g", 5, 10),  # ANC -> ORD
        (41.979, -87.904, "g", 5, 600), (43.0, -88.0, 30000, 30, 60), (41.979, -87.904, "g", 3, 10),   # ORD -> ORD (loop)
    ])
    legs = flights.segment_hex(pts, ap, cfg_id)
    assert [(l["origin"], l["dest"]) for l in legs] == [("PANC", "KORD"), ("KORD", "KORD")]


def test_link_tech_stops_through_destination():
    legs = pl.DataFrame({
        "hex": ["h", "h"], "dep_ts": [0.0, 10 * 3600.0], "arr_ts": [8 * 3600.0, 16 * 3600.0],
        "origin": ["ZSPD", "PANC"], "dest": ["PANC", "KORD"],
        "origin_region": ["CN_HK", "ANC"], "dest_region": ["ANC", "US"],
    })
    out = flights.link_tech_stops(legs, ["PANC"], 6)
    assert out["tech_stop"].to_list() == [True, False]
    assert out["through_dest_region"][0] == "US"
    assert out["through_origin_region"][1] == "CN_HK"


def test_route_inference_requires_observed_endpoint():
    r = pl.DataFrame({"callsign": ["GTI8904", "CAO1005"], "route": [["ZSPD", "PANC", "KORD"], ["ZSPD", "EBLG", "LEMD"]]})
    legs = pl.DataFrame({"callsign": ["GTI8904", "CAO1005"], "origin": [None, None], "dest": ["PANC", "PANC"],
                         "origin_how": [None, None], "dest_how": ["ground", "ground"]},
                        schema_overrides={"origin": pl.Utf8, "origin_how": pl.Utf8})
    out = routes.infer_endpoints(legs, r)
    assert out["origin"].to_list() == ["ZSPD", None]      # CAO1005's route doesn't include PANC
    assert out["origin_how"].to_list() == ["route_db", None]
