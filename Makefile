CONFIG ?= configs/mvp.yaml
RUN = uv run python -m cargo_index

.PHONY: all refdata metar ingest identify takeoffs audit index charts test lint

all: refdata metar ingest identify takeoffs audit index charts

refdata:  ; $(RUN) refdata --config $(CONFIG)
metar:    ; $(RUN) metar --config $(CONFIG)
ingest:   ; $(RUN) ingest --config $(CONFIG)
identify: ; $(RUN) identify --config $(CONFIG)
takeoffs: ; $(RUN) takeoffs --config $(CONFIG)
audit:    ; $(RUN) audit --config $(CONFIG)
index:    ; $(RUN) index --config $(CONFIG)
charts:   ; $(RUN) charts --config $(CONFIG)

test: ; uv run pytest -q
lint: ; uv run ruff check src tests
