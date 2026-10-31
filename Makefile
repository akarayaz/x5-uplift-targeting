PY ?= python

.PHONY: download convert audit features all test freeze

download:
	$(PY) scripts/01_download_data.py

convert:
	$(PY) scripts/02_convert_to_parquet.py

audit:
	$(PY) scripts/03_inspect_data.py

features:
    $(PY) scripts/04_build_features.py

criteo:
	$(PY) scripts/05_sample_criteo.py
	
# Data stage only for now. Notebook execution is added once the notebooks exist.
all: download convert audit features

test:
	$(PY) -m pytest -q

freeze:
	$(PY) -m pip freeze --exclude-editable > requirements.lock
