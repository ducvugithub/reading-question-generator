PYTHON := .venv/bin/python

.PHONY: venv install download script app clean

venv:
	/Users/anh-duc.vu/miniconda3/bin/python -m venv .venv --system-site-packages
	$(PYTHON) -m pip install "numpy<2" streamlit langdetect -q

install:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) -c "import stanza; stanza.download('en'); stanza.download('fi')"

script:
	$(PYTHON) scripts/question_generation_script.py $(INPUT) $(if $(OUTPUT),--output $(OUTPUT),) $(if $(TARGET_CEFR),--target-cefr $(TARGET_CEFR),)

app:
	$(PYTHON) -m streamlit run scripts/question_generation_streamlit.py --server.fileWatcherType none

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
