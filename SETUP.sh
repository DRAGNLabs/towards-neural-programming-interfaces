echo setting up: installations
pip -r install requirements.txt
python3 -m spacy download en_core_web_sm
echo setting up: unzipping
unzip small_corpus.zip
unzip transformers.zip
echo setting up: moving files
mv transformers/run_generation.py .
