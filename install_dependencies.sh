echo "installing packages"
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
echo "unzip transformers directory and corpus"
unzip -f transformers.zip
unzip -f data/raw/small_corpus.zip
mv transformers/run_generation.py .
echo "make necessary directories"
mkdir models/classifiers
mkdir models/npi_models
echo "done"
