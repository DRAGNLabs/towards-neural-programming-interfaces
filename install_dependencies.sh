echo "installing packages"
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
echo "unzip transformers directory and corpus"
unzip transformers.zip -d src/npi
unzip data/raw/small_corpus.zip
echo "done"
