echo "installing packages"
pip install -r requirements.txt
#pip3 -r requirements.txt
echo "unzip transformers directory and corpus"
unzip transformers.zip
unzip small_corpus.zip
mv transformers/run_generation.py .
echo "make necessary directories"
#mkdir data
mkdir classifiers
mkdir npi_models
echo "done"
