echo "installing packages"
pip -r requirements.txt
#pip3 -r requirements.txt
echo "unzip transformers directory"
unzip transformers.zip
mv transformers/run_generation.py .
mkdir data
mkdir classifiers
mkdir npi_models
