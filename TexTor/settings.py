from os.path import join, dirname

SPACY_MODEL = "en_core_web_md" # https://spacy.io/models/
# python -m spacy download en_core_web_sm

COREF_MODEL = "en_coref_md" # en_coref_sm, en_coref_md, en_coref_lg
# pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz

# https://github.com/allenai/allennlp-demo  <- host your own
# demo server - http://demo.allennlp.org/predict/
ALLENNLP_URL = "http://demo.allennlp.org/predict/"


MODELS_PATH = join(dirname(__file__), "models")
