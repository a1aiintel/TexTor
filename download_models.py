import nltk
import sense2vec
import spacy
from TexTor.settings import SPACY_MODEL, COREF_MODEL

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download("stopwords")
nltk.download('brown')
nltk.download('punkt')
# TODO spacy + neuralcoref models