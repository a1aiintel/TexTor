from TexTor.settings import SPACY_MODEL, COREF_MODEL
import spacy
import sense2vec

_nlp = None
_coref_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def get_corefnlp():
    global _coref_nlp
    if _coref_nlp is None:
        _coref_nlp = sense2vec.load(COREF_MODEL)
    return _coref_nlp
