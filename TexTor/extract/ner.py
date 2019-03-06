import nltk

from TexTor.utils import normalize
from TexTor import get_nlp


def polyglot_NER(blob):
    """

    Args:
        blob:

    Returns:

    """
    from polyglot.text import Text

    text = Text(blob)
    ENTS = []
    for sent in text.sentences:
        for entity in sent.entities:
            ENTS.append((entity[0], entity.tag))

    return ENTS


def fox_NER(text):
    """
    source https://github.com/dice-group/fox
    demo http://fox-demo.aksw.org/#!/home

    Args:
        text:

    Returns:

    """

    from foxpy.utils import extractNifPhrases
    f = Fox()
    json_ld = f.recognizeText(text)
    nif_phrases = extractNifPhrases(json_ld)
    if len(nif_phrases) < 1:
        return text

    nif_phrases = sorted(nif_phrases, key=lambda t: t["endIndex"],
                         reverse=True)
    ents = []
    for n in nif_phrases:
        name = n["anchorOf"]
        tag = n["taClassRef"][1].split(":")[1].lower()
        ents.append((name, tag))
    return ents


def spacy_NER(text, nlp=None):
    ents = []
    nlp = nlp or get_nlp()
    text = normalize(text, nlp=nlp)
    doc = nlp(text)
    for entity in doc.ents:
        ents.append((entity.text, entity.label_))

    return ents


def nltk_NER(paragraph):
    words = nltk.word_tokenize(paragraph)
    tagged_Words = nltk.pos_tag(words)

    named_Entity = nltk.ne_chunk(tagged_Words)
    entities = []
    for x in named_Entity:
        if isinstance(x, nltk.tree.Tree):
            entities += [{"label": x.__dict__["_label"],
                          "pos_tag": [e[1] for e in x],
                          "tokens": [e[0] for e in x]}]
    return entities


def NER(paragraph, engine="nltk"):
    engine = engine.lower().strip()
    if engine == "spacy":
        return spacy_NER(paragraph)
    elif engine == "fox":
        return fox_NER(paragraph)
    elif engine == "polyglot":
        return polyglot_NER(paragraph)
    return nltk_NER(paragraph)


if __name__ == "__main__":
    from pprint import pprint

    pprint(NER("The Taj Mahal was built by Emperor Shah Jahan"))
    """
    [{'label': 'ORGANIZATION',
      'pos_tag': ['NNP', 'NNP'],
      'tokens': ['Taj', 'Mahal']},
     {'label': 'PERSON',
      'pos_tag': ['NNP', 'NNP', 'NNP'],
      'tokens': ['Emperor', 'Shah', 'Jahan']}]  
    """

    assert NER("The Taj Mahal was built by Emperor Shah Jahan",
               engine="fox") == [{'label': 'ORGANIZATION',
                                    'pos_tag': ['NNP', 'NNP'],
                                    'tokens': ['Taj', 'Mahal']},
                                   {'label': 'PERSON',
                                    'pos_tag': ['NNP', 'NNP', 'NNP'],
                                    'tokens': ['Emperor', 'Shah', 'Jahan']}]
