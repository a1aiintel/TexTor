from TexTor import get_nlp
from TexTor.understand.coreference import replace_coreferences
from TexTor.understand.inflect import singularize as make_singular
from spacy.parts_of_speech import NOUN


def singularize(text, nlp=None):
    nlp = nlp or get_nlp()
    doc = nlp(text)
    ignores = ["this", "data", "my", "was"]
    replaces = {"are": "is"}
    words = []
    for tok in doc:
        if tok.pos == NOUN and str(tok) not in ignores:
            words.append(make_singular(str(tok)))
        elif str(tok) in replaces:
            words.append(replaces[str(tok)])
        else:
            words.append(str(tok))
    return " ".join(words)


def normalize(text, remove_articles=False, solve_corefs=False,
              make_singular=False, coref_nlp=None, nlp=None):
    text = str(text)
    words = text.split()  # this also removed extra spaces
    normalized = ""

    for word in words:
        if remove_articles and word in ["the", "a", "an"]:
            continue

        # Expand common contractions, e.g. "isn't" -> "is not"
        contraction = ["ain't", "aren't", "can't", "could've", "couldn't",
                       "didn't", "doesn't", "don't", "gonna", "gotta",
                       "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's",
                       "how'd", "how'll", "how's", "I'd", "I'll", "I'm",
                       "I've", "isn't", "it'd", "it'll", "it's", "mightn't",
                       "might've", "mustn't", "must've", "needn't",
                       "oughtn't",
                       "shan't", "she'd", "she'll", "she's", "shouldn't",
                       "should've", "somebody's", "someone'd", "someone'll",
                       "someone's", "that'll", "that's", "that'd", "there'd",
                       "there're", "there's", "they'd", "they'll", "they're",
                       "they've", "wasn't", "we'd", "we'll", "we're", "we've",
                       "weren't", "what'd", "what'll", "what're", "what's",
                       "whats",  # technically incorrect but some STT outputs
                       "what've", "when's", "when'd", "where'd", "where's",
                       "where've", "who'd", "who'd've", "who'll", "who're",
                       "who's", "who've", "why'd", "why're", "why's", "won't",
                       "won't've", "would've", "wouldn't", "wouldn't've",
                       "y'all", "ya'll", "you'd", "you'd've", "you'll",
                       "y'aint", "y'ain't", "you're", "you've"]
        if word.lower() in contraction:
            expansion = ["is not", "are not", "can not", "could have",
                         "could not", "did not", "does not", "do not",
                         "going to", "got to", "had not", "has not",
                         "have not", "he would", "he will", "he is",
                         "how did",
                         "how will", "how is", "I would", "I will", "I am",
                         "I have", "is not", "it would", "it will", "it is",
                         "might not", "might have", "must not", "must have",
                         "need not", "ought not", "shall not", "she would",
                         "she will", "she is", "should not", "should have",
                         "somebody is", "someone would", "someone will",
                         "someone is", "that will", "that is", "that would",
                         "there would", "there are", "there is", "they would",
                         "they will", "they are", "they have", "was not",
                         "we would", "we will", "we are", "we have",
                         "were not", "what did", "what will", "what are",
                         "what is",
                         "what is", "what have", "when is", "when did",
                         "where did", "where is", "where have", "who would",
                         "who would have", "who will", "who are", "who is",
                         "who have", "why did", "why are", "why is",
                         "will not", "will not have", "would have",
                         "would not", "would not have", "you all", "you all",
                         "you would", "you would have", "you will",
                         "you are not", "you are not", "you are", "you have"]
            word = expansion[contraction.index(word.lower())]

        if make_singular:
            nlp = nlp or get_nlp()
            word = singularize(word, nlp=nlp)
        normalized += " " + word

    if solve_corefs:
        normalized = replace_coreferences(normalized[1:], coref_nlp)

    return normalized.strip()


if __name__ == "__main__":
    sentence = "What's    the weather     like?"
    print(normalize(sentence))
    # NOTE, contractions are lower cased when expanded
    assert normalize(sentence) == "what is weather like?"

    sentence = "My sister loves dogs."
    assert normalize(sentence, make_singular=True) == "My sister love dog."

    sentence = "My sister has a dog. She loves him."

    assert  normalize(sentence, solve_corefs=True) == "My sister has a dog. " \
                                                     "My sister loves a dog."