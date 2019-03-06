# TexTor

nlp toolbox, built on the shoulders of giants

Nothing new or too fancy here, just a handy collection of tools in a single place


## Text Normalization

text is messy, TexTor normalizes text to make it easier to work with

remove extra spaces, expand contractions, remove articles, solve coreference

```python
from TexTor.utils import normalize

sentence = "What's    the weather     like?"
# NOTE, contractions are lower cased when expanded
assert normalize(sentence, remove_articles=True) == "what is weather like?"

sentence = "My sister loves dogs."
assert normalize(sentence, make_singular=True) == "My sister love dog."

sentence = "My sister has a dog. She loves him."

assert  normalize(sentence, solve_corefs=True) == "My sister has a dog. " \
                                                 "My sister loves a dog."
```


## Grammar Shenanigans

Grammar rules are very useful

```python
from TexTor.understand.tagging import SentenceTagger

t = SentenceTagger()

tagged = t.tag_sentence('Machine Learning is awesome')
print(tagged)
# [('Machine', 'NN-TL'), ('Learning', 'VBG'), ('is', 'BEZ'), ('awesome', 'JJ')]
    
assert t.is_passive('Mistakes were made.')
assert not t.is_passive('I made mistakes.')
# Notable fail case. Fix me. I think it is because the 'to be' verb is omitted.
# assert t.is_passive('guy shot by police')

assert t.change_tense("I am making dinner",
                      "past") == "I was making dinner"
assert t.change_tense("I am making dinner",
                      "future") == "I will be making dinner"
assert t.change_tense("I am making dinner",
                      "present") == "I am making dinner"
```

TexTor bundles Regular expressions-based rules for English word [inflection](./TexTor/understand/inflect.py):

* pluralization and singularization of nouns and adjectives,
* conjugation of verbs,
* comparative and superlative of adjectives.

```python
from TexTor.understand.inflect import singularize, pluralize, definite_article, indefinite_article, comparative, superlative

assert singularize("dogs") == "dog"
assert pluralize("dog") == "dogs"
assert definite_article("dog") == "the"
assert indefinite_article("dog") == "a"
assert indefinite_article("airplane") == "an"
assert comparative("ugly") == "uglier"
assert superlative("ugly") == "ugliest"

```

## Lexicon based analysis

The simplest approach to retrieve information about words is to use 
wordlists or lexicons, this however ignores all context surrounding the word

Lexicons/Wordlists Used

* NRC Emotion Lexicon - http://www.saifmohammad.com/WebPages/lexicons.html
* Bing Liu's Opinion Lexicon - http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
* MPQA Subjectivity Lexicon - http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
* Harvard General Inquirer - http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm
* NRC Word-Colour Association Lexicon - http://www.saifmohammad.com/WebPages/lexicons.html
* AFINN - https://github.com/fnielsen/afinn

```python
from TexTor.lexicons.word_analysis import *

word = "love"
print(get_color(word))
print(get_emotion(word))
print(get_orientation(word))
print(get_sentiment(word))
print(get_subjectivity(word))

from TexTor.lexicons.sentiment_analysis import get_afinn_score

assert get_afinn_score('This is utterly excellent!') == 3.0

```

## NER

A common task is to tag entities in text, TexTor wraps Named Entity 
Recognition functionality from several libs

nltk is used by default, other options are Fox, Polyglot and Spacy

```python
from TexTor.extract.ner import NER

assert NER("The Taj Mahal was built by Emperor Shah Jahan") == \
           [{'label': 'ORGANIZATION',
             'pos_tag': ['NNP', 'NNP'],
             'tokens': ['Taj', 'Mahal']},
            {'label': 'PERSON',
             'pos_tag': ['NNP', 'NNP', 'NNP'],
             'tokens': ['Emperor', 'Shah', 'Jahan']}]
             
```

## Segmentation

When we have big corpus of text we usually want to split it at the sentence 
level

```python
from TexTor.extract.segmentation import extract_formatted_sentences

document = "London is the capital and most populous city of England and " \
           "the United Kingdom. Standing on the River Thames in the south" \
           " east of the island of Great Britain, London has been a major" \
           " settlement for two millennia.  It was founded by the Romans," \
           " who named it Londinium."

assert extract_formatted_sentences(document) == [
    'London is the capital and most populous city of England and the United Kingdom .',        
    'Standing on the River Thames in the south east of the island of Great Britain , London has been a major settlement for two millennia .',
    'It was founded by the Romans , who named it Londinium .']


```

## Text Summarization

Very often we want to extract key sentences from corpus of text

Textor provides methods for this


```python
from TexTor.extract.summarize import summarize, summarize_web

for s in summarize_web(text_corpus):
    print(s)
    
url = "https://en.wikipedia.org/wiki/Dog"
for s in summarize_web(url, 
                       html_processing_callback=wikipedia_pre_process):
    print(s)

"""
Dogs were depicted to symbolize guidance, protection, loyalty, fidelity, faithfulness, watchfulness, and love.
Tigers in Manchuria, Indochina, Indonesia, and Malaysia are also reported to kill dogs.
In China, Korea, and Japan, dogs are viewed as kind protectors.
Some other signs are abdominal pain, loss of coordination, collapse, or death.
Every year, between 6 and 8 million dogs and cats enter US animal shelters.
In some cultures, however, dogs are also a source of meat.
For instance, dogs would have improved sanitation by cleaning up food scraps.
In some hunting dogs, however, the tail is traditionally docked to avoid injuries.
Striped hyenas are known to kill dogs in Turkmenistan, India, and the Caucasus.
However, pet dog populations grew significantly after World War II as suburbanization increased.
In Norse mythology, a bloody, four-eyed dog called Garmr guards Helheim.
In 2013, an estimate of the global dog population was 987 million.
The museum contains ancient artifacts, fine art, and educational opportunities for visitors.
Male French Bulldogs, for instance, are incapable of mounting the female.
A common breeding practice for pet dogs is mating between close relatives
"""
             
```

## allennlp toolbox

The juicy stuff uses actual machine learning, i opted for not including 
everything in TexTor, but wrappers are provided to use remote services

https://github.com/allenai/allennlp-demo
http://demo.allennlp.org


```python
from TexTor.remote.allennlp import textual_entailment, documentqa, event2mind, information_extraction, semantic_role_labeling, machine_comprehension, constituency_parse, NER

# TODO usage examples and testing

```

## Word2Vec

Word2Vec functionality is also provided, basically a small wrapper around gensim.Word2Vec

```python
from TexTor.understand.word_vectors import WordTwoVec

wv = WordTwoVec("your_model.bin")
vector1 = wv.embed(["list", "of", "words"])
vector2 = wv.embed(["another", "sentence"])
print(wv.cosine_similarity(vector1, vector2))

```