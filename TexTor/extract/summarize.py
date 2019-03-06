import nltk
import urllib
import bs4 as bs
import re
import heapq


def wikipedia_pre_process(text):
    text = re.sub(r"\[[0-9]*\]", " ", text)  # For creating the "Summary"
    text = re.sub(r"\s+", " ", text)
    return text

def summarize_web(url, html_processing_callback=None):
    # Getting the data
    source = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(source, "html.parser")  # parsed source

    text = ""
    for paragraph in soup.find_all("p"):  # extracting the text from <p> tag of wikipedia page source
        text += paragraph.text

    # Pre-processing the text
    if html_processing_callback:
        text = html_processing_callback(text)

    return summarize(text)


def summarize(text):
    # Tokenizing the sentence into sentences
    sentences = nltk.sent_tokenize(text)

    # Adding all stopwords to a list
    stop_words = nltk.corpus.stopwords.words("english")

    # Histogram
    word2Count = {}
    for word in nltk.word_tokenize(text):
        if word not in stop_words:
            if word not in word2Count.keys():
                word2Count[word] = 1
            else:
                word2Count[word] += 1

    # Weighted Histogram
    for key in word2Count.keys():
        word2Count[key] = word2Count[key] / max(word2Count.values())

    sent_Score = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2Count.keys():
                if len(sentence.split(" ")) < 15:  # We only want smaller length sentences
                    if sentence not in sent_Score.keys():
                        sent_Score[sentence] = word2Count[word]
                    else:
                        sent_Score[sentence] += word2Count[word]

    # Summary
    return heapq.nlargest(15, sent_Score, sent_Score.get)


if __name__ == "__main__":

    text = "London is the capital and most populous city of England and  the United Kingdom." \
           "Standing on the River Thames in the south east of the island of Great Britain," \
           "London has been a major settlement  for two millennia.  It was " \
           "founded by the Romans, who named it Londinium."
    for s in summarize(text):
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