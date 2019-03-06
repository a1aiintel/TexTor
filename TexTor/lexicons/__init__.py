# Lexicons Used
# NRC Emotion Lexicon - http://www.saifmohammad.com/WebPages/lexicons.html
# Bing Liu's Opinion Lexicon - http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
# MPQA Subjectivity Lexicon - http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
# Harvard General Inquirer - http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm
# NRC Word-Colour Association Lexicon - http://www.saifmohammad.com/WebPages/lexicons.html

from os.path import dirname, join


def load_lexicon():
    bucket = {}

    lexicon_path = join(dirname(__file__), "word_emotion_lexicon.csv")
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for l in lines[1:]:
            l = l.replace("\n", "")
            word, emotion, color, orientation, sentiment, subjectivity, source = l.split(",")
            bucket[word] = {"emotion": emotion,
                            "color": color,
                            "orientation": orientation,
                            "sentiment": sentiment,
                            "subjectivity": subjectivity,
                            "source": source}
    return bucket


LEXICON = load_lexicon()


if __name__ == "__main__":
    from pprint import pprint

    pprint(LEXICON)