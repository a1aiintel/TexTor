from TexTor.settings import ALLENNLP_URL
import requests


def NER(text):
    """

    Args:
        text:

    Returns:

    """
    try:
        url = ALLENNLP_URL + "named-entity-recognition"
        data = {"sentence": text}
        r = requests.post(url, json=data).json()
        words = r["words"]
        tags = r["tags"]
        ents = []
        for idx, tag in enumerate(tags):
            if tag != "O":
                ents.append((words[idx], tag))
        return ents
    except Exception as e:
        print(e)
    return text


def textual_entailment(premise, hypothesis):
    """
    Textual Entailment (TE) takes a pair of sentences and predicts whether the facts in the first necessarily imply the facts in the second one.
    The AllenNLP toolkit provides the following TE visualization, which can be run for any TE model you develop.
    This page demonstrates a reimplementation of the decomposable attention model (Parikh et al, 2017) ,
    which was state of the art for the SNLI benchmark (short sentences about visual scenes) in 2016.
    Rather than pre-trained Glove vectors, this model uses ELMo embeddings,
    which are completely character based and improve performance by 2%

    :param premise:
    :param hypotheses:
    :return:
    """
    url = ALLENNLP_URL + "textual-entailment"
    data = {"premise": premise,
            "hypothesis": hypothesis}
    r = requests.post(url, json=data).json()
    probs = r["label_probs"]
    return {"entailment": probs[0], "contradiction": probs[1],
            "neutral": probs[2]}


def machine_comprehension(question, passage):
    """
    Machine Comprehension (MC) answers natural language questions by selecting an answer span within an evidence text.
    The AllenNLP toolkit provides the following MC visualization, which can be used for any MC model in AllenNLP.
    This page demonstrates a reimplementation of BiDAF (Seo et al, 2017), or Bi-Directional Attention Flow,
    a widely used MC baseline that achieved state-of-the-art accuracies on the SQuAD dataset (Wikipedia sentences) in early 2017.

    :param question:
    :param passage:
    :return:
    """
    url = ALLENNLP_URL + "machine-comprehension"
    data = {"passage": passage, "question": question}
    r = requests.post(url, json=data).json()
    return r["best_span_str"]


def semantic_role_labeling(sentence):
    """
    Semantic Role Labeling (SRL) recovers the latent predicate argument structure of a sentence,
    providing representations that answer basic questions about sentence meaning,
    including “who” did “what” to “whom,” etc.
    The AllenNLP toolkit provides the following SRL visualization, which can be used for any SRL model in AllenNLP.
    This page demonstrates a reimplementation of a deep BiLSTM model (He et al, 2017),
    which is currently state of the art for PropBank SRL (Newswire sentences).
    :param sentence:
    :return:
    """
    url = ALLENNLP_URL + "semantic-role-labeling"
    data = {"sentence": sentence}
    r = requests.post(url, json=data).json()
    roles = {}
    words = r["words"]
    verbs = r["verbs"]
    for v in verbs:
        arg0 = []
        arg1 = []
        verb = v["verb"]
        tags = v["tags"]
        for idx, t in enumerate(tags):
            if "I-ARGM" in t:
                arg0.append(words[idx])
            elif "I-ARG1" in t:
                arg1.append(words[idx])
        if not len(arg0):
            continue
        arg0 = " ".join(arg0)
        arg1 = " ".join(arg1)
        roles[verb] = [arg0, arg1]
    return roles


def constituency_parse(sentence):
    """
    A constituency parse tree breaks a text into sub-phrases, or constituents.
    Non-terminals in the tree are types of phrases, the terminals are the words in the sentence.
    This demo is an implementation of a minimal neural model for constituency
    parsing based on an independent scoring of labels and spans described in
    Extending a Parser to Distant Domains Using a Few Dozen Partially
    Annotated Examples (Joshi et al, 2018). This model uses ELMo embeddings,
    which are completely character based and improves single model performance
    from 92.6 F1 to 94.11 F1 on the Penn Treebank, a 20% relative error reduction.
    :param sentence:
    :return:
    """
    url = ALLENNLP_URL + "constituency-parsing"
    data = {"sentence": sentence}
    r = requests.post(url, json=data).json()
    return r


def information_extraction(sentence):
    """
    Given an input sentence, Open Information Extraction (Open IE) extracts a list of propositions,
    each composed of a single predicate and an arbitrary number of arguments.
    These often simplify syntactically complex sentences, and make their
    predicate-argument structure easily accessible for various downstream tasks

    """
    url = ALLENNLP_URL + "open-information-extraction"
    data = {"sentence": sentence}
    r = requests.post(url, json=data).json()
    data["propositions"] = {}
    for v in r["verbs"]:
        verb = v["verb"]
        data["propositions"][verb] = {}
        data["propositions"][verb]["tags"] = v["tags"]
        data["propositions"][verb]["args"] = [
            a.split("]")[0].split(":")[1].strip()
            for a in v["description"].split("[")
            if a.startswith("ARG")]
    return data


def event2mind(sentence):
    """
    The Event2Mind dataset proposes a commonsense inference task between
    events and mental states. In particular, it takes events as lightly
    preprocessed text and produces likely intents and reactions for
    participants of the event. This page demonstrates a reimplementation of
    the original Event2Mind system (Rashkin et al, 2018). An event with
    people entities should be typed as "PersonX" or "PersonY".
    Optionally, "___" can be used as a placeholder for objects or phrases.
    :param sentence:
    :return:
    """
    url = ALLENNLP_URL + "event2mind"
    data = {"source": sentence}
    r = requests.post(url, json=data).json()
    data = {"sentence": sentence,
            # "subject": "PersonX",
            # "object": "PersonY",
            "subject_intent": [" ".join(a) for a in r[
                'xintent_top_k_predicted_tokens']],
            "subject_reaction": [" ".join(a) for a in r[
                'xreact_top_k_predicted_tokens']],
            "object_reaction": [" ".join(a) for a in r[
                'oreact_top_k_predicted_tokens']],
            "raw": r
            }
    return data


def documentqa(sentence):
    """
    run a web search on the question, and additionally try to identify
     Wikipedia articles about entities mentioned in the question.
    The resulting documents will be passed to a machine learning algorithm
    which will try to read the text and identify a span of text within one of
    the documents that answers your questions. No knowledge bases or
    other sources of information are used.

    Example Questions
        Who won the World Cup in 2014?
        What is a group of porcupines called?
        Which artist created the sculpture "The Thinker"?
        Where did Harry Potter go to school?
        What has the strongest magnet field in the Universe?
        The reaction where two atoms of hydrogen combine to form an atom of helium is called what?
        Who the president of Spain?
    Weaknesses/Limitations
        The system can answer short answer questions, most other forms of questions are unlikely work, including:
        yes/no questions ("Are tomatoes vegetables?")
        math problems ("What is 21*123?")
        multiple choice questions ("Which is taller, the Space Needle taller or the Empire States building?")
        questions that do not have a concrete answer or require longer output ("What happened during WW2?", "Who is Barrack Obama?")
        questions that ask for a list ("What are some of the uses of aluminum?")
    The system has some weaknesses you might observe
        Time: It tends to return answers that might have once been true, but are not true currently.
        Fact vs. Opinion: It does not have a good sense of when a statement can be trusted as a fact.
        Complex reasoning: It can perform multiple steps of inference (ex "Who won the world Cup during Obama's first term as President?")

    :param sentence:
    :return:
    """
    url = "https://documentqa.allenai.org/answer"
    data = {"question": sentence}
    r = requests.get(url, data).json()

    data = {"sentence": sentence,
            "answers": [],
            "short_answers": [],
            "sources": [],
            "corpus": [],
            "conf": [],
            "raw": r
            }
    for answer in r:
        data["sources"].append(answer["source_url"])
        data["corpus"].append(answer["text"])
        data["conf"].append(answer["answers"][0]["conf"])
        ans = answer["text"][
              answer["answers"][0]["start"]:answer["answers"][0]["end"]]
        data["short_answers"].append(ans)
        for sent in answer["text"].replace(":", ".").replace("\n", ".").split(
                "."):
            if ans in sent:
                data["answers"].append(sent)
                break
    return data


