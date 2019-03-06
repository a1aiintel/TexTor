import requests
from TexTor.remote import cogcomp_demo


def neuralcoref_demo(text):
    try:
        params = {"text": text}
        r = requests.get("https://coref.huggingface.co/coref",
                         params=params).json()
        text = r["corefResText"] or text
    except Exception as e:
        print(e)
    return text


def cogcomp_coref_resolution_demo(text):
    replaces = ["he", "she", "it", "they", "them", "these", "whom", "whose",
                "who", "its", "it's"]
    data = cogcomp_demo(text)
    links = data["links"]
    node_ids = {}
    replace_map = {}
    for n in data["nodes"]:
        node_ids[int(n["id"])] = n["name"]
    for l in links:
        # only replace some stuff
        if node_ids[l["target"]].lower() not in replaces:
            continue
        replace_map[node_ids[l["target"]]] = node_ids[l["source"]]
    for r in replace_map:
        text = text.replace(r, replace_map[r])
    return text

