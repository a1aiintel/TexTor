from TexTor.remote import cogcomp_demo


def cogcomp_coref_triples(text):
    ignores = ["he", "she", "it", "they", "them", "these", "whom", "whose",
               "who", "its", "it's"]
    triples = []
    data = cogcomp_demo(text)
    links = data["links"]
    node_ids = {}
    for n in data["nodes"]:
        node_ids[int(n["id"])] = n["name"]
    for l in links:
        if l["source"] not in node_ids.keys() or l["target"] not in node_ids.keys():
            continue
        if node_ids[l["source"]] in ignores or node_ids[
            l["target"]] in ignores:
            continue
        if node_ids[l["source"]].lower() == node_ids[l["target"]].lower():
            continue
        triple = (node_ids[l["source"]], "is", node_ids[l["target"]])
        if triple not in triples:
            triples.append(triple)
    return triples


if __name__ == "__main__":
    print(cogcomp_coref_triples("my dog is alive"))
