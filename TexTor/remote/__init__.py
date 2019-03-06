import requests

# use the source https://cogcomp.org/page/demo_view/Coref
def cogcomp_demo(text):
    url = "https://cogcomp.org/demo_files/Coref.php"
    data = {"lang": "en", "text": text}
    r = requests.post(url, json=data)
    return r.json()