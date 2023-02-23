import openai
import requests
from bs4 import BeautifulSoup
import numpy as np
import json

def get_embeddings() -> dict:
    embeddings = {}
    # 57315
    start = 57315 - 15000
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
    for i in range(0,25000):
        url = f"https://www.workatastartup.com/jobs/{start + i}"
        print(url)
        html = requests.get(url=url, headers=headers).text
        soup = BeautifulSoup(html, 'html.parser')
        job_element = soup.find(class_="company-details")
        if not job_element: 
            print("no job element")
            continue
        content = job_element.get_text()
        if not content: 
            print("no conetent")
            continue
        data = openai.Embedding.create(input=content, model="text-embedding-ada-002")
        embeddings[url] = data["data"][0]["embedding"]
    return embeddings

def cosine_similarity(emb1, emb2) -> float:
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity

def cosine_similarity_dict(embeddings_dict: dict):
    similarities_dict = {}
    urls = list(embeddings_dict.keys())
    for i in range(len(urls)):
        for j in range(i+1, len(urls)):
            emb1 = embeddings_dict[urls[i]]
            emb2 = embeddings_dict[urls[j]]
            # similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities_dict[(urls[i], urls[j])] = cosine_similarity(emb1, emb2)
    sorted_similarities = sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities


# embeddings = { "www.google.com": [0.2, 0.1, 0.4], "www.facebook.com": [0.3, 0.2, 0.1], "www.twitter.com": [0.1, 0.3, 0.2] }

def make_embeddings():
    embeddings = get_embeddings()
    with open("similarities.json", "w") as f:
        json.dump(embeddings, f)

def run_embeddings():
    with open("similarities.json", "r") as f:
        embeddings = json.load(f)
        res = cosine_similarity_dict(embeddings)
        for x in res:
            print("------------------------", x[0][0], x[0][1], x[1], "\n", sep="\n")

def match_input():
    with open("similarities.json", "r") as f:
        embeddings = json.load(f)

    val = """
    The individual's CV showcases their experience and expertise in the tech industry. They dropped out of high school to join Depict.ai, where they built the first product recommender system using image CNN and other content-based techniques.

    They then founded and built hittavaran.com, a price comparison site with scraping capabilities for hand sanitizers and masks during the pandemic. They also owned and built the video-call product troubleshooter at Mavenoid.

    They became the interim CTO of Curb Food, where they built a full kitchen management system and a customer app.

    Finally, they were a founding engineer at Dataland, where they built a developer-first Airtable focused on performance, and implemented features such as search, sort, and filter. They built a browser in a browser, taking inspiration from Chrome's internal workings, and wrote GPU calls in Rust to render rows at 60fps on a x6 CPU slowdown. Their focus on frontend engineering fundamentals and building products that never drop a frame while still meeting deadlines is impressive. Technologies used include image CNN, webRTC, Rust, and V8's JIT, among others.
    """ #input("Enter your value: ")
    new_embedding = openai.Embedding.create(input=val, model="text-embedding-ada-002")["data"][0]["embedding"]
    print("Running similarity for: ", val, "...")
    similarities_dict = {}
    for (url, embedding) in embeddings.items():
        similarity = np.dot(new_embedding, embedding) / (np.linalg.norm(new_embedding) * np.linalg.norm(embedding))
        similarities_dict[url] = similarity

        
    sorted = sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True)
    for i, x in enumerate(sorted):
        print(x[0], x[1], sep="\n")
        if i == 10: break
    print("\n\n\n")

make_embeddings()


