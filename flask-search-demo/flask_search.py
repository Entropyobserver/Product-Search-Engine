from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os
import time

app = Flask(__name__)

# Global variables
model = None
df_products = None
product_embeddings = None
product_ids = None
product_titles = None
product_descriptions = None

def load_model_and_data():
    global model, df_products, product_embeddings
    global product_ids, product_titles, product_descriptions

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    SAMPLE_SIZE = 2000
    cache_dir = "flask_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Load or sample product data
    cache_file = os.path.join(cache_dir, f"product_data_sample_{SAMPLE_SIZE}.pkl")
    if os.path.exists(cache_file):
        df_products = pd.read_pickle(cache_file)
    else:
        df_products = pd.read_parquet('../shopping_queries_dataset/shopping_queries_dataset_products.parquet')
        df_products = df_products[df_products['product_locale'] == 'us']
        df_products = df_products.sample(SAMPLE_SIZE, random_state=42)
        df_products['product_description'] = df_products['product_bullet_point'].apply(lambda x: x if isinstance(x, str) else '')
        df_products.to_pickle(cache_file)

    product_ids = df_products['product_id'].tolist()
    product_titles = df_products['product_title'].tolist()
    product_descriptions = df_products['product_description'].tolist()

    embeddings_file = os.path.join(cache_dir, f"product_embeddings_sample_{SAMPLE_SIZE}.pt")
    if os.path.exists(embeddings_file):
        product_embeddings = torch.load(embeddings_file)
    else:
        product_embeddings = model.encode(product_titles, convert_to_tensor=True)
        torch.save(product_embeddings, embeddings_file)

    print("Model and data loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    top_k = int(request.form.get('top_k', 10))

    if not query:
        return jsonify({'error': 'No query provided'})

    start = time.time()
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]
    top_values, top_indices = torch.topk(similarity_scores, k=min(top_k, len(similarity_scores)))

    results = []
    for score, idx in zip(top_values.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({
            'product_id': product_ids[idx],
            'title': product_titles[idx],
            'description': product_descriptions[idx],
            'score': score
        })

    return jsonify({
        'query': query,
        'results': results,
        'time': time.time() - start
    })

@app.route('/sample_queries')
def sample_queries():
    return jsonify({'queries': [
        "women's running shoes",
        "iphone charger cable",
        "gaming laptop",
        "wireless bluetooth earbuds",
        "kitchen knife set",
        "water bottle",
        "men's watch",
        "yoga mat"
    ]})

if __name__ == '__main__':
    os.makedirs("templates", exist_ok=True)

    # Ensure index.html exists
    if not os.path.exists("templates/index.html"):
        with open('templates/index.html', 'w') as f:
            f.write("<h1>Product Search Engine is Running</h1>")

    load_model_and_data()
    app.run(debug=False, host='0.0.0.0', port=8080)
