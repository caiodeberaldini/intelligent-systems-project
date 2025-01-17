import os
import sys

import pandas as pd

from dotenv import load_dotenv
from flask import Flask, request, jsonify

from utils.load_utils import load_utils
from utils.process_text import process_text

app = Flask(__name__)

@app.route('/<name>', methods=["GET"])
def hello_world(name):
    return {'Name': f'Hello, {name}!'}

@app.route('/v1/categorize', methods=["POST"])
def classifyProducts():
    try:
        content = request.get_json()
    except:
        return {"Message": "Unsupported input!"}, 400

    (model, vectorizer, le) = load_utils()

    products = [product['title'] for i, product in enumerate(content['products'])]
    process_products = process_text(pd.Series(products))

    proc_prods = vectorizer.transform(process_products)

    results = [model.predict(prod)[0] for prod in proc_prods]
    results = list(le.inverse_transform(results))
    results = {"categories": results}

    # Assessment of the model's test accuracy (test purpose only!)
    # with open('./metrics_test.txt', 'w+') as f:
    #     categories = [product['category'] for i, product in enumerate(content['products'])]
    #     metrics = 100*model.score(proc_prods, categories)
    #     metrics = f'Accuracy (Test data): {metrics:.2f}%'

    #     f.write(metrics)
    #     f.close()

    return jsonify(results)