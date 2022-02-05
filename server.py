from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time
import sklearn
import joblib
import pickle
import pandas as pd

app = Flask(__name__, template_folder='.')
build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query)  # возвращает список из (document, ind)
    print(documents)
    documents = sorted(score(query, documents), key=lambda x: -x[1])
    results = [doc[0].format(query) + ['%.2f' % doc[1]] for doc in documents][:30]
    return render_template(
        'index.html',
        time="%.2f" % (time() - start_time),
        query=query,
        search_engine_name='Yandex',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
