import flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
app = flask.Flask(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

def training(df_train, clf_params):
    params = {'alpha':1.0, 'min_samples':2, 'min_cluster_size':5, 'cluster_selection_epsilon':0.0}
    params.update(clf_params)
    
    alpha = params['alpha']
    min_samples = params['min_samples']
    min_cluster_size = params['min_cluster_size']
    cluster_selection_epsilon = params['cluster_selection_epsilon']
    
    if 'emb' in df_train.columns:
        X = df_train['emb'].tolist()
    else:
        corpus = df_train['text'].tolist()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        
    clustering = hdbscan.HDBSCAN(alpha=alpha, min_samples=min_samples, min_cluster_size=min_cluster_size,
                                cluster_selection_epsilon=cluster_selection_epsilon).fit(X)

    df_result = pd.DataFrame(columns=['id','cluster','probability'])
    df_result['id'] = df_train['id'].copy()
    df_result['cluster'] = clustering.labels_
    df_result['probability'] = clustering.probabilities_

    res = {}
    res['df_res'] = df_result.to_dict()
            
    return clustering, res

@app.route('/hdbscan/', methods=['POST'])
def home():
    df_train = request.json['train_data']
    df_train = pd.DataFrame(df_train)
    df_train.index = df_train.index.astype(int)
#     print(type(df_train), df_train.shape, df_train.columns)
#     print(df_train.index)
    try:
        clf_params = request.json['clf_params']
    except:
        clf_params = {}
    
    clf, res = training(df_train, clf_params)
            
    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7657)
