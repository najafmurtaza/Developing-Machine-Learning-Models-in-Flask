import flask
from flask import request
from flask import jsonify
import numpy as np
app = flask.Flask(__name__)

from gensen import GenSen, GenSenSingle
import json


def embeddings(list_mystr):
    reps_h, reps_h_t = gensen_1.get_representation(
        list_mystr, pool='last', return_numpy=True, tokenize=True
    )
    vectors = reps_h_t.tolist()
    return vectors


@app.route('/get_embeddings/', methods = ['POST'])
def home():
    sentences_list = list(request.json['sentences_list'])
    sentences_list = [x.lower().encode("unicode_escape").decode("utf8") for x in sentences_list]

    if(not sentences_list):
        return "Arg \"sentences_list\", not found"
    vec = embeddings(sentences_list)
    # print(type(vec), len(vec))
    return jsonify(vectors=vec)


gensen_1 = GenSenSingle(
        model_folder='gensen/data/models',
        filename_prefix='nli_large_bothskip',
        pretrained_emb='gensen/data/embedding/glove.840B.300d.h5'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7654)
