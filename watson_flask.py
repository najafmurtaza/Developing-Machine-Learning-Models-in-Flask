import flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
app = flask.Flask(__name__)

import subprocess
import json
from pathlib import Path
import os
import string
import random
from ibm_watson import NaturalLanguageClassifierV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from watson_test import testing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def training(f_name, api_key, model_name):
    authenticator = IAMAuthenticator(api_key)
    natural_language_classifier = NaturalLanguageClassifierV1(
        authenticator=authenticator)
    natural_language_classifier.set_service_url('https://gateway.watsonplatform.net/natural-language-classifier/api')
    
    with open(os.path.join(os.path.dirname('__file__'), f_name), 'rb') as training_data:
        metadata = json.dumps({'name': model_name, 'language': 'en'})
        classifier = natural_language_classifier.create_classifier(
            training_metadata=metadata,
            training_data=training_data
        ).get_result()
    
    subprocess.call(f'rm {f_name}', shell=True)
    
    return classifier

@app.route('/watson/', methods=['POST'])
def home():
    df_train = request.json['train_data']
    df_train = pd.DataFrame(df_train)
#     print(type(df_train), df_train.shape, df_train.columns)
    df_train.index = df_train.index.astype(int)
    api_key = request.json['api_key']
#     email_id = request.json['email_id']
    try:
        model_name = request.json['model_name']
    except:
        model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    
    f_name = 'tmp/'+''.join(random.choices(string.ascii_uppercase + string.digits, k=7))+'.csv'
    df_train = df_train[:15000]
    df_train['text'] = df_train['text'].str[:1024]
    df_train['text'] = df_train['text'].apply(lambda x: ((x.encode("unicode_escape").decode("utf-8"))[:1024]).strip())
    df_train.to_csv(f_name, header=False, index=False)

    res = training(f_name, api_key, model_name)
    params = {'df_test':request.json['test_data'], 'api_key':api_key, 'model_name':model_name}
    try:
        ml_flow_params = request.json['ml_flow_params']
        params['ml_flow_params'] = ml_flow_params
    except:
        pass
    params = json.dumps(params)
    p = subprocess.Popen(['python', 'watson_test.py', params])
    
    res['message'] = 'Here is model info. Results will be emailed once classifier is trained'
    res['model_name'] = model_name
    
    return jsonify(res)

@app.route('/watson/test', methods=['POST'])
def test():
    df_test = request.json['test_data']
    model_name = request.json['model_name']
    api_key = request.json['api_key']
#     email_id = request.json['email_id']
    try:
        emailed = request.json['emailed']
    except:
        emailed = False
    
    res = {}
    if emailed==False:
        res = testing(df_test, model_name, api_key)
    else:
        params = {'df_test':request.json['test_data'], 'api_key':api_key, 'model_name':model_name}
        params = json.dumps(params)
        p = subprocess.Popen(['python', 'watson_test.py', params])
        res['message'] = 'Results will be emailed once data is tested'
    
    return jsonify(res)

@app.route('/watson/results', methods=['POST'])
def results():
    model_name = request.json['model_name']
    try:
        delete_files = request.json['delete_files']
    except:
        delete_files = False
        
    my_file = Path(f'watson_results/{model_name}.json')
    if not my_file.is_file():
        res = {'message':f"Results of '{model_name}' Doesn't Exists"}
    else:
        with open(f'watson_results/{model_name}.json', 'r') as fp:
            res = json.load(fp)
    
    if delete_files:
        subprocess.call(f'rm watson_results/{model_name}.json', shell=True)
        res['message'] = f"Model '{model_name}' Results Deleted"
    
    return jsonify(res)
        
@app.route('/watson/delete', methods=['POST'])
def delete():
    model_name = request.json['model_name']
    api_key = request.json['api_key']
    
    authenticator = IAMAuthenticator(api_key)
    natural_language_classifier = NaturalLanguageClassifierV1(
        authenticator=authenticator)
    natural_language_classifier.set_service_url('https://gateway.watsonplatform.net/natural-language-classifier/api')
    
    classifier_id = None
    for c in natural_language_classifier.list_classifiers().get_result()['classifiers']:
        if c['name']==model_name:
            classifier_id = c['classifier_id']
    if not classifier_id:
        return {'message':"Classifier not found, check its name again"}
    else:
        natural_language_classifier.delete_classifier(classifier_id)
        res = {'message':f"Model '{model_name}' Deleted"}
        
    return jsonify(res)

@app.route('/watson/mlflow', methods=['POST'])
def mlflow():
    res = {'url':'http://{system_ip}:3457/'}

    return jsonify(res)

if __name__ == '__main__':
    DATA_PATH = Path('watson_results/')
    DATA_PATH.mkdir(exist_ok=True)
    DATA_PATH = Path('watson_mlruns/')
    DATA_PATH.mkdir(exist_ok=True)
    DATA_PATH = Path('tmp/')
    DATA_PATH.mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=7656)
