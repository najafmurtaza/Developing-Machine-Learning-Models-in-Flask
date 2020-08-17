import flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
app = flask.Flask(__name__)

import subprocess
from pathlib import Path
import json
import string
import random
import fasttext
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from mlflow import log_metric, set_tag, log_param, log_params, log_artifact, set_experiment, end_run, start_run, set_tracking_uri

do_ml_flow = False

def training(df_train, df_valid, model_name, clf_params):
    params = {'lr':0.1, 'epoch':200, 'word_ngrams':2, 'bucket':200000, 'dim':300, 'loss':'softmax',
             'autotuneDuration':1200, 'autotuneMetric':'f1'}
    params.update(clf_params)
    
    global do_ml_flow
    if do_ml_flow:
        log_params(params)
    
    lr = params['lr']
    epoch = params['epoch']
    word_ngrams = params['word_ngrams']
    bucket = params['bucket']
    dim = params['dim']
    loss = params['loss']
    autotuneDuration = params['autotuneDuration'] # 300 for 5 minutes
    autotuneMetric = params['autotuneMetric']
    
    f_name_trn = 'tmp/'+''.join(random.choices(string.ascii_uppercase + string.digits, k=7))+'.txt'
    f = open(f_name_trn,'w',encoding='utf8')
    train_size=df_train.shape[0]
    for i in range(df_train.shape[0]):
#         print(i, end='\r')
        label=str(df_train.loc[i,'label'])
        label=label.replace(" ", "-")
        label=label.lower()
        
        body=str(df_train.loc[i,'text'])
        body=body.translate(str.maketrans('','',string.punctuation))
        body=body.replace('\n',' ')
        body=body.lower()
        f.write('__label__'+label+' '+body+'\n')
    f.close()
#     print('\nWritten')
    
    if df_valid:
        f_name_val = 'tmp/'+''.join(random.choices(string.ascii_uppercase + string.digits, k=7))+'.txt'
        f = open(f_name_val,'w',encoding='utf8')
        valid_size=df_valid.shape[0]
    #     print(valid_size)
        for i in range(df_valid.shape[0]):
    #             print(i)
            label = str(df_valid.loc[i,'label'])
            label = label.replace(" ", "-")
            label = label.lower()

            body = str(df_valid.loc[i,'text'])
            body = body.translate(str.maketrans('','',string.punctuation))
            body = body.replace('\n',' ')
            body = body.lower()
            f.write('__label__'+label+' '+body+'\n')
        f.close()
#             print('\nWritten')

        classifier = fasttext.train_supervised(f_name_trn, lr=lr, epoch=epoch,
                                         word_ngrams=word_ngrams, bucket=bucket, dim=dim,
                                         autotuneValidationFile=f_name_val,
                                         autotuneDuration=autotuneDuration, #300 for 5 minutes
                                        autotuneMetric=autotuneMetric,
                                        loss=loss)
        subprocess.call(f'rm {f_name_trn}', shell=True)
        subprocess.call(f'rm {f_name_val}', shell=True)
        
        return classifier

    classifier = fasttext.train_supervised(f_name_trn, lr=lr, epoch=epoch,
                                     word_ngrams=word_ngrams, bucket=bucket, dim=dim,
                                     loss=loss)
    subprocess.call(f'rm {f_name_trn}', shell=True)
    
    return classifier

def testing(df_test, classifier=None, model_name=''):
    if classifier==None:
        classifier = fasttext.load_model(f'fasttext_models/{model_name}.bin')
    sent = []
    labels = []
    conf = []
    numbers = []
    
    t_size = df_test.shape[0]
    for i in range(df_test.shape[0]):
        numbers.append(df_test.loc[i,'id'])
        label = str(df_test.loc[i,'label'])
        label = label.replace(" ", "-")
        label = label.lower()
        labels.append(label)

        body = str(df_test.loc[i,'text'])
        body = body.translate(str.maketrans('','',string.punctuation))
        body = body.replace('\n',' ')
        body = body.lower()
        sent.append(body)

    pred = []
    for i, s in enumerate(sent):
        prediction = classifier.predict(s)
        pred.append(prediction[0][0][9:])
        conf.append(prediction[1][0])
        
    df_result = pd.DataFrame(columns=['id','real','pred','conf'])
    df_result['id'] = numbers
    df_result['pred'] = pred
    df_result['conf'] = conf
    df_result['real'] = labels

    acc = accuracy_score(labels, pred)
    report = classification_report(labels, pred, output_dict=True)
    report = pd.DataFrame(report).T
#     report = report[~report.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    report_txt = classification_report(labels, pred)
    conf_mat = pd.crosstab(df_result['real'], df_result['pred'], rownames=['Actual'], colnames=['Pred'])
    
    global do_ml_flow
    if do_ml_flow:
        log_metric("Test Accuracy", acc)
        
        df_result.to_excel(f'tmp/df_result -- {model_name}.xlsx', index=False)
        report.to_excel(f'tmp/report -- {model_name}.xlsx')
        with open(f'tmp/report -- {model_name}.txt', 'w') as f:
            print(report_txt, file=f)
        conf_mat.to_excel(f'tmp/conf_mat -- {model_name}.xlsx')
    
        log_artifact(f'tmp/df_result -- {model_name}.xlsx')
        log_artifact(f'tmp/report -- {model_name}.xlsx')
        log_artifact(f'tmp/report -- {model_name}.txt')
        log_artifact(f'tmp/conf_mat -- {model_name}.xlsx')
        
        subprocess.call(f'rm "tmp/df_result -- {model_name}.xlsx"', shell=True)
        subprocess.call(f'rm "tmp/report -- {model_name}.xlsx"', shell=True)
        subprocess.call(f'rm "tmp/report -- {model_name}.txt"', shell=True)
        subprocess.call(f'rm "tmp/conf_mat -- {model_name}.xlsx"', shell=True)

    res = {}
    res['df_res'] = df_result.to_dict()
    res['acc'] = acc
    res['report'] = report.to_dict()
    res['report_txt'] = report_txt
    res['conf_mat'] = conf_mat.to_dict()
    
    return res

@app.route('/fasttext/', methods=['POST'])
def home():
    df_train = request.json['train_data']
    df_train = pd.DataFrame(df_train)
    df_train.index = df_train.index.astype(int)
#     print(type(df_train), df_train.shape, df_train.columns)
#     print(df_train.index)
    df_test = request.json['test_data']
    df_test = pd.DataFrame(df_test)
#     print(type(df_test), df_test.shape, df_test.columns)
    df_test.index = df_test.index.astype(int)
    try:
        df_valid = request.json['valid_data']
        df_valid = pd.DataFrame(df_valid)
#         print(type(df_valid), df_valid.shape, df_valid.columns)
        df_valid.index = df_valid.index.astype(int)
    except:
        df_valid = None
    try:
        clf_params = request.json['clf_params']
    except:
        clf_params = {}
    try:
        save_model = request.json['save_model']
    except:
        save_model = False
    try:
        model_name = request.json['model_name']
    except:
        model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    try:
        ml_flow_params = request.json['ml_flow_params']
        
        experiment_name = ml_flow_params['experiment_name']
        run_name = ml_flow_params['run_name']
        description = ml_flow_params['description']
        
        global do_ml_flow
        do_ml_flow = True
    except:
        pass
    
    if do_ml_flow:
        set_experiment(experiment_name)
        start_run(run_name=run_name)
        
        set_tag("mlflow.note.content", description)

    clf = training(df_train, df_valid, model_name, clf_params)
    if save_model:
        clf.save_model(f'fasttext_models/{model_name}.bin')
    res = testing(df_test, clf, model_name)
    res['model_name'] = model_name
    
    if do_ml_flow:
        log_param("model_name", model_name)
        end_run()
    
    return jsonify(res)

@app.route('/fasttext/test', methods=['POST'])
def test():
    df_test = request.json['test_data']
    df_test = pd.DataFrame(df_test)
#     print(type(df_test), df_test.shape, df_test.columns)
    df_test.index = df_test.index.astype(int)
    model_name = request.json['model_name']
    
    my_file = Path(f'fasttext_models/{model_name}.bin')
    if not my_file.is_file():
        res = {'status':f'Model \'{model_name}\' Doesn\'t Exists'}
    else:
        res = testing(df_test, model_name=model_name)

    return jsonify(res)

@app.route('/fasttext/delete', methods=['POST'])
def delete():
    model_name = request.json['model_name']
    
    my_file = Path(f'fasttext_models/{model_name}.bin')
    if not my_file.is_file():
        res = {'status':f'Model \'{model_name}\' Doesn\'t Exists'}
    else:
        subprocess.call(f'rm fasttext_models/{model_name}.bin', shell=True)
        res = {'status':f'Model \'{model_name}\' Deleted'}
        
    return jsonify(res)

@app.route('/fasttext/mlflow', methods=['POST'])
def mlflow():
    res = {'url':'http://{system_ip}:3456/'}

    return jsonify(res)

if __name__ == '__main__':
    DATA_PATH = Path('fasttext_models/')
    DATA_PATH.mkdir(exist_ok=True)
    DATA_PATH = Path('fasttext_mlruns/')
    DATA_PATH.mkdir(exist_ok=True)
    DATA_PATH = Path('tmp/')
    DATA_PATH.mkdir(exist_ok=True)
    
    set_tracking_uri('./fasttext_mlruns/')
    
    app.run(host='0.0.0.0', port=7655)
