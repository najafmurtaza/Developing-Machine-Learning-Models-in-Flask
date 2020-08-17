from ibm_watson import NaturalLanguageClassifierV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import numpy as np
import json
import sys
import pandas as pd
import subprocess

from mlflow import log_metric, set_tag, log_param, log_params, log_artifact, set_experiment, end_run, start_run, set_tracking_uri

do_ml_flow = False

def testing(df_test, model_name, api_key, email_id=[], emailed=False):
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
    
    df_test = pd.DataFrame(df_test)
#     print(type(df_test), df_test.shape, df_test.columns)
    df_test.index = df_test.index.astype(int)
    df_test['text'] = df_test['text'].str[:1024]
    df_test['text'] = df_test['text'].apply(lambda x: ((x.encode("unicode_escape").decode("utf-8"))[:1024]).strip())
    
    y_true = df_test['label'].to_numpy()
    y_pred = []
    conf = []
    numbers = []
    while(True):
        status = natural_language_classifier.get_classifier(classifier_id).get_result()['status']
        if (status=='Training') and (emailed==False):
            return {'message':"Classifier not Trained, try when its available"}
        if status=='Training':
            continue
        for num, example in zip(df_test['id'], df_test['text']):
            numbers.append(num)
            classes = natural_language_classifier.classify(classifier_id, example).get_result()
        #     print(classes)
            pred_label = classes['top_class']
            pred_conf = classes['classes'][0]['confidence']
        #     print(pred_label, pred_conf)
            y_pred.append(pred_label)
            conf.append(pred_conf)
        break

    y_pred = np.array(y_pred, dtype=np.object)
    numbers = np.array(numbers, dtype=np.object)
    conf = np.array(conf, dtype=np.float64)
    
    df_result = pd.DataFrame(columns=['real','pred','conf','id'])
    df_result['id'] = numbers
    df_result['pred'] = y_pred
    df_result['conf'] = conf
    df_result['real'] = y_true
    
    report = classification_report(y_true, y_pred, output_dict=True)
    report = pd.DataFrame(report).T
#     report = report[~report.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    report_txt = classification_report(y_true, y_pred)
    conf_mat = pd.crosstab(df_result['real'], df_result['pred'], rownames=['Actual'], colnames=['Pred'])
    acc = accuracy_score(y_true, y_pred)
    
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
        
    if emailed==False:
        res = {}
        res['df_res'] = df_result.to_dict()
        res['acc'] = acc
        res['report'] = report.to_dict()
        res['report_txt'] = report_txt
        res['conf_mat'] = conf_mat.to_dict()
        
        return res
    else:
        res = {}
        res['df_res'] = df_result.to_dict()
        res['acc'] = acc
        res['report'] = report.to_dict()
        res['report_txt'] = report_txt
        res['conf_mat'] = conf_mat.to_dict()
        
        with open(f'watson_results/{model_name}.json', 'w') as fp:
            json.dump(res, fp)
    
if __name__=='__main__':
    params = sys.argv[1]
    params = json.loads(params)
    
    df_test = params['df_test']
    model_name = params['model_name']
    api_key = params['api_key']
#     email_id = params['email_id']
    
    set_tracking_uri('./watson_mlruns/')
    try:
        ml_flow_params = params['ml_flow_params']
        
        experiment_name = ml_flow_params['experiment_name']
        run_name = ml_flow_params['run_name']
        description = ml_flow_params['description']
        
        do_ml_flow = True
    except:
        pass
    
    if do_ml_flow:
        set_experiment(experiment_name)
        start_run(run_name=run_name)
        
        set_tag("mlflow.note.content", description)
    
    testing(df_test, model_name, api_key, emailed=True)
    
    if do_ml_flow:
        log_param("model_name", model_name)
        end_run()
