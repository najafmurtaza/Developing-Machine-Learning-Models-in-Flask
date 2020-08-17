# Developing Machine Learning Models in Flask
Flask for training/testing Watson, FastText, Gensen Embeddings and hDBScan. You can setup them in one of your development server and access them from all of your servers.
This saves environment setup time and all of your team members can have access to best algorithm hyper parameters.
Plus it supports [MLFlow](https://github.com/mlflow/mlflow/) for model info logging.

**In files, where required, replace `system_ip` with your systemp IP address or localhost.**

### Supported Models
Right now following models are available:
1. Gensen
2. Watson<sup>[1]</sup>
3. Fast Text<sup>[1]</sup>
4. hDBScan

[1]. [MLFlow](https://github.com/mlflow/mlflow/) is integrated for logging. See Test.ipynb for usage

#### Env setup
    conda create -n new_environment --file conda-req.txt
    pip install -r pip-req.txt

    Install MLflow. Then run it as from this repo root directory as `mlflow server -p 3457 -h 0.0.0.0 --backend-store-uri watson_mlruns/`
    and `mlflow server -p 3456 -h 0.0.0.0 --backend-store-uri fasttext_mlruns/`

## Gensen
[Official Repo: https://github.com/Maluuba/gensen](https://github.com/Maluuba/gensen)

[Paper: https://arxiv.org/abs/1804.00079](https://arxiv.org/abs/1804.00079)

- Usage:
    ```
    def f(sent):
        r = requests.post('http://{system_ip}:7654/get_embeddings/', json={'sentences_list': [sent]})
        arr = np.asarray(json.loads(r.text)['vectors']).flatten()
        return arr
    
    data = pd.read_pickle('data/path')
    data['emb'] = data['text'].apply(f)
    data[['id','emb']].to_pickle('path_to_save/{f_name}.pkl')
    ```
        
## Watson:
**Due to security issues, `emailed:True` param doesn't email the results, but it stores them on disk and can be retrieved at `watson/results`. See below for more.**

Right now Watson endpoint offers 5 operations.
1) watson/

    It trains new watson nlc and perform testing on test data once it is trained. Results will be saved on disk and can be accessed at `watson/Results`.
    
    - Required Params:
        
        1. train_data: must have 'text', 'label' and 'id' col
        2. test_data: must have 'text', 'label' and 'id' col
        3. api_key
        
    - Optional Params:
        
        1. model_name: by default it will be randomnly generated for the user.
        2. ml_flow_params: (dict) to log the model in ML-flow
            
            should have
            ```
            experiment_name: For e.g "Text Classifier"
            run_name: for e.g "experiment v4"
            description: for e.g "using latest train data"
            ```
        
    - Usage:
    
        ```
        params = {'train_data': data[['text','label','id']].to_dict(),
                  'test_data': data[['text','label','id']].to_dict(),
                  'model_name':'my-test_clf',
                  'api_key':''}

        r = requests.post('http://{system_ip}:7656/watson/', json=params)
        res = json.loads(r.text)
        print(res.keys())
        ```
2) watson/test

    It gives prediction on passed data using trained nlc.
    
    - Required Params:
        
        1. test_data: must have 'text', 'label' and 'id' col
        2. api_key
        3. model_name: Trained nlc name
        
    - Optional Params:
        
        1. emailed: (default=False) if True, test func. will execute in background, save results on disk and can be accessed at `watson/Results`.
        
            if False, client will wait for the results.
        
    - Usage:
    
        ```
        params = {'test_data': data[['text','label','id']].to_dict(),
                  'model_name':'my-test_clf',
                  'api_key':''}

        r = requests.post('http://{system_ip}:7656/watson/test', json=params)
        res = json.loads(r.text)
        print(res.keys())
        df_res, acc, report, report_txt, conf_mat = res['df_res'], res['acc'], res['report'], res['report_txt'], res['conf_mat']
        df_res = pd.DataFrame(df_res)
        report = pd.DataFrame(report)
        conf_mat = pd.DataFrame(conf_mat)
        print(report_txt)
        ```
3) watson/results

    Get results of test data stored on server.
    
    - Required Params:
        
        1. model_name: Trained nlc name
        
    - Optional Params:
        
        1. delete_files: (default=False) if True, delete results from server after retrieving them.
        
            if False, do nothing.
        
    - Usage:
    
        ```
        params = {'model_name':'my-test_clf'}

        r = requests.post('http://{system_ip}:7656/watson/results', json=params)
        res = json.loads(r.text)
        print(res.keys())
        ```
4) watson/delete

    Delete nlc from Watson instance.
    
    - Required Params:
        
        1. model_name: Trained nlc name
        
    - Usage:
    
        ```
        params = {'model_name':'my-test_clf',
                  'api_key':''}

        r = requests.post('http://{system_ip}:7656/watson/delete', json=params)
        res = json.loads(r.text)
        print(res.keys())
        ```
5) watson/mlflow

    Get URL of Ml-flow endpoint.
    
    - Usage:
    
        ```
        r = requests.post('http://{system_ip}:7656/watson/mlflow')
        res = json.loads(r.text)
        print(res.keys())
        ```

## Fast Text:
Fast Text endpoint offers 4 operations.
1) fasttext/
    
    It trains new fast text clf and perform testing on test data once it is trained.
    
    - Required Params:
        
        1. train_data: must have 'text', 'label' and 'id' col
        2. test_data: must have 'text', 'label' and 'id' col
        
    - Optional Params:
        
        1. valid_data: must have 'text', 'label' and 'id' col
        2. model_name: by default it will be randomnly generated for the user.
        3. clf_params: user defined params
        4. save_model: (default=False) Save model on server or not
        5. ml_flow_params: (dict) to log the model in ML-flow
            
            should have
            ```
            experiment_name: For e.g "Text Classifier"
            run_name: for e.g "experiment v4"
            description: for e.g "using latest train data"
            ```
        
    - Usage:
    
        ```
        params = {'train_data': data[['text','label','id']].to_dict(),
                  'test_data': data[['text','label','id']].to_dict(),
                  'model_name':'my-test_clf'}

        r = requests.post('http://{system_ip}:7655/fasttext/', json=params)
        res = json.loads(r.text)
        print(res.keys())
        ```
2) fasttext/test

    It gives prediction on passed data if model was saved on server during training.
    
    - Required Params:
        
        1. test_data: must have 'text', 'label' and 'id' col
        2. model_name: Trained model name
        
    - Usage:
    
        ```
        params = {'test_data': data[['text','label','id']].to_dict(),
                  'model_name':'my-test_clf'}
                  
        r = requests.post('http://{system_ip}:7655/fasttext/test', json=params)
        res = json.loads(r.text)
        print(res.keys())
        ```
3) fasttext/delete

    Delete model from server.
    
    - Required Params:
        
        1. model_name: Trained model name
        
    - Usage:
    
        ```
        params = {'model_name':'my-test_clf'}
        
        r = requests.post('http://{system_ip}:7655/fasttext/delete', json=params)
        res = json.loads(r.text)
        print(res.keys())
        ```
4) fasttext/mlflow

    Get URL of Ml-flow endpoint.
    
    - Usage:
    
        ```
        r = requests.post('http://{system_ip}:7655/fasttext/mlflow')
        res = json.loads(r.text)
        print(res.keys())
        ```

## hDBScan:
1) hdbscan/
    
    Perform hDBScan on given data.
    
    - Required Params:
        
        1. train_data: must have 'emb' and 'id' col
        
            if 'emb' is not available, then df must have 'text' col
        
    - Optional Params:
        
        1. clf_params: user defined params    
    
    - Usage:
    
        ```
        data['emb'] = data['emb'].apply(lambda x: x.tolist())

        params = {'train_data': data[['emb' or 'text','id']].to_dict()}

        r = requests.post('http://{system_ip}:7657/hdbscan/', json=params)
        res = json.loads(r.text)
        print(res.keys())
        
        df_res = res['df_res']
        df_res = pd.DataFrame(df_res)
        ```