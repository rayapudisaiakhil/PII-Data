
from predict import predict

def model_version(**kwargs):
    ti= kwargs['ti']
    _,_,_,_,latest_version_model = ti.xcom_pull(task_ids='inference')  
    
    best_model_path,_=ti.xcom_pull(task_ids='train_new_model') 
    
    _,_,test_mapped=ti.xcom_pull(task_ids='tokenize_data') 
    
    precision_o, recall_o, f1_o = predict(test_mapped_path = test_mapped, trained_model_path = latest_version_model)
    precision_n, recall_n, f1_n = predict(test_mapped_path = test_mapped, trained_model_path = best_model_path)

    if recall_o < recall_n and f1_o < f1_n:
        version_retrained_model = True
    else:
        version_retrained_model = False
        
    return version_retrained_model
    