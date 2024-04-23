from airflow import DAG
from airflow.operators.python_operator import PythonOperator
# from airflow.operators.bash_operator import BashOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.utils.email import send_email

# Optional: Define a DummyOperator or BranchPythonOperator for conditional branching
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator

# Import your functions from their modules

from src.data_slicing import load_data_from_gcp_and_save_as_json
from src.anomalyDetect import anomalyDetect
from src.missing_values import naHandler
from src.duplicates import dupeRemoval
from src.resampling import resample_data
from src.label_encoder import target_label_encoder
from src.tokenize_data import tokenize_data
# from src.predict_data import predict
from src.inference import evaluate_model
from src.train import train_model
from src.model_performance_check import check_performance
from src.model_versioning import model_version
from src.upload_model_gcp import upload_model_to_gcp
from src.serve import serve_model

import json
import os

send_alert_to='gvk7663@gmail.com'
data_dir =os.getcwd()
# os.path.join(PROJECT_DIR, "dags", "processed") #sere to save the batched data
num_data_points = 10 #number of data points to fetch
# cumulative = True #make this true if new batc has to be retrained with old data
#GCLOUD
bucket_name='pii_train_data'
projectid='piidatadetection'
PROJECT_DIR = os.getcwd()
KEY_PATH=os.path.join(PROJECT_DIR, "config", "key.json")
now = datetime.now()
# Adjust the start_date to be one minute before the current time
start_date = now - timedelta(minutes=1)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    # 'start_date': now, # Using the current time for start_date
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,  # Turn off default email on failure
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'test_PII_Data_Detection',
    default_args=default_args,
    description='A DAG to load data from GCP and process it',
    schedule_interval="*/5 * * * *",# Manually triggered or specify a schedule
    catchup=False,
)

# Define callback functions for email notifications
def task_success_alert(context):
    subject = f"Airflow Task Success: {context['task_instance'].task_id}"
    html_content = f"""
    <h3>Task Success</h3>
    DAG: {context['task_instance'].dag_id}<br>
    Task: {context['task_instance'].task_id}<br>
    Execution Time: {context['execution_date']}<br>
    """
    send_email(to='gvk7663@gmail.com', subject=subject, html_content=html_content)

def task_failure_alert(context):
    subject = f"Airflow Task Failure: {context['task_instance'].task_id}"
    html_content = f"""
    <h3>Task Failure</h3>
    DAG: {context['task_instance'].dag_id}<br>
    Task: {context['task_instance'].task_id}<br>
    Execution Time: {context['execution_date']}<br>
    Log URL: {context['task_instance'].log_url}<br>
    """
    send_email(to='gvk7663@gmail.com', subject=subject, html_content=html_content)

# Task definitions
# User to set keys to keyPath location 
# load_data_task = PythonOperator(
#     task_id='load_data_from_gcp',
#     python_callable=load_data_from_gcp_and_save_as_json,
#     on_failure_callback=task_failure_alert,
#     op_kwargs = {'keyPath': '/home/vineshgvk/PII-Data/key.json'},
#     dag=dag,
# )
# load_data_task = PythonOperator(
#     task_id='load_data_from_gcp',
#     python_callable=load_data_from_gcp_and_save_as_json,
#     on_failure_callback=task_failure_alert,
#     dag=dag,
# )


data_slicing_batches_task=PythonOperator(
    task_id='data_slicing_batches_task',
    python_callable=load_data_from_gcp_and_save_as_json,
    op_kwargs={'data_dir': data_dir, 'num_data_points': num_data_points,'bucket_name':bucket_name,
               'KEY_PATH':KEY_PATH},
    on_failure_callback=task_failure_alert,
    dag=dag,
)

# Task for anomaly detection
anomaly_detect_task = PythonOperator(
    task_id='anomaly_detect',
    python_callable=anomalyDetect,
    on_failure_callback=task_failure_alert,
    dag=dag,
)


def check_anomalies_and_alert(**kwargs):
    ti = kwargs['ti']
    anomaly_results = ti.xcom_pull(task_ids='anomaly_detect')
    issues = anomaly_results.get('issues', {})

    if issues:  # Check if there are any issues reported
        subject = "Anomaly Detected in Dataset"
        html_content = "<h3>Detected Anomalies:</h3>"
        for issue, description in issues.items():
            html_content += f"<p><strong>{issue}:</strong> {description}</p>"

        # Add more stats or details if necessary
        stats = anomaly_results.get('stats', {})
        html_content += "<h4>Statistics:</h4>"
        for stat, value in stats.items():
            html_content += f"<p><strong>{stat}:</strong> {value}</p>"

        send_email(to=send_alert_to, subject=subject, html_content=html_content)
    else:
        return True

# Configure the above function as a task in your DAG
alert_task = PythonOperator(
    task_id='send_alert_if_anomalies',
    python_callable=check_anomalies_and_alert,
    provide_context=True,
    dag=dag,
)

handle_missing_values_task = PythonOperator(
    task_id='missing_values_removal',
    python_callable=naHandler,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates',
    python_callable=dupeRemoval,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

resample_data_task = PythonOperator(
    task_id='resample_data',
    python_callable=resample_data,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

label_encode_task = PythonOperator(
    task_id='label_encoder',
    python_callable=target_label_encoder,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

tokenize_data_task = PythonOperator(
    task_id='tokenize_data',
    python_callable=tokenize_data,
    on_failure_callback=task_failure_alert,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='inference',
    python_callable=evaluate_model,
    on_failure_callback=task_failure_alert,
    op_kwargs={'KEY_PATH':KEY_PATH,'projectid': projectid,'bucket_name':bucket_name},
    provide_context=True,
    dag=dag,
)


model_performance_evaluation = PythonOperator(
    task_id='model_performance_evaluation',
    python_callable=check_performance,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

# Define branching logic
def decide_which_path(**kwargs):
    ti = kwargs['ti']
    retrain = ti.xcom_pull(task_ids='model_performance_evaluation', key='retrain')
    
    if retrain:
        return 'train_new_model'
    else:
        return 'end_of_flow'

branching = BranchPythonOperator(
    task_id='branching',
    python_callable=decide_which_path,
    dag=dag,
)


train_new_model_task = PythonOperator(
    task_id='train_new_model',
    python_callable=train_model,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

end_of_flow_task = DummyOperator(
    task_id='end_of_flow',
    dag=dag,
)


model_version_task = PythonOperator(
    task_id='model_version',
    python_callable=model_version,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

# update_model_directory_task = PythonOperator(
#     task_id='update_model_directory',
#     python_callable=update_model_directory,
#     on_failure_callback=task_failure_alert,
#     provide_context=True,
#     dag=dag,
# )
def decide_version_upload(**kwargs):
    ti = kwargs['ti']
    version_retrained_model = ti.xcom_pull(task_ids='model_version', key='version_retrained_model')
    if version_retrained_model():  # Use the function here
        return 'upload_model_to_gcp'
    else:
        return 'end_of_flow'
    
branching_version_compare = BranchPythonOperator(
    task_id='branching_version_compare',
    python_callable=decide_version_upload,
    dag=dag,
)


upload_model_to_gcp_task = PythonOperator(
    task_id='upload_model_to_gcp',
    python_callable=upload_model_to_gcp,
    op_kwargs={'KEY_PATH':KEY_PATH,'projectid': projectid,'bucket_name':bucket_name},
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

# serve_model_task = PythonOperator(
#     task_id='model_serving_streamlit',
#     python_callable=serve_model,
#     on_failure_callback=task_failure_alert,
#     provide_context=True,
#     dag=dag,
# )

streamlit_path=os.path.join(PROJECT_DIR, "dags", "src","serve.py")

serve_model_task = BashOperator(
    task_id='model_serving_streamlit',
    bash_command=f'streamlit run {streamlit_path}',
    dag=dag,
)

# Task dependencies
data_slicing_batches_task >> anomaly_detect_task >> alert_task

data_slicing_batches_task >> handle_missing_values_task >> remove_duplicates_task >> resample_data_task >> label_encode_task >> tokenize_data_task >> predict_task >> model_performance_evaluation >> branching
# if retrain is true
branching >> train_new_model_task
# if retraining is false
branching >> end_of_flow_task

train_new_model_task >> model_version_task >> branching_version_compare
# if compare_models is true
branching_version_compare >> upload_model_to_gcp_task >> serve_model_task
# if compare_models is false
branching_version_compare >> end_of_flow_task

'''
load_data_task runs first.
After load_data_task completes, handle_missing_values_task runs.
After handle_missing_values_task completes, remove_duplicates_task runs.
After remove_duplicates_task completes, resample_data_task runs.
After resample_data_task completes, label_encode_task runs.
After label_encode_task completes, tokenize_data_task runs.
After tokenize_data_task completes, predict_task runs.
After predict_task completes, evaluate_model_task runs.
After evaluate_model_task completes, the branching task runs. This task checks the output of evaluate_model_task.
If evaluate_model_task returns True, train_new_model_task runs next.
If evaluate_model_task returns False, the DAG ends at end_of_flow_task.
After train_new_model_task completes, compare_models_task runs.
After compare_models_task completes, the branching_after_compare task runs. This task checks the output of compare_models_task.
If compare_models_task returns True, upload_model_to_gcp_task runs next, followed by serve_model_task.
If compare_models_task returns False, the DAG ends at end_of_flow_task
'''

if __name__ == "__main__":
    dag.cli()
    