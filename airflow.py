"""
Airflow Dag for the preprocessing datapipeline
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow.utils.email import send_email

# Make sure to import your function. Adjust the import path as necessary.
from src.data_download import load_data_from_gcp_and_save_as_json
from src.missing_values import naHandler
from src.duplicates import dupeRemoval
from src.resampling import resample_data
from src.label_encoder import target_label_encoder
from src.tokenize_data import tokenize_data
from src.anomalyDetect import anomalyDetect

now = datetime.now()

# Adjust the start_date to be one minute before the current time
start_date = now - timedelta(minutes=1)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'PII_Data_Detection',
    default_args=default_args,
    description='A DAG to load data from GCP and process it',
    #schedule_interval="*/2 * * * *",
    catchup=False
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
    
load_data_task = PythonOperator(
    task_id='load_data_from_gcp',
    python_callable=load_data_from_gcp_and_save_as_json,
    dag=dag,
)


handle_missing_values_task = PythonOperator(
    task_id='missing_values_removal',
    python_callable=naHandler,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

# Task to handle duplicates
remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates',
    python_callable=dupeRemoval,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)


# Task to resample the data
resample_data_task = PythonOperator(
    task_id='resample_data',
    python_callable=resample_data,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)


# Task for label encoding
label_encode_task = PythonOperator(
    task_id='label_encoder',
    python_callable=target_label_encoder,
    on_failure_callback=task_failure_alert,
    provide_context=True,
    dag=dag,
)

# Task to tokenize data
tokenize_data_task = PythonOperator(
    task_id='tokenize_data_task',
    python_callable=tokenize_data,
    on_failure_callback=task_failure_alert,
    on_success_callback=task_success_alert,
    dag=dag,
)

# Task for anomaly detection
anomaly_Detection_task = PythonOperator(
    task_id='anomaly_detect',
    python_callable=anomalyDetect,
    on_failure_callback=task_failure_alert,
    on_success_callback=task_success_alert,
    dag=dag,
)

load_data_task >> handle_missing_values_task >>remove_duplicates_task >> resample_data_task >> label_encode_task >> tokenize_data_task 
load_data_task >> anomaly_Detection_task

if __name__ == "__main__":
    dag.cli()
