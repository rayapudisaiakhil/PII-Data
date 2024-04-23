def check_performance(**kwargs):
    ti = kwargs['ti']
    # Pull the results tuple from the inference task
    results = ti.xcom_pull(task_ids='inference', key='return_value')

    if results:
        precision, recall, f1, metrics_model_decay = results

        # Determine if retraining is necessary based on recall and f1 score
        if recall < 0.9 and f1 < 0.8:
            retrain = True
        else:
            retrain = False

        # Optionally, log these metrics or use them further
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Model Decay Metrics: {metrics_model_decay}")

        # Push this decision to XCom for other tasks
        ti.xcom_push(key='retrain', value=retrain)
        return retrain
    else:
        raise ValueError("No data received from 'inference' task")