import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import re
import logging
import time

def predict(test_mapped_path, trained_model_path):
    print("Received test data path at", test_mapped_path)
    logging.info(f"Received test data path at {test_mapped_path}")
    test_mapped = load_from_disk(test_mapped_path)

    input_ids = torch.tensor(test_mapped["input_ids"], dtype=torch.long)
    # token_type_ids = torch.tensor(test_mapped["token_type_ids"], dtype=torch.long)
    attention_mask = torch.tensor(test_mapped["attention_mask"], dtype=torch.long)
    og_labels = test_mapped['labels']
    logging.info(f'fetched input_id, attention_mask, og_labels')

    model = AutoModelForTokenClassification.from_pretrained(trained_model_path)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = outputs.logits
    pred_softmax = torch.nn.functional.softmax(predictions, dim=-1).detach().numpy()
    id2label = model.config.id2label
    o_index = {v: k for k, v in id2label.items()}['O']

    preds_without_O = pred_softmax[:, :, :o_index].argmax(-1)
    O_preds = pred_softmax[:, :, o_index]
    threshold = 0.9
    preds_final = np.where(O_preds < threshold, preds_without_O, predictions.argmax(-1))

    flat_og_labels = [label for sublist in og_labels for label in sublist]
    flat_pred_labels = [label for sublist in preds_final for label in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(flat_og_labels, flat_pred_labels, average='weighted')

    # with mlflow.start_run(run_id=run_id):
    #     mlflow.log_metrics({
    #         "Precision": precision,
    #         "Recall": recall,
    #         "F1 Score": f1
    #     })

    return precision, recall, f1


# Function to mask tokens with predictions
def mask_tokens_with_predictions(text, model, tokenizer, id2label):
    # Tokenize the input text
    tokenized_input = tokenizer(text, return_offsets_mapping=True, padding="max_length", truncation=True)

    # Convert the tokenized input to tensors
    input_ids = torch.tensor(tokenized_input['input_ids']).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(tokenized_input['attention_mask']).unsqueeze(0).to(model.device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Extract predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    # Reconstruct text with predicted labels
    reconstructed_text = ''
    for i, (token_id, offset_mapping) in enumerate(zip(tokenized_input['input_ids'], tokenized_input['offset_mapping'])):
        start_idx, end_idx = offset_mapping
        if start_idx == 0 and end_idx == 0:
            continue  # Skip special tokens like [CLS] and [SEP]

        # Get the predicted label for the current token
        predicted_label = id2label[predictions[i]]

        # Convert token ID to token string
        token = tokenizer.convert_ids_to_tokens(token_id)

        # Remove underscores from the token
        token = token.replace('▁', '')

        # Mask the token in the reconstructed text if it's not a special token
        reconstructed_text += token if predicted_label == 'O' else f'<{predicted_label}>'

        # Add trailing whitespace if present
        if i < len(tokenized_input['input_ids']) - 1 and tokenized_input['attention_mask'][i+1] == 1:
            reconstructed_text += ' '

    return reconstructed_text


def preprocess_text(text):
    # Define base token types without the B- or I- prefixes
    base_token_types = [
        'EMAIL',
        'ID_NUM',
        'NAME_STUDENT',
        'PHONE_NUM',
        'STREET_ADDRESS',
        'URL_PERSONAL',
        'USERNAME'
    ]
    
    # For each base token type, create a pattern that matches both its B- and I- prefixed forms
    patterns = {token: re.compile(rf'<[BI]-{token}>') for token in base_token_types}
    
    # Replace all occurrences of each pattern with a single instance of the simplified token,
    # and then remove all additional occurrences.
    for token_type, pattern in patterns.items():
        if pattern.search(text):
            # Replace the first occurrence with the simplified token
            text = pattern.sub(f'<{token_type}>', text, 1)
            # Remove all other occurrences of this token type
            text = pattern.sub('', text)
    
    # Remove any extra spaces that might have been introduced
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def serve_model(**kwargs):
    ti= kwargs['ti']
    _,_,_,_,model_path = ti.xcom_pull(task_ids='inference')  

    # id2label = {i: label for i, label in enumerate(model.config.id2label)}
    id2label = {
        0: 'B-EMAIL',
        1: 'B-ID_NUM',
        2: 'B-NAME_STUDENT',
        3: 'B-PHONE_NUM',
        4: 'B-STREET_ADDRESS',
        5: 'B-URL_PERSONAL',
        6: 'B-USERNAME',
        7: 'I-ID_NUM',
        8: 'I-NAME_STUDENT',
        9: 'I-PHONE_NUM',
        10: 'I-STREET_ADDRESS',
        11: 'I-URL_PERSONAL',
        12: 'O'
    }
    # Load the model and tokenizer together
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    st.title("Mask your text and get free from PII")
    input_text = st.text_area("Enter text to mask", height = 200)
    logging.info('Model served at streamlit')
    
    # create csv table for running time execution
    csv_header = ['timeNow', 'execution_time']
    latency_table = os.path.join(PROJECT_DIR,'data','latency_metrics.csv')
    # If latency_metrics.csv file does not exist, create it and add the header.
    if not os.path.exists(latency_table):
        with open(latency_table, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
    
    if st.button("Mask Text"):
        if input_text:
            start_time = time.time() # code when it started executing
            masked_text = mask_tokens_with_predictions(input_text, model, tokenizer, id2label)
            # Postprocess the masked text
            postprocessed_text = preprocess_text(masked_text)
            st.text_area("Postprocessed Text:", postprocessed_text, height=200)
            execution_time = time.time() - start_time # execution time = start - execution time
            to_append = [start_time, execution_time] # append latency metrics to the latency

            # Append latency metrics to latency_metrics.csv
            with open(latency_table, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(to_append)
        else:
            st.write("Please enter some text before masking.")  