import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import re

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
    input_text = st.text_area("Enter text to mask", height=200)
    if st.button("Mask Text"):
        if input_text:
            masked_text = mask_tokens_with_predictions(input_text, model, tokenizer, id2label)
            # Postprocess the masked text
            postprocessed_text = preprocess_text(masked_text)
            st.text_area("Postprocessed Text:", postprocessed_text, height=200)
        else:
            st.write("Please enter some text before masking.")  



