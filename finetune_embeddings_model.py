from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import json

def fetch_training_data_json(): # Used for training embeddings model with question and subject pairs
    with open('training_data.json', 'r') as f:
        data = json.load(f)
    return data["text"]

def finetune_embeddings_model(base_embeddings_model_name, num_epochs=1, batch_size=8, lr=1e-3):
    # cuda then mps then cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    base_model = AutoModel.from_pretrained(base_embeddings_model_name).to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(base_embeddings_model_name)

    training_data_json = fetch_training_data_json()
    print("Original training data:", training_data_json[:2])  # Print first two items
    
    unique_subjects = sorted(set(item['subject'] for item in training_data_json))
    adapter_names = unique_subjects
    num_unique_subjects = len(unique_subjects)
    
    print("Unique subjects:", unique_subjects)
    print("Adapter names:", adapter_names)

    # Create subject_to_idx mapping
    subject_to_idx = {subject: idx for idx, subject in enumerate(unique_subjects)}

    classification_head = nn.Linear(base_model.config.hidden_size, num_unique_subjects).to(device)

    optimizer = torch.optim.AdamW(list(base_model.parameters()) + list(classification_head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_data, val_data = train_test_split(training_data_json, test_size=0.2, random_state=42, stratify=[item['subject'] for item in training_data_json])

    def process_data(data):
        input_ids, attention_masks, labels = [], [], []
        for item in data:
            encoding = base_tokenizer(item['question'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids.append(encoding['input_ids'].squeeze(0))
            attention_masks.append(encoding['attention_mask'].squeeze(0))
            labels.append(subject_to_idx[item['subject']])  # Now using the defined subject_to_idx

        input_ids = pad_sequence(input_ids, batch_first=True).to(device)
        attention_masks = pad_sequence(attention_masks, batch_first=True).to(device)
        labels = torch.tensor(labels).to(device)

        return {"input_ids": input_ids, "attention_mask": attention_masks}, labels

    train_inputs, train_labels = process_data(train_data)
    val_inputs, val_labels = process_data(val_data)
    train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_mcc = 0
    
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}")
        base_model.train()
        classification_head.train()
        
        for batch in train_dataloader:
            inputs, labels = batch[:-1], batch[-1]
            inputs = {key: value.to(device) for key, value in zip(['input_ids', 'attention_mask'], inputs)}
            outputs = base_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            logits = classification_head(embeddings)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        base_model.eval()
        classification_head.eval()
        
        total_val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch[:-1], batch[-1]
                inputs = {key: value.to(device) for key, value in zip(['input_ids', 'attention_mask'], inputs)}
                outputs = base_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                logits = classification_head(embeddings)
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        mcc = matthews_corrcoef(all_labels, all_preds)

        print('Validation loss:', avg_val_loss, 'Validation MCC:', mcc)
        
        if mcc > best_mcc:
            best_mcc = mcc
            
            # Save the entire model, not just the state dict
            torch.save(base_model, "finetuned_embeddings_model.pt")

    print(adapter_names)
    print("Final training data:", training_data_json[:5])  # Print first five items
    print("Adapter names:", adapter_names)
    return training_data_json, adapter_names