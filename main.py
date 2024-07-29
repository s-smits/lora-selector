import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TrainingArguments
import os
from torch.nn.functional import cosine_similarity
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from sklearn.model_selection import train_test_split
from finetune_embeddings_model import finetune_embeddings_model
from trl import SFTTrainer, SFTConfig
import pandas as pd
from datasets import Dataset


base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_base_model_and_tokenizer():
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return base_model, tokenizer
base_language_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
base_embeddings_model_name = "BAAI/bge-small-en-v1.5"

output_dir = os.curdir
print("Current output dir",output_dir)

training_data_json = os.path.join(output_dir, "training_data.json")

# Device is cuda if not mps if not cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Chosen device is
print("Device:", device)

def format_for_training(input_data):  # Used for training large language model
    training_data_df = []
    for qa_pair in input_data:
        qa_pair_text = f"###Question:{qa_pair['question']}###Answer:{qa_pair['answer']}"
        training_data_df.append({"text": qa_pair_text})
    return pd.DataFrame(training_data_df)


def train_lora_adapters(base_language_model_name, adapter_names, training_data_json):
    # Load the base language model and tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_language_model_name)

    output_dir = "./saved_adapters"
    os.makedirs(output_dir, exist_ok=True)
    
    for lora_adapter_name in adapter_names:
        print(f"Training LoRA adapter for {lora_adapter_name}")
        # Define your LoRA adapter configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "gate_proj",
                "v_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "down_proj"
            ],
        )
                
        # Load training data for the current adapter
        input_current_adapter_training_data = [data for data in training_data_json if data['subject'] == lora_adapter_name]
        print(f"Training data for {lora_adapter_name}: {input_current_adapter_training_data[:2]}")  # Print first two items

        training_data_df = format_for_training(input_current_adapter_training_data)
        
        # Split the DataFrame into training and evaluation sets
        train_df, val_df = train_test_split(training_data_df, test_size=0.1, random_state=42)
        
        print(train_df.head())
        
        # Convert DataFrames to Datasets
        train_data = Dataset.from_pandas(train_df)
        eval_data = Dataset.from_pandas(val_df)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{lora_adapter_name}",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=8,
            warmup_steps=5,
            weight_decay=0.01,
            logging_dir=f'./logs/{lora_adapter_name}',
            logging_steps=10,
        )

        # Define SFTConfig
        sft_args = SFTConfig(
            max_seq_length=512,
            dataset_text_field="text",
            **training_args.to_dict(),
        )

        # Create a fresh copy of the model for each adapter
        adapter_model = AutoModelForCausalLM.from_pretrained(base_language_model_name, torch_dtype=torch.float16).to(device)
        adapter_model = get_peft_model(adapter_model, peft_config)
        adapter_model.print_trainable_parameters()

        trainer = SFTTrainer(
            model=adapter_model,
            args=sft_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=base_tokenizer,
        )

        # Train the model
        trainer.train()
                
        # Save the entire model including the adapter
        adapter_save_path = f"./saved_adapters/lora_adapter_{lora_adapter_name}"
        trainer.model.save_pretrained(adapter_save_path)
        print(f"Saved adapter to {adapter_save_path}")

    return adapter_names, training_data_json, training_data_df


def infer_embeddings(base_embeddings_model_name, query):
    device = torch.device("mps")
    
    # Load the tokenizer
    embeddings_tokenizer = AutoTokenizer.from_pretrained(base_embeddings_model_name)
    
    # Load the model
    embeddings_model = AutoModel.from_pretrained(base_embeddings_model_name).to(device)
    embeddings_model.eval()

    # Ensure query is a string
    if not isinstance(query, str):
        raise ValueError(f"Expected query to be a string, got {type(query)}")

    # Tokenize the input query
    inputs = embeddings_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = embeddings_model(**inputs)
        # Use mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        input_query_embeddings = (sum_embeddings / sum_mask).float()
    
    return input_query_embeddings


def load_model_with_lora(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # Ensure the model is in evaluation mode
    print(f"Active adapters: {model.active_adapters}")
    print(f"LoRA config: {model.peft_config}")
    return model

def select_best_lora_adapter(base_language_model_name, training_data_json, text_row, base_model):
    print("Starting select_best_lora_adapter function")
    print("Text row:", text_row)
    
    try:
        # Generate a list of all LoRA adapters in folder "saved_adapter"
        lora_adapters = [f"lora_adapter_{name}" for name in adapter_names]
        print("LoRA adapters:", lora_adapters)

        input_query_embeddings = infer_embeddings(base_embeddings_model_name, text_row)
        print("Input query embeddings shape:", input_query_embeddings.shape)

        adapter_training_data = {}
        for adapter_name in lora_adapters:
            adapter_subject = adapter_name.split("_")[-1]  # Get the subject name
            adapter_training_data[adapter_name] = [
                item for item in training_data_json 
                if item['subject'] == adapter_subject
            ]
        
        print("Adapter training data:", {k: len(v) for k, v in adapter_training_data.items()})  # Print lengths

        best_similarity = float('-inf')
        best_lora_adapter = None

        for lora_adapter_name, adapter_training_data in adapter_training_data.items():
            print(f"Processing adapter: {lora_adapter_name}")
            if not adapter_training_data:
                print(f"No training data found for LoRA adapter: {lora_adapter_name}. Skipping this iteration.")
                continue  # Skip this iteration if training_data_json is empty

            # Batch processing for training query embeddings
            training_query_embeddings = [infer_embeddings(base_embeddings_model_name, row['question']) for row in adapter_training_data]
            training_query_embeddings = torch.stack(training_query_embeddings)  # Stack into a tensor
            print(f"Training query embeddings shape for {lora_adapter_name}:", training_query_embeddings.shape)

            # Calculate cosine similarities in a vectorized manner
            similarities = cosine_similarity(input_query_embeddings.unsqueeze(0), training_query_embeddings)
            avg_similarity = similarities.mean().item()

            print(f"Average similarity for LoRA adapter: {lora_adapter_name} is {avg_similarity}")
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_lora_adapter = lora_adapter_name
                print(f"New best LoRA adapter: {best_lora_adapter} with similarity: {best_similarity}")

        if best_lora_adapter is None:
            raise ValueError("No suitable LoRA adapter found for the given input query.")

        print(f"Final best LoRA adapter: {best_lora_adapter}")
        
        # Load only the best adapter
        adapter_path = f"./saved_adapters/lora_adapter_{best_lora_adapter}"
        if not os.path.exists(adapter_path):
            print(f"Adapter {best_lora_adapter} not found. Using base model.")
            model = base_model
        else:
            model = load_model_with_lora(base_model, adapter_path)
            print(f"Loaded best adapter model: {best_lora_adapter}")

        return best_lora_adapter, model

    except Exception as e:
        print(f"An error occurred in select_best_lora_adapter: {str(e)}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception after printing the traceback


def generate_text(model, tokenizer, messages, max_new_tokens=256):
    # Apply chat template without tokenizing
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Applied chat template:", prompt)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    print("Input shape:", inputs.input_ids.shape)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
    
    print("Output shape:", outputs.shape)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Full model output:", full_output)
    
    # Extract only the assistant's response
    assistant_response = full_output.split("You are a helpful AI assistant.")[-1].strip()
    assistant_response = assistant_response.split("<|user|>")[0].strip()
    
    return assistant_response.strip()

def user_input_inference(base_language_model_name, base_embeddings_model_name, training_data_json, adapter_names):
    base_tokenizer = AutoTokenizer.from_pretrained(base_language_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_language_model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Ensure the tokenizer is correctly configured
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = "left"
    
    print("\nEnter your question (or 'quit' to exit):")
    user_input = input().strip()
    
    while user_input.lower() != 'quit':
        best_lora_adapter, model = select_best_lora_adapter(base_language_model_name, training_data_json, user_input, base_model)
        
        print(f"Selected LoRA adapter: {best_lora_adapter}")
        
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },
            {"role": "user", "content": user_input},
        ]
        
        # Generate the output
        output = generate_text(model, base_tokenizer, messages)
        
        print("\nModel's response:")
        print(output)
        
        print("\nEnter your next question (or 'quit' to exit):")
        user_input = input().strip()

def run_llm_inference(base_language_model_name, training_data_json, training_data_df):
    base_tokenizer = AutoTokenizer.from_pretrained(base_language_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_language_model_name, torch_dtype=torch.float16, device_map="auto")
    
    results = []
    for _, row in training_data_df.iterrows():
        text_row = row['text']
        best_lora_adapter, _ = select_best_lora_adapter(base_language_model_name, training_data_json, text_row, base_model)
        
        # Load the LoRA adapter
        adapter_path = os.path.join("./saved_adapters", best_lora_adapter)
        if not os.path.exists(adapter_path):
            print(f"Adapter {best_lora_adapter} not found. Using base model.")
            model = base_model
        else:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },
            {"role": "user", "content": text_row},
        ]
        
        # Apply the chat template
        prompt = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the output
        output = generate_text(model, base_tokenizer, prompt)
        results.append(output)
    
    return results

def test_lora_functionality(base_model, tokenizer, adapter_path):
    base_output = generate_text(base_model, tokenizer, "Test prompt for base model")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_output = generate_text(lora_model, tokenizer, "Test prompt for LoRA model")
    
    print("Base model output:", base_output)
    print("LoRA model output:", lora_output)
    print("Outputs are different:", base_output != lora_output)

if __name__ == "__main__":
    try:
        training_data_json, adapter_names = finetune_embeddings_model(base_embeddings_model_name)
        print("Training data after finetune_embeddings_model:", training_data_json[:2])
        print("Adapter names after finetune_embeddings_model:", adapter_names)
        training_data_df = format_for_training(training_data_json)
        train_lora_adapters(base_language_model_name, adapter_names, training_data_json)
        
        # Add user input inference
        user_input_inference(base_language_model_name, base_embeddings_model_name, training_data_json, adapter_names)
        
        # Test LoRA functionality
        base_model, base_tokenizer = load_base_model_and_tokenizer()
        test_lora_functionality(base_model, base_tokenizer, "./saved_adapters/lora_adapter_cities")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()