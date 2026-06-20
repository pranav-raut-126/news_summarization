from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_abstractive_model(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_path,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    return tokenizer, model

def get_abstractive_summary(text, tokenizer, model, length_category="Medium"):
    # Length mapping
    params = {
        "Short": {"max": 50, "min": 10},
        "Medium": {"max": 100, "min": 30},
        "Long": {"max": 250, "min": 70}
    }.get(length_category)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=params["max"], 
        min_length=params["min"],
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Cleaning the <n> tags and extra spaces
    summary = summary.replace("<n>", " ").replace("  ", " ")
    return summary.strip()