from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# Charger les données
data_path = 'Combined_ESRS_Corpus_and_Glossary.xlsx'
df_corpus = pd.read_excel(data_path, sheet_name='Corpus')
df_glossary = pd.read_excel(data_path, sheet_name='Glossary')

# Préparer les données pour l’entraînement
train_data = [
    {"translation": {"en": row["Source (EN)"], "fr": row["Target (FR)"]}}
    for _, row in df_corpus.iterrows()
]

# Convertir en Dataset compatible Hugging Face
train_dataset = Dataset.from_list(train_data)

# Charger le modèle et le tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Fonction de tokenisation
def preprocess_data(examples):
    inputs = tokenizer(examples['translation']['en'], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples['translation']['fr'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = targets['input_ids']
    return inputs

# Pré-traiter le dataset
train_dataset = train_dataset.map(preprocess_data, batched=True)

# Configurer les arguments d’entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

# Initialiser l’entraîneur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Lancer l’entraînement
trainer.train()

# Sauvegarder le modèle ajusté
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

from transformers import MarianMTModel, MarianTokenizer

# Charger le modèle affiné
model_path = "./fine_tuned_model"
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Traduire une phrase
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Exemple
text = "The company's sustainability report includes multiple sections."
translation = translate_text(text)
print("Translation:", translation)

from transformers import MarianMTModel, MarianTokenizer

# Charger le modèle affiné
model_path = "./fine_tuned_model"
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Traduire une phrase
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Exemple
text = "The company's sustainability report includes multiple sections."
translation = translate_text(text)
print("Translation:", translation)
