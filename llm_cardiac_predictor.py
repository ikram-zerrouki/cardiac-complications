import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Chargement des données (fictives) ===
df = pd.read_csv("sample_patient_data.csv")  # ⚠️ Remplacez par vos données fictives anonymisées

# === Nettoyage et prétraitement ===
columns_to_drop = ["Horodateur", "nom prénom", "ATCD CHIRURGICAUX CARDIAQUES ", "CHIRURGIE CARDIAQUE PRECEDENTE: "]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

target_col = "Complications post-opératoires précoces (< 30 jours)"
df[target_col] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['oui', 'yes', '1', 'true'] else 0)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != "numero du dossier" and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

valve_columns = ["ANATOMIE DE LA VALVE CHIRURGIE", "ANATOMIE DE LA VALVE ECHO"]
for col in valve_columns:
    if col in df.columns:
        df[col] = df[col].str.upper().str.strip()
        df[col] = df[col].apply(lambda x: 'BICUSPIDE' if isinstance(x, str) and 'BICU' in x else
                                         'TRICUSPIDE' if isinstance(x, str) and 'TRICU' in x else 'INCONNU')

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# === Séparation des données ===
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

train_df = X_train.copy(); train_df[target_col] = y_train
test_df = X_test.copy(); test_df[target_col] = y_test

# === Augmentation avec CTGAN ===
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)
ctgan = CTGANSynthesizer(metadata, enforce_rounding=True)
ctgan.fit(train_df)
synthetic_data = ctgan.sample(len(train_df))

augmented_df = pd.concat([train_df, synthetic_data], ignore_index=True)
full_dataset = pd.concat([augmented_df, test_df], ignore_index=True)

# === Création des prompts ===
def create_prompt(row, df_source):
    prompt = "Vous êtes un assistant médical expert.\nVoici les données d’un patient hospitalisé pour une chirurgie cardiaque :\n\n"
    for col in df_source.columns:
        if col not in [target_col, 'prompt', 'response']:
            prompt += f"- {col} : {row[col]}\n"
    prompt += (
        "\nD'après les informations ci-dessus, ce patient a-t-il présenté "
        "des complications post-opératoires précoces (dans les 30 jours suivant l’intervention) ?\n"
        "Répondez uniquement par : 'oui' ou 'non'."
    )
    return prompt

full_dataset["prompt"] = full_dataset.apply(lambda row: create_prompt(row, full_dataset), axis=1)
full_dataset["response"] = full_dataset[target_col].apply(lambda x: "oui" if x == 1 else "non")

train_df, test_df = train_test_split(full_dataset, test_size=0.2, stratify=full_dataset[target_col], random_state=42)

# === Préparation pour le modèle ===
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant_config)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, peft_config)
model.train()

train_df["prompt"] = train_df["prompt"]
train_df["response"] = train_df["response"]

dataset = Dataset.from_pandas(train_df[["prompt", "response"]])

def tokenize(example):
    outputs = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./phi2-lora-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("phi2-lora-adapter")
tokenizer.save_pretrained("phi2-lora-adapter")

# === Évaluation du modèle ===
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant_config)
model = PeftModel.from_pretrained(base_model, "phi2-lora-adapter")
model.eval()

results = []
y_true = []
y_pred = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    prompt = (
        f"### Question:\n{row['prompt']}\n"
        "### Instruction: Répondez uniquement par 'oui' ou 'non'.\n"
        "### Réponse:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=20, do_sample=False, temperature=0.7, top_p=0.9, num_beams=3
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("### Réponse:")[-1].strip().lower()
    pred = "oui" if "oui" in answer else "non" if "non" in answer else "?"
    expected = row["response"].strip().lower()
    results.append((row["prompt"], pred, expected))
    y_true.append(expected)
    y_pred.append(pred)

print("\n🎯 Rapport de classification :")
print(classification_report(y_true, y_pred, labels=["oui", "non", "?"], zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=["oui", "non", "?"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["oui", "non", "?"])
disp.plot(cmap="Blues")
plt.title("Matrice de confusion")
plt.show()
