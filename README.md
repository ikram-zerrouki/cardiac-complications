#  LLM-Based Predictor for Cardiac Surgery Complications

This project leverages **Large Language Models (LLMs)** : **Phi-2** and **Gemma-2B** fine-tuned with LoRA to predict **early post-operative complications** in cardiac surgery patients using structured clinical data.

The pipeline includes data preprocessing, augmentation with **CTGAN**, and fine-tuning of LLMs using **instruction-style prompts** to turn tabular clinical data into a question-answering task for generative models. Final predictions are evaluated with standard metrics and visualizations.

---

##  Project Summary

> "Can a language model accurately predict post-operative outcomes from patient data?"

To answer this, the pipeline:
- Prepares structured patient records into Q&A prompts
- Uses synthetic data generation (CTGAN) to enhance model training
- Fine-tunes lightweight LLMs in 4-bit precision using **PEFT (LoRA)**
- Evaluates model performance with real-world test cases

---

## 🚀 Objectives

-  Predict early (<30 days) post-operative complications after cardiac surgery
-  Generate synthetic training data using CTGAN
-  Fine-tune LLMs (Phi-2, Gemma-2B) with LoRA adapters
-  Evaluate and visualize classification performance

---

## Key Components

-  **Data Cleaning**: Encoding, normalization, and preparation of patient records
-  **Data Augmentation**: Synthetic sampling with CTGAN (SDV)
-  **Prompt Generation**: Automated generation of LLM-compatible prompts
-  **Model Fine-Tuning**: Phi-2 & Gemma 2B trained using HuggingFace + PEFT (LoRA adapters)
-  **Evaluation**: Accuracy, confusion matrix, classification report, ROC AUC
-  **Baseline**: Traditional model (Random Forest) for benchmark comparison

---
  
## Tech Stack

| Category            | Tools / Libraries                                                                 |
|---------------------|------------------------------------------------------------------------------------|
|  LLM & Tuning      | [Transformers](https://huggingface.co/transformers), [PEFT (LoRA)](https://github.com/huggingface/peft), [Phi-2](https://huggingface.co/microsoft/phi-2), [Gemma](https://huggingface.co/google/gemma-2b-it) |
|  Data Prep         | `pandas`, `scikit-learn`, `LabelEncoder`, `CTGAN` (SDV)                            |
|  Data Augmentation | [`sdv`](https://github.com/sdv-dev/SDV) for synthetic patient record generation   |
|  Evaluation        | `classification_report`, `confusion_matrix`, `ROC AUC`, `matplotlib`, `seaborn`   |

