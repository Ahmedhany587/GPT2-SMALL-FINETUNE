## Fine-Tuned GPT2-small on all william shakespeare's work
Built a generatinve model to generate text related to william shakespeare writing style.
The dataset consists of plays,literature,and poems he wrote.


---


This project involves the fine-tuning of a GPT-2 language model using the Hugging Face Transformers library. The primary goal is to leverage the power of GPT-2 for a specific task using a provided dataset. Additionally, the project incorporates Peft with loRa for which hepled to train only 20% of the model parameters .

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Package Installation](#package-installation)
  - [Reading Data](#reading-data)
  - [Downloading GPT-2 Weights and Tokenizer](#downloading-gpt-2-weights-and-tokenizer)
  - [Lora Configuration](#lora-configuration)
- [Data Preparation](#data-preparation)
  - [Splitting Data](#splitting-data)
  - [Tokenizing Data](#tokenizing-data)
- [GPT-2 Fine-Tuning](#gpt-2-fine-tuning)
  - [Training Setup](#training-setup)
  - [Custom Collation Function](#custom-collation-function)
  - [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Notes](#notes)

## Introduction

This project aims to fine-tune the GPT-2 language model for a specific task using a provided dataset. The fine-tuning process involves loading a pre-trained GPT-2 model, configuring Lora local re-attention for enhanced performance, and training the model on the dataset.

## Setup

### Package Installation

Ensure the required Python packages are installed by running the following commands:

```bash
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
!pip install accelerate -U
!pip install transformers==4.33.2 datasets torch
```

### Reading Data

Mount Google Drive to access project files and set the base directory:

```python
from google.colab import drive
drive.mount('/content/drive')
base_dir = "/content/drive/MyDrive/GPT2-finetune"
```

...

[Continue with the rest of the setup]

## Data Preparation

### Splitting Data

The dataset is loaded and split into training and validation subsets. The subsets are then saved to separate files.

### Tokenizing Data

The GPT-2 tokenizer is applied to tokenize the text data, and the tokenized datasets are saved to disk for future use.

## GPT-2 Fine-Tuning

### Training Setup

Training arguments are set up using the `TrainingArguments` class from the `transformers` library.

### Custom Collation Function

A custom collation function is defined to handle the batching of data during training.

### Training Process

The model is trained using the `Trainer` class, which includes the model, training arguments, datasets, and the custom data collator.

## Evaluation

After training, the model is evaluated using the validation dataset, and the evaluation results are stored.






# Load the model
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
```

...

## Generating Text

Generate text using the fine-tuned GPT-2 model:

```python
# Generate some text
input_text = "On Sept. 24, 2021, 18-year-old Joshua Bennett of Etobicoke was stabbed in the Paulander Drive area in Kitchener around 4:30 a.m."

input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Create an attention mask
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

output = model.generate(input_ids,
                        max_length=100,
                        min_length=70,
                        do_sample=True,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=2.0,
                        #top_p=0.95,
                        #top_k=50
                        )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
```

...

[Continue with the rest of the README]

## Notes

...
