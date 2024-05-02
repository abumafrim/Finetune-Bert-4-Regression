import sys
import math
import pandas as pd
from datasets import load_dataset
from torch.nn.functional import mse_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

class RegressionTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs[0][:, 0]
    loss = mse_loss(logits, labels.float())
    return (loss, outputs) if return_outputs else loss

def convert_labels(label):
  '''
    Converting the Afrisenti labels to a regression problem
    
    Current labels:
      0 - positive
      1 - neutral
      2 - negative
    
    New labels:
      1 - positive
      0 - neutral
      -1 - negative
  '''
  if label == 2:
    return -1
  elif label == 1:
    return 0
  elif label == 0:
    return 1
  sys.exit('invalid label code')

def preprocess_function(examples):
  label = convert_labels(examples["label"])
  examples = tokenizer(examples["tweet"], truncation=True, padding="max_length", max_length=256)
  examples["label"] = float(label)
  return examples

def compute_metrics_for_regression(eval_pred):
  logits, labels = eval_pred
  labels = labels.reshape(-1, 1)
  
  mse = mean_squared_error(labels, logits)
  mae = mean_absolute_error(labels, logits)
  r2 = r2_score(labels, logits)
  single_squared_errors = ((logits - labels).flatten()**2).tolist()
  accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
  
  return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

EPOCHS = 20
BATCH_SIZE = 32
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
BASE_MODEL = 'google-bert/bert-base-multilingual-cased'

training_args = TrainingArguments(
  output_dir="models/hausa-mbert-fine-tuned-regression",
  learning_rate=LEARNING_RATE,
  per_device_train_batch_size=BATCH_SIZE,
  per_device_eval_batch_size=BATCH_SIZE,
  num_train_epochs=EPOCHS,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  save_total_limit=2,
  metric_for_best_model="accuracy",
  load_best_model_at_end=True,
  weight_decay=0.01,
  report_to="tensorboard"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

ds = load_dataset("HausaNLP/AfriSenti-Twitter", "hau")
raw_train_ds, raw_val_ds, raw_test_ds = ds['train'], ds['validation'], ds['test']

# using only 70% of the training data for training
raw_train_ds = raw_train_ds.train_test_split(test_size=0.3)
raw_train_ds, emotion_ds = raw_train_ds['train'], raw_train_ds['test']

ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds, "emotion": emotion_ds}

for split in ds:
  ds[split] = ds[split].map(preprocess_function, remove_columns=["tweet"])

trainer = RegressionTrainer(
  model=model,
  args=training_args,
  train_dataset=ds["train"],
  eval_dataset=ds["validation"],
  compute_metrics=compute_metrics_for_regression,
)

trainer.train()

trainer.eval_dataset=ds["test"]
trainer.evaluate()

nb_batches = math.ceil(len(raw_test_ds)/BATCH_SIZE)
y_preds = []

for i in range(nb_batches):
  input_texts = raw_test_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["tweet"]
  encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
  y_preds += model(**encoded).logits.reshape(-1).tolist()

pd.set_option('display.max_rows', 500)
df = pd.DataFrame([raw_test_ds["tweet"], ds['test']['label'], y_preds], ["Text", "Label", "Prediction"]).T

df.to_csv('test_predictions.csv', index=False)

nb_batches = math.ceil(len(emotion_ds)/BATCH_SIZE)
y_preds = []

for i in range(nb_batches):
  input_texts = emotion_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["tweet"]
  encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
  y_preds += model(**encoded).logits.reshape(-1).tolist()

df = pd.DataFrame([emotion_ds["tweet"], ds['emotion']['label'], y_preds], ["Text", "Label", "Prediction"]).T

df.to_csv('emotion_predictions.csv', index=False)