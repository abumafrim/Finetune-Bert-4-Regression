import os
import sys
import math
import torch
import argparse
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

parser = argparse.ArgumentParser(description='Fine-tune a regression model on the AfriSenti dataset')
parser.add_argument('--base_model', type=str, default='google-bert/bert-base-multilingual-cased', help='Base model to use')
parser.add_argument('--output_dir', type=str, default='models/hausa-mbert-fine-tuned-regression', help='Output directory')
parser.add_argument('--dataset', type=str, default='HausaNLP/AfriSenti-Twitter', help='Dataset to use')
parser.add_argument('--language', type=str, default='hau', help='Language in the dataset to use')
parser.add_argument('--use_all_training_data', action='store_true', help='Use all the training data for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--max_length', type=int, default=256, help='Maximum length of the input text')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Evaluation strategy')
parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy')
parser.add_argument('--save_total_limit', type=int, default=2, help='Number of models to save')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--metric_for_best_model', type=str, default='accuracy', help='Metric for the best model')
parser.add_argument('--report_to', type=str, default='tensorboard', help='Report to')
parser.add_argument('--held_out_percentage', type=float, default=0.3, help='Percentage of the training data to hold out')
parser.add_argument('--load_best_model_at_end', action='store_true', help='Load the best model at the end of training')
parser.add_argument('--do_train', action='store_true', help='Train the model')
parser.add_argument('--do_eval', action='store_true', help='Evaluate the model')
parser.add_argument('--do_predict', action='store_true', help='Predict using the model')
parser.add_argument('--do_predict_held_out', action='store_true', help='Predict using the held out data')
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
LEARNING_RATE = args.learning_rate
BASE_MODEL = args.base_model
OUTPUT_DIR = args.output_dir

training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  learning_rate=LEARNING_RATE,
  per_device_train_batch_size=BATCH_SIZE,
  per_device_eval_batch_size=BATCH_SIZE,
  num_train_epochs=EPOCHS,
  evaluation_strategy=args.evaluation_strategy,
  save_strategy=args.save_strategy,
  save_total_limit=args.save_total_limit,
  metric_for_best_model=args.metric_for_best_model,
  load_best_model_at_end=str(args.load_best_model_at_end),
  weight_decay=args.weight_decay,
  report_to=args.report_to
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

ds = load_dataset(args.dataset, args.language)
raw_train_ds, raw_val_ds, raw_test_ds = ds['train'], ds['validation'], ds['test']

ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}

if not args.use_all_training_data:
  raw_train_ds = raw_train_ds.train_test_split(test_size=args.held_out_percentage, seed=args.seed)
  raw_train_ds, hold_out_ds = raw_train_ds['train'], raw_train_ds['test']
  ds["train"], ds["hold_out"] = raw_train_ds, hold_out_ds

for split in ds:
  ds[split] = ds[split].map(preprocess_function, remove_columns=["tweet"])

trainer = RegressionTrainer(
  model=model,
  args=training_args,
  train_dataset=ds["train"],
  eval_dataset=ds["validation"],
  compute_metrics=compute_metrics_for_regression,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.do_train:
  trainer.train()

if args.do_eval:
  trainer.eval_dataset=ds["test"]
  trainer.evaluate()

if args.do_predict:
  nb_batches = math.ceil(len(raw_test_ds)/BATCH_SIZE)
  y_preds = []

  for i in range(nb_batches):
    input_texts = raw_test_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["tweet"]
    encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt").to(device)
    y_preds += model(**encoded).logits.reshape(-1).tolist()

  df = pd.DataFrame([raw_test_ds["tweet"], ds['test']['label'], y_preds], ["Text", "Label", "Prediction"]).T

  df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

if args.do_predict_held_out:
  if not args.use_all_training_data:
    nb_batches = math.ceil(len(hold_out_ds)/BATCH_SIZE)
    y_preds = []

    for i in range(nb_batches):
      input_texts = hold_out_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["tweet"]
      encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
      y_preds += model(**encoded).logits.reshape(-1).tolist()

    df = pd.DataFrame([hold_out_ds["tweet"], ds['hold_out']['label'], y_preds], ["Text", "Label", "Prediction"]).T

    df.to_csv(os.path.join(OUTPUT_DIR, 'held_out_predictions.csv'), index=False)
  else:
    sys.exit('Cannot predict on held out data when all training data is used for training')