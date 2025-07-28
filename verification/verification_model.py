from transformers import AutoTokenizer, AutoModelForSequenceClassification
from enum import Enum
import torch
import os
import math
import json

class NLIModelType(Enum):
  PRISM = 0
  BART = 1
  DEBERTA1 = 2
  DEBERTA2 = 3
  DEBERTA3 = 4

class NLIModel():
  def __init__(self, model_type: NLIModelType, model_name: str, device: str, grade_threshold: float):
    self.model_type = model_type
    self.model_name = model_name
    self.device = device
    self.grade_threshold = grade_threshold
    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
    
  def verify_step(self, premise: str, hypothesis: str) -> str:
    tokens = self.tokenizer(premise, hypothesis, truncation = True, return_tensors = "pt")
    output = self.model(tokens["input_ids"].to(self.device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment"]
    prediction = {label: pred for pred, label in zip(prediction, label_names)}
    
    return prediction["entailment"]
  
  def verify_proof(self, proof: dict) -> bool:
    avg_entailment = 0
    premise = proof["premise"]
    steps = proof["proof"]
    start = 1
    
    # for single step proofs:
    if len(proof["proof"]) == 1:
      start = 0
    else:
      premise += f'\n{steps[0]}'

    for step in steps[start:]:
      avg_entailment += self.verify_step(premise, step)
      premise += f' {step}'
      
    avg_entailment /= len(steps)
    
    if avg_entailment * 100 >= self.grade_threshold:
      return True
    
    return False
  
class VerificationModel():
  def __init__(self, models: list):
    self.models = models
    
  def verify_proof(self, proof: dict) -> dict:
    classifications = [False] * len(self.models)
    majority = math.ceil(len(self.models) / 2)
    
    for model in self.models:
      classifications[model.model_type.value] = model.verify_proof(proof)
      
    success = sum(classifications) >= majority
    
    return {
      "id": proof["id"],
      "prompt type": proof["prompt type"],
      "classifications": classifications,
      "success": success,
      "success-human": False, # human eval metrics here and below
      "clarity": 0,
      "descriptiveness": 0,
      "redundancy": 0
    }
  
  def verify_math_proof(self, proof: dict) -> dict:
    classifications = [False] * len(self.models)
    majority = math.ceil(len(self.models) / 2)

    for model in self.models:
      classifications[model.model_type.value] = model.verify_proof(proof)

    success = sum(classifications) >= majority
    
    return {
      "id": proof["premise"],
      "type": proof["type"],
      "classifications": classifications,
      "success": success
    }

  def write_result(self, path: str, result: dict):
    with open(path, "a") as f:
      f.write(f'{json.dumps(result)}\n')