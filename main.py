from Code.utils.utils import parse_jsonl, build_prompts, verify_latex
from Code.generation.generation_model import GenModelType, GenModel
from Code.verification.verification_model import NLIModelType, NLIModel, VerificationModel
from datetime import datetime

import torch
import json
import os
import sys

SYSTEM_PROMPT = '''
You are a mathematician with an excellent understanding of undergraduate and high-school level mathematics.
You write clear, concise and correct informal proofs in English that clearly explain the logic behind each step.

INSTRUCTIONS:
  - When you write proofs, each sentence is on a separate line.
  - When you write proofs, you do not embolden, italicize or underline any text.
  - When you are writing proofs, incorporate correct LaTeX code to represent mathematical notation.
  - When writing informal proofs, start by defining all the variables you will use within the proof.
  - When writing informal proofs, state whether you use direct proof, proof by contradiction, proof by contraposition, proof by mathematical induction or proof by exhaustion.
  - If you use a combination of the proof types above, list the approaches you have used.

When writing your proof, structure your proof using the following template:

Variable definitions:
<Your definitions here>

Proof type(s):
<Your approaches here>

Proof:
<Your proof here>
QED
'''
LATEX_TEMPLATE = rf'''
\documentclass{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}

\begin{{document}}
%s
\end{{document}}
'''
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
GRADE_THRESHOLD = 50

nli_model_names = ["Jaehun/PrismNLI-0.4B", "facebook/bart-large-mnli", "MoritzLaurer/DeBERTa-v3-base-mnli", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]
nli_models = [NLIModel(model_type, nli_model_names[model_type.value], DEVICE, GRADE_THRESHOLD) for model_type in NLIModelType]
verification_model = VerificationModel(nli_models)

base_url, api_key, folder_path, proofs_path, attempt = sys.argv
temperatures = [0.0, 0.4, 0.8, 1.0]
model_names = ["deepseek/deepseek-r1-0528", "openai/gpt-4.1", "anthropic/claude-sonnet-4", "meta-llama/llama-3.1-405b-instruct", "openai/o4-mini-high"]

for model_type in GenModelType:
  path = os.path.join(folder_path, f'{model_type.name}')
  attempt_1_folder = os.path.join(path, "Attempt 1")
  attempt_2_folder = os.path.join(path, "Attempt 2")
  
  if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(attempt_1_folder)
    os.makedirs(attempt_2_folder)
    
    for temperature in temperatures:
      os.makedirs(os.path.join(attempt_1_folder, f"Temperature-{temperature}"))
      os.makedirs(os.path.join(attempt_2_folder, f"Temperature-{temperature}"))
    
theorems_json = parse_jsonl(proofs_path)
theorems = build_prompts(theorems_json)

for temperature in temperatures:
  for model_type in GenModelType:
    path = os.path.join(folder_path, f'{model_type.name}')
    
    match attempt:
      case 1:
        path = os.path.join(path, "Attempt 1")
      case 2:
        path = os.path.join(path, "Attempt 2")
      case _:
        raise ValueError("Input a valid attempt")
    
    path = os.path.join(path, f"Temperature-{temperature}")
    error_path = os.path.join(path, "generation-error.jsonl")
    log_path = os.path.join(path, "generation-error-log.txt")
    verification_path = os.path.join(path, "verification.jsonl")
    syntax_path = os.path.join(path, "syntax-verification.jsonl")
    failed_path = os.path.join(path, "failed-proofs.jsonl")
    latex_path = os.path.join(path, "trial.tex")
    model = GenModel(model_type, model_names[model_type.value], temperature, base_url, api_key, path)
    
    for i in range(len(theorems)):
      theorem = theorems[i]

      for j in range(len(theorem)):
        prompt = theorem[j]
        prompt_type = ""

        match i:
          case 0:
            prompt_type = "zero shot"
          case 1:
            prompt_type = "chain of thought"
          case 2:
            prompt_type = "few shot"

        try:
          response = model.get_response(SYSTEM_PROMPT, prompt)
          response_dict = model.parse_response(response)
        except Exception as e:
          with open(error_path, "a") as f:
            f.write(f"{json.dumps(prompt[0])}\n")
          
          with open(log_path, "a") as f:
            string_in = f"{str(datetime.now())}: Failed to generate proof for theorem {theorem["id"]} prompt type %s \n"  
            f.write(string_in % prompt_type)
          
          continue
        
        written_response = model.write_response(response_json, theorems_json[i], prompt_type, response_dict)

        grading = verification_model.verify_proof(response_dict)
        verification_model.write_result(verification_path, grading)

        syntax_grading = verify_latex(latex_path, written_response, LATEX_TEMPLATE)

        with open(syntax_path, "a") as f:
          f.write(f'{json.dumps(syntax_grading)}\n')

        # store failed proofs for proof correction
        if not grading["success"] or not syntax_grading["success"]:
          failed_proof_dict = {
            "id": theorems_json[i]["id"],
            "prompt type": prompt_type,
            "statement": theorems_json[i]["statement"],
            "reason": "" # blank for human reviwe
          }

          if not syntax_grading["success"]:
            failed_proof_dict["reason"] = "Incorrect LaTeX syntax."
          
          with open(failed_path, "a") as f:
            f.write(f'{json.dumps(failed_proof_dict)}\n')