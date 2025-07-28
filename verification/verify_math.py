from verification_model import NLIModelType, NLIModel, VerificationModel
from datasets import load_dataset
import torch
import sys
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
GRADE_THRESHOLD = 50

ds_nt  = load_dataset("EleutherAI/hendrycks_math", 'number_theory', split = "test")
ds_nt = ds_nt.filter(lambda x: int(x['level'][-1]) > 1)
ds_al = load_dataset("EleutherAI/hendrycks_math", 'algebra', split = "test")
ds_al = ds_al.filter(lambda x: int(x['level'][-1]) > 1)
dataset = ds_nt[:500] + ds_al[:500]

folder_path = sys.argv[0]
path = os.path.join(folder_path, "math-verification.jsonl")
f1_path = os.path.join(folder_path, "f1-scores.txt")

nli_model_names = ["Jaehun/PrismNLI-0.4B", "facebook/bart-large-mnli", "MoritzLaurer/DeBERTa-v3-base-mnli", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]
nli_models = [NLIModel(model_type, nli_model_names[model_type.value], DEVICE, GRADE_THRESHOLD) for model_type in NLIModelType]
verification_model = VerificationModel(nli_models)

tp_ensemble, fn_ensemble = 0, 0
f1_ensemble = 0

tp_baseline, fn_baseline = 0, 0
f1_baseline = 0

for problem in dataset:
  solution = problem["solution"].replace(".$", "$.").replace(".\\]", "\\].").split(".")
  solution = list(
    map(
      lambda x: x.strip() + "",
      filter(
        lambda x: x.strip() != "",
        solution
      )
    )
  )
  proof = {
    "premise": proof["problem"],
    "type": problem["type"],
    "proof": solution
  }
  
  result = verification_model.verify_math_proof(proof)
  verification_model.write_result(path, result)

  if result["success"]:
    tp_ensemble += 1
  else:
    fn_ensemble += 1 # no fp here, all proofs are ground truth and thankfully correct thank god
  
  if result["classifications"][0]:
    tp_baseline += 1
  else:
    fn_baseline += 1

f1_ensemble = 2 * tp_ensemble / (2 * tp_ensemble + fn_ensemble)
f1_baseline = 2 * tp_baseline / (2 * tp_baseline + fn_baseline)

with open(f1_path, "w") as f:
  f.write(f'''
F1-score of ensemble model: {f1_ensemble}
F1-score of baseline: {f1_baseline}         
''')