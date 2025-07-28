from Code.generation.prompt_type import PromptType
import json
import os

def parse_jsonl(path: str) -> list:
  with open(path, "r") as f:
    jsonl = list(f)
    
  return [json.loads(json_str) for json_str in jsonl]

def build_prompt(prompt_type: PromptType, theorem: dict) -> str:
  msg = ""
  theorem_statement = f'''
%s
Prove the following proposition: {theorem["statement"]}
%s
'''

  match prompt_type:
    case PromptType.ZERO_SHOT:
      msg = theorem_statement % ("", "")
    case PromptType.CHAIN_OF_THOUGHT:
      msg = theorem_statement % ("", "Let's think step by step")
    case PromptType.FEW_SHOT:
      msg = theorem_statement % (f'Here is an example:\n {theorem["example"]}',"")
    case _:
      raise ValueError("Invalid prompt type")
    
  return msg.strip()

def build_prompts(path: str) -> list:
  proofs = parse_jsonl(path)
  prompts = [] # 2D array, each row contains FS, COT, ZS for one proof
  
  for proof in proofs:
    proof_zs = build_prompt(PromptType.ZERO_SHOT, proof)
    proof_cot = build_prompt(PromptType.CHAIN_OF_THOUGHT, proof)
    
    arr = [
      [{
        "type": "text",
        "text": proof_zs
      }],
      [{
        "type": "text",
        "text": proof_cot
      }]
    ]
    
    if proof["example"] != "":
      prompt_fs = build_prompt(PromptType.FEW_SHOT, proof)
      arr += [{
        "type": "text",
        "text": prompt_fs
      }]
      
    prompts.append(arr)
    
  return prompts

def verify_latex(path: str, proof: dict, template: str) -> dict:
  result = {
    "id": proof["id"],
    "prompt type": proof["prompt type"],
    "success": False
  }

  contents = " ".join([proof["premise"]] + proof["proof"])
  
  with open(path, "w") as f:
    f.write(template % contents)
  
  result["success"] = os.system(f"latexmk -pdf {path}") == 0
  
  return result

def find_thing(things: list, id: int, prompt_type: str = "") -> dict | int:
  for thing in things:
    if thing["id"] == id:
      if not "prompt type" in thing.keys() or thing["prompt type"] == prompt_type:
        return thing
      
  return -1