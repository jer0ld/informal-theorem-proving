from enum import Enum
from openai import OpenAI
import os
import json

class GenModelType(Enum):
  DEEPSEEK = 0
  GPT = 1
  CLAUDE = 2
  LLAMA = 3
  O4 = 4

class GenModel():
  def __init__(self, model_type: GenModelType, model_name: str, temperature: float, url: str, api_key: str, folder_path: str) -> None:
    self.model_type = model_type
    self.model_name = model_name
    self.temperature = temperature
    self.path = os.path.join(folder_path, "proofs.jsonl")
    
    self.url = url
    self.api_key = api_key
    self.client = OpenAI(base_url = self.base_url, api_key = self.api_key)

  def get_response(self, system_prompt: str, prompt: list) -> str:
    response = self.client.chat.completions.create(
      model = self.model_name,
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
      ]
  	)

    try:
      return response.choices[0].message.content
    except Exception as e:
      raise Exception(f"Unable to generate proof")
        
  def parse_response(self, response):
    response_by_line = map(
      lambda x: x.strip().strip("*_").strip(),
      	filter(
          lambda x: x != "",
          response.split("\n")
				)
		)
    response = []

    for line in response_by_line:
      response += list(
        map(
          lambda x: x + ".",
          line.split(".")
        )
      )
        
    idx_1 = response.index("Variable definitions:")
    idx_2 = response.index("Proof type(s):")
    idx_3 = response.index("Proof:")
    idx_4 = response.index("QED")

    var_def = "\n".join(response[idx_1 + 1: idx_2])
    proof_type = response[idx_2 + 1: idx_3]
    proof = response[idx_3 + 1: idx_4]

    return {
      "premise": var_def,
      "proof type": proof_type,
      "proof": proof
    }
  
  def write_response(self, theorem: dict, prompt_type: str, response: dict) -> dict:
    response_dict = {
			"id": theorem["id"],
			"prompt type": prompt_type,
      "proof type": response["proof type"],
      "premise": response["premise"],
      "proof": response["proof"],
		}
      
    with open(self.path, "a") as file:
      file.write(f'{json.dumps(response_dict)}\n')

    return response_dict
      