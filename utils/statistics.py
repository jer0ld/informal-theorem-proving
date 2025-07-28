from utils import parse_jsonl, find_thing

def calc_avg_clarity(attempt_1_path: str, attempt_2_path: str) -> float:
  proofs = parse_jsonl(attempt_1_path)
  second_proofs = parse_jsonl(attempt_2_path)
  avg_clarity, ctr = 0, 0

  for proof in proofs:
    correct = proof["success"]
    clarity = proof["clarity"]

    if not correct:
      try:
        second_proof = find_thing(second_proofs, proof["id"], proof["prompt type"])
        clarity = second_proof["clarity"]
      except Exception as e:
        print(f'Could not find the result of proof {proof["id"]} prompt type {proof["prompt type"]}: {e}')
        continue  
    
    avg_clarity += clarity
    ctr += 1

  return avg_clarity / ctr

def calc_avg_descriptiveness(attempt_1_path: str, attempt_2_path: str) -> float:
  proofs = parse_jsonl(attempt_1_path)
  second_proofs = parse_jsonl(attempt_2_path)
  avg_descriptiveness, ctr = 0, 0

  for proof in proofs:
    correct = proof["success"]
    descriptiveness = proof["descriptiveness"]

    if not correct:
      try:
        second_proof = find_thing(second_proofs, proof["id"], proof["prompt type"])
        descriptiveness = second_proof["descriptiveness"]
      except Exception as e:
        print(f'Could not find the result of proof {proof["id"]} prompt type {proof["prompt type"]}: {e}')
        continue  
    
    avg_descriptiveness += descriptiveness
    ctr += 1

  return avg_descriptiveness / ctr

def calc_avg_redundancy(attempt_1_path: str, attempt_2_path: str) -> float:
  proofs = parse_jsonl(attempt_1_path)
  second_proofs = parse_jsonl(attempt_2_path)
  tot_redundancy = 0
  tot_proof_length = 0

  for proof in proofs:
    length = len(proof["proof"])
    redundancy = proof["redundancy"] * length / 100
    correct = proof["success"]

    if not correct:
      try:
        second_proof = find_thing(second_proofs, proof["id"], proof["prompt type"])
        length = len(second_proof["proof"])
        redundancy = second_proof["redundancy"] * length / 100
      except Exception as e:
        print(f'Could not find the result of proof {proof["id"]} prompt type {proof["prompt type"]}: {e}')
        continue       

    tot_proof_length += length
    tot_redundancy += redundancy

  return tot_redundancy / tot_proof_length

def calc_ensemble_f1(attempt_1_path: str, attempt_2_path: str) -> float:
  results = parse_jsonl(attempt_1_path)
  second_results = parse_jsonl(attempt_2_path)
  tp, fp, fn = 0, 0, 0

  for result in results:
    correct = result["success-human"]
    classification = result["success"]

    if not correct:
      try:
        second_result = find_thing(second_results, result["id"], result["prompt type"])
        correct = second_result["success-human"]
        classification = second_result["success"]
      except Exception as e:
        print(f'Could not find the result of proof {result["id"]} prompt type {result["prompt type"]}: {e}')
        continue      

    if correct and classification:
      tp += 1
    elif not correct and classification:
      fp += 1
    elif correct and not classification:
      fn += 1
  
  return 2 * tp / (2 * tp + fp + fn)

def calc_baseline_f1(attempt_1_path: str, attempt_2_path: str, baseline_idx: int) -> float:
  try:
    assert type(baseline_idx) == int
  except Exception as e:
    raise ValueError("The index of the baseline model should be an integer.")

  results = parse_jsonl(attempt_1_path)
  second_results = parse_jsonl(attempt_2_path)
  tp, fp, fn = 0, 0, 0

  for result in results:
    correct = result["success-human"]
    classification = result["classifications"][baseline_idx]

    if not correct:
      try:
        second_result = find_thing(second_results, result["id"], result["prompt type"])
        correct = second_result["success-human"]
        classification = second_result["classifications"][baseline_idx]
      except Exception as e:
        print(f'Could not find the result of proof {result["id"]} prompt type {result["prompt type"]}: {e}')
        continue      

    if correct and classification:
      tp += 1
    elif not correct and classification:
      fp += 1
    elif correct and not classification:
      fn += 1
  
  return 2 * tp / (2 * tp + fp + fn)

def calc_math_accuracy(attempt_1_path: str, attempt_2_path: str) -> float:
  attempt_1_result = parse_jsonl(attempt_1_path)
  attempt_2_result = parse_jsonl(attempt_2_path)
  tot_correct, tot = 0, 0

  for result in attempt_1_result:
    correct = result["success-human"]

    if not correct:
      try:
        second_result = find_thing(attempt_2_result, result["id"], result["prompt type"])
        correct = second_result["success-human"]
      except Exception as e:
        print(f'Could not find the result of proof {result["id"]} prompt type {result["prompt type"]}: {e}')
        continue
    
    tot_correct += correct
    tot += 1

  return round(tot_correct * 100 / tot, 2)