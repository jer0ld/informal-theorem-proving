# On the Application of NLP Models to Reasoning-Based Informal Theorem Proving #

This repo contains:

- Code for proof generation
- Code for proof verification, including the ensemble model developed within the paper
- Code for proof evaluation
- The 150 propositions used

# Software Prerequisites #

To run any code used within this project, the following need to be installed.

Python packages:
- Transformers (`pip install transformers`)
- OpenAI (`pip install openai`)
- PyTorch (`pip install torch`)

Software:
- TeXLive (`apt-get install texlive-full`)
- latexmk (`sudo apt install latexmk`)

# Generating Proofs #

To generate proofs for a given attempt, a base URL for provider and an API key is needed. Specify which folder you want the proofs and where the propositions to be proven are stored. Then run

```
python main.py <base_url> <api_key> /path/to/desired/folder /path/to/proposition/file <attempt_number>
```

The following file structure will be generated:

```
folder_name/
  DEEPSEEK/
    Attempt 1/
      Temperature-0.0/
        proofs.jsonl
        failed-proofs.jsonl
        verification.jsonl
        syntax-verification.jsonl
        generation-error.jsonl
        generation-error-log.txt
        trial.tex
      Temperature-0.4/
        ...
      ...
    Attempt 2/
      ...
  GPT/
  CLAUDE/
  ...
```

with all responses contained in `proofs.jsonl`, all proofs marked as incorrect by the verification modl in `failed-proofs.jsonl`, the verification model's grading of each proof in `verification.jsonl`, syntax verification results for each proof in `syntax-verification.jsonl`, prompts where errors in generation occur in `generation-error.jsonl` and the corresponding log of errors in `generation-error-log.txt`. `trial.tex` is the LaTeX file the proof is dumped into during syntax verification.

Running the script only automatically grades the proof. Evaluating clarity, descriptiveness, redundancy and the proof verification to determined false positives / negatives will all need to be done manually.

# Calculating Data #

All of the functions needed to calculate the mean clarity, descriptiveness, redundancy and the F1-scores of both the baseline model and the ensemble model are contained in `statistics.py`. These may be imported and run on the generated proofs.

# Additional Data for the Verification Models #

You may also want to run the proof verification models on the MATH dataseet. To do this, install the datasets library (`pip install datasets`) and then run the script

```
python verify_math.py /path/to/desired/folder
```

which creates the file structure

```
folder_name/
  math_verification.jsonl
  f1-scores.txt
```

which contains the results of the verification models in `math_verification.jsonl` and the F1-scores of each model in `f1-scores.txt`. 