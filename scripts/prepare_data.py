import os
import re
import json
import random
from datasets import load_dataset
from tqdm import tqdm

RANDOM_SEED = 42
MAX_NUMINA_SAMPLES = 25000
MAX_CHAR_LEN = 2000  # Filters out massive multi-page Olympiad proofs
MIN_CHAR_LEN = 40    # Filters out malformed/trivial single-word rows

SYSTEM_PROMPT = (
    "You are Spectre, an expert math tutor. Always think through the problem "
    "step-by-step inside <thought> tags before providing the final answer."
)

def clean_text(text: str) -> str:
    """Normalize whitespace and strip noisy artifacts."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    return text.strip()

def format_gsm8k(example):
    """GSM8K encodes intermediate steps and final answer separated by '####'."""
    q = clean_text(example.get("question", ""))
    a = clean_text(example.get("answer", ""))
    
    if not q or not a:
        return None

    # Remove inline calculator callouts like <<12+3=15>>
    cleaned_steps = re.sub(r"<<.*?>>", "", a)

    if "####" in cleaned_steps:
        steps, final_ans = cleaned_steps.split("####", 1)
        assistant_reply = (
            f"<thought>\n{steps.strip()}\n</thought>\n\n"
            f"Final Answer: {final_ans.strip()}"
        )
    else:
        assistant_reply = f"<thought>\n{cleaned_steps.strip()}\n</thought>"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_reply}
        ]
    }

def format_numina(example):
    """NuminaMath-CoT stores problems in 'problem' and solutions in 'solution'."""
    problem = clean_text(example.get("problem", ""))
    solution = clean_text(example.get("solution", ""))

    if not problem or not solution:
        return None

    # Filter out overly verbose or trivial proofs
    if len(solution) > MAX_CHAR_LEN or len(solution) < MIN_CHAR_LEN:
        return None

    # Look for LaTeX boxed answers common in Numina: \boxed{...}
    boxed_match = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if boxed_match:
        final_ans = boxed_match[-1].strip()
        assistant_reply = (
            f"<thought>\n{solution}\n</thought>\n\n"
            f"Final Answer: {final_ans}"
        )
    else:
        assistant_reply = f"<thought>\n{solution}\n</thought>"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_reply}
        ]
    }

def main():
    random.seed(RANDOM_SEED)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    all_examples = []

    # 1. Process GSM8K
    print("Fetching GSM8K...")
    gsm = load_dataset("openai/gsm8k", "main")
    
    gsm_train = [format_gsm8k(ex) for ex in gsm["train"]]
    gsm_train = [ex for ex in gsm_train if ex is not None]
    all_examples.extend(gsm_train)
    print(f"Loaded {len(gsm_train)} valid GSM8K examples.")

    # 2. Stream and filter NuminaMath-CoT
    print("Streaming and filtering NuminaMath-CoT...")
    numina_stream = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    
    numina_collected = 0
    pbar = tqdm(total=MAX_NUMINA_SAMPLES, desc="NuminaMath")
    
    for ex in numina_stream:
        formatted = format_numina(ex)
        if formatted:
            all_examples.append(formatted)
            numina_collected += 1
            pbar.update(1)
            if numina_collected >= MAX_NUMINA_SAMPLES:
                break
    pbar.close()

    # 3. Shuffle dataset blend
    random.shuffle(all_examples)
    total = len(all_examples)
    print(f"Total curated dataset: {total} examples.")

    # 4. Split 95% train, 5% validation
    val_size = min(1500, int(total * 0.05))
    train_split = all_examples[:-val_size]
    valid_split = all_examples[-val_size:]

    # 5. Write to JSONL
    print(f"Writing to {data_dir}...")
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_split:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for ex in valid_split:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Finished. Saved {len(train_split)} rows to {train_path} and {len(valid_split)} rows to {valid_path}.")

if __name__ == "__main__":
    main()