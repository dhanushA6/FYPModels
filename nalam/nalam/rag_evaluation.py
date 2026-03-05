import csv
import json
import os
import time
from typing import Any, Dict, List, Tuple

from config import GOOGLE_API_KEY
from nalam_retriever import NalamRetriever
from nalam_generator import NalamGenerator
from nalam_risk_engine import RiskAnalyzer, UserProfile
from macro_engine import get_macro_targets, get_meal_macro_split
from food_recommendations import mock_food_recommendation
from main import (
    get_mock_user,
    _userprofile_to_medical_dict,
    _userprofile_to_lifestyle_dict,
    _has_sufficient_medical_data,
)


DATASET_PATH = "evaluation_dataset_v2.json"
CSV_PATH = "rag_evaluation_results.csv"
DELAY_SECONDS = 1.0

# For evaluation we fix the user profile to case "1"
EVAL_PROFILE_CHOICE = "1"


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset.

    The file contains a JavaScript-style comment line
    ("// ------------------ NEW CATEGORY ------------------"),
    which is not valid JSON. We strip such comment lines
    before parsing so we don't have to modify the dataset file.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Drop lines that start with // (after optional whitespace)
    cleaned_lines: List[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    return json.loads(cleaned)


def is_special_question(example: Dict[str, Any]) -> bool:
    return example.get("domain") == "Food Recommendation"


def get_meal_type_for_example(example: Dict[str, Any]) -> str:
    return example.get("meal_type", "breakfast")


def ensure_csv_with_header(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question",
                "mode",
                "is_special_question",
                "retrieval_time_seconds",
                "generation_time_seconds",
                "retrieved_context",
                "response",
            ]
        )


def load_completed_combinations(path: str) -> List[Tuple[str, str]]:
    """
    Return list of (question, mode) pairs already present in the CSV.
    This keeps the CSV schema exactly as requested (no extra ID column).
    """
    if not os.path.exists(path):
        return []
    completed: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question")
            m = row.get("mode")
            if q and m:
                completed.append((q, m))
    return completed


def append_result_row(
    *,
    question: str,
    mode: str,
    is_special: bool,
    retrieval_time: float,
    generation_time: float,
    retrieved_context: str,
    response: str,
) -> None:
    ensure_csv_with_header(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                question,
                mode,
                "True" if is_special else "False",
                f"{retrieval_time:.4f}",
                f"{generation_time:.4f}",
                retrieved_context,
                response,
            ]
        )


def compute_averages_from_csv(path: str) -> Tuple[float, float]:
    if not os.path.exists(path):
        return 0.0, 0.0

    retrieval_times: List[float] = []
    generation_times: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rt = float(row.get("retrieval_time_seconds", "") or 0.0)
                gt = float(row.get("generation_time_seconds", "") or 0.0)
            except ValueError:
                continue
            retrieval_times.append(rt)
            generation_times.append(gt)

    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
    avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0.0
    return avg_retrieval, avg_generation


def main() -> None:
    # 1. Initialize core system (retriever + generator)
    retriever = NalamRetriever()
    generator = NalamGenerator(api_key=GOOGLE_API_KEY)

    # 2. Prepare fixed evaluation user profile (from main.py)
    current_user: UserProfile = get_mock_user(EVAL_PROFILE_CHOICE)
    medical_dict: Dict[str, Any] = _userprofile_to_medical_dict(current_user)
    lifestyle_dict: Dict[str, Any] = _userprofile_to_lifestyle_dict(current_user)

    risk_profile = None
    if _has_sufficient_medical_data(medical_dict):
        risk_profile = RiskAnalyzer.analyze(current_user)

    macro_targets = get_macro_targets()
    meal_splits_all = {
        m: get_meal_macro_split(m) for m in macro_targets["distribution"].keys()
    }

    # 3. Load dataset and completed combinations (for resume)
    dataset = load_dataset(DATASET_PATH)
    completed = set(load_completed_combinations(CSV_PATH))

    print(f"Loaded {len(dataset)} evaluation questions.")
    print(f"Already completed {len(completed)} (question, mode) pairs from CSV.")

    # 4. Loop through all questions
    for example in dataset:
        question = example.get("question", "").strip()
        if not question:
            continue

        special = is_special_question(example)
        modes: List[str] = ["normal"]
        if special:
            modes.append("food_recommendation")

        for mode in modes:
            key = (question, mode)
            if key in completed:
                continue

            retrieval_time = 0.0
            generation_time = 0.0
            retrieved_context = ""
            response = ""

            try:
                if mode == "food_recommendation":
                    # Food recommendation mode: no vector retrieval
                    meal_type = get_meal_type_for_example(example)

                    # Validate meal type against macro distribution, fallback if needed
                    if meal_type not in macro_targets["distribution"]:
                        # Fallback to breakfast if something weird is in the dataset
                        meal_type = "breakfast"

                    meal_macro_split = get_meal_macro_split(meal_type)
                    food_rec = mock_food_recommendation(meal_type)

                    structured_context = {
                        "mode": mode,
                        "user_profile": {
                            "medical": medical_dict,
                            "lifestyle": lifestyle_dict,
                        },
                        "macro_targets": {
                            "daily": macro_targets["daily"],
                            "distribution": macro_targets["distribution"],
                            "meal": meal_macro_split,
                        },
                        "food_recommendation": food_rec,
                    }

                    t_gen_start = time.perf_counter()
                    response = generator.generate_response(
                        context="",
                        user_question=question,
                        risk_profile=None,
                        structured_context=structured_context,
                    )
                    t_gen_end = time.perf_counter()
                    generation_time = t_gen_end - t_gen_start

                else:
                    # Normal RAG mode: retrieval + structured context
                    t_ret_start = time.perf_counter()
                    retrieved_context = retriever.get_relevant_context(question)
                    t_ret_end = time.perf_counter()
                    retrieval_time = t_ret_end - t_ret_start

                    structured_context = {
                        "mode": mode,
                        "user_profile": {
                            "medical": medical_dict,
                            "lifestyle": lifestyle_dict,
                        },
                        "risk_analysis": risk_profile,
                        "macro_targets": {
                            "daily": macro_targets["daily"],
                            "distribution": macro_targets["distribution"],
                            "meal_splits": meal_splits_all,
                        },
                    }

                    t_gen_start = time.perf_counter()
                    response = generator.generate_response(
                        context=retrieved_context,
                        user_question=question,
                        risk_profile=risk_profile,
                        structured_context=structured_context,
                    )
                    t_gen_end = time.perf_counter()
                    generation_time = t_gen_end - t_gen_start

            except Exception as e:
                response = f"Error during evaluation: {e}"

            # 5. Append result immediately to CSV (incremental saving)
            append_result_row(
                question=question,
                mode=mode,
                is_special=special,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                retrieved_context=retrieved_context,
                response=response,
            )

            # Track that we've done this combination
            completed.add(key)

            # 6. Delay between calls to avoid rate limiting (not included in timings)
            time.sleep(DELAY_SECONDS)

    # 7. Compute final averages from the CSV (all completed runs)
    avg_ret, avg_gen = compute_averages_from_csv(CSV_PATH)
    print("\n=== RAG Evaluation Summary ===")
    print(f"Average retrieval time (seconds): {avg_ret:.4f}")
    print(f"Average generation time (seconds): {avg_gen:.4f}")


if __name__ == "__main__":
    main()

