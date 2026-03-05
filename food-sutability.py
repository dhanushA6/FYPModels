import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

genai.configure(api_key=GEMINI_API_KEY)

# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class NutritionData:
    food_name: str
    quantity_g: float
    calories_kcal: float
    carbohydrates_g: float
    protein_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    glycemic_index: float
    sodium_mg: float

@dataclass
class FoodResult:
    food_name: str
    classification: str
    risk_score: int
    reasons: list[str]
    nutrition: dict = field(default_factory=dict)

@dataclass
class MealResult:
    foods: list[dict]
    total_meal_nutrition: dict
    overall_classification: str
    overall_risk_score: int
    overall_reasons: list[str]

# ── Gemini Nutrition Fetch ───────────────────────────────────────────────────
def get_food_nutrition_llm(food_name: str, quantity: float, unit: str) -> NutritionData:
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are a clinical nutrition database.

Food: {food_name}
Quantity: {quantity} {unit}

Convert this quantity into grams internally.

Rules:
- If unit is ml → assume appropriate density.
- If cup/tbsp/tsp → use standard Indian household measures.
- If piece/pieces → assume average standard size.
- If already grams → use directly.

Return ONLY valid JSON in EXACTLY this format:

{{
  "food_name": "{food_name}",
  "quantity_g": <converted grams>,
  "calories_kcal": <number>,
  "carbohydrates_g": <number>,
  "protein_g": <number>,
  "fat_g": <number>,
  "fiber_g": <number>,
  "sugar_g": <number>,
  "glycemic_index": <number>,
  "sodium_mg": <number>
}}

No explanation. JSON only.
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not json_match:
            raise ValueError("No JSON found")

        data = json.loads(json_match.group())

        return NutritionData(
            food_name=str(data["food_name"]),
            quantity_g=float(data["quantity_g"]),
            calories_kcal=float(data["calories_kcal"]),
            carbohydrates_g=float(data["carbohydrates_g"]),
            protein_g=float(data["protein_g"]),
            fat_g=float(data["fat_g"]),
            fiber_g=float(data["fiber_g"]),
            sugar_g=float(data["sugar_g"]),
            glycemic_index=float(data["glycemic_index"]),
            sodium_mg=float(data["sodium_mg"]),
        )

    except Exception as e:
        print(f"[WARNING] LLM failed for {food_name}: {e}")
        return NutritionData(
            food_name=food_name,
            quantity_g=100,
            calories_kcal=100,
            carbohydrates_g=15,
            protein_g=5,
            fat_g=2,
            fiber_g=2,
            sugar_g=5,
            glycemic_index=55,
            sodium_mg=100,
        )

# ── RULE ENGINE ──────────────────────────────────────────────────────────────
def evaluate_diabetes_risk(profile: dict, nutrition: NutritionData):
    score = 0
    reasons = []

    if nutrition.glycemic_index > 70:
        score += 2
        reasons.append("High glycemic index.")

    if nutrition.sugar_g > 10 and profile.get("post_prandial_glucose_mg_dl", 0) > 160:
        score += 2
        reasons.append("High sugar with elevated PPG.")

    if nutrition.fiber_g < 3:
        score += 1
        reasons.append("Low fiber.")

    if profile.get("diabetes_type") == "Type2" and nutrition.carbohydrates_g > 30:
        score += 1
        reasons.append("High carbs for Type2.")

    return score, reasons

def evaluate_lipid_risk(profile, nutrition):
    score = 0
    reasons = []

    if profile.get("ldl_cholesterol_mg_dl", 0) > 100 and nutrition.fat_g > 10:
        score += 1
        reasons.append("High LDL + fat.")

    return score, reasons

def evaluate_bp_risk(profile, nutrition):
    score = 0
    reasons = []

    if nutrition.sodium_mg > 500:
        score += 1
        reasons.append("High sodium.")

    return score, reasons

def evaluate_kidney_safety(profile, nutrition):
    score = 0
    reasons = []

    if profile.get("eGFR", 100) < 60 and nutrition.protein_g > 15:
        score += 2
        reasons.append("High protein for kidney.")

    return score, reasons

def evaluate_calorie_balance(profile, nutrition):
    return 0, []

def generate_reasoning(profile, nutrition, risk_score):
    reasons = []

    if risk_score == 0:
        reasons.append("Metabolically safe.")

    return reasons

def classify_food_item(profile: dict, nutrition: NutritionData) -> FoodResult:
    total_score = 0
    all_reasons = []

    for evaluator in [
        evaluate_diabetes_risk,
        evaluate_lipid_risk,
        evaluate_bp_risk,
        evaluate_kidney_safety,
        evaluate_calorie_balance
    ]:
        score, reasons = evaluator(profile, nutrition)
        total_score += score
        all_reasons.extend(reasons)

    all_reasons.extend(generate_reasoning(profile, nutrition, total_score))

    if total_score <= 2:
        classification = "SUITABLE"
    elif total_score <= 5:
        classification = "MODERATE"
    else:
        classification = "NOT SUITABLE"

    return FoodResult(
        food_name=nutrition.food_name,
        classification=classification,
        risk_score=total_score,
        reasons=list(dict.fromkeys(all_reasons)),
        nutrition=asdict(nutrition)
    )

def aggregate_nutrition(nutrition_list):
    totals = {
        "total_calories_kcal": 0.0,
        "total_carbohydrates_g": 0.0,
        "total_protein_g": 0.0,
        "total_fat_g": 0.0,
        "total_fiber_g": 0.0,
        "total_sugar_g": 0.0,
        "average_glycemic_index": 0.0,
        "total_sodium_mg": 0.0,
    }

    gi_values = []

    for n in nutrition_list:
        totals["total_calories_kcal"] += n.calories_kcal
        totals["total_carbohydrates_g"] += n.carbohydrates_g
        totals["total_protein_g"] += n.protein_g
        totals["total_fat_g"] += n.fat_g
        totals["total_fiber_g"] += n.fiber_g
        totals["total_sugar_g"] += n.sugar_g
        totals["total_sodium_mg"] += n.sodium_mg
        gi_values.append(n.glycemic_index)

    if gi_values:
        totals["average_glycemic_index"] = round(sum(gi_values)/len(gi_values),1)

    return {k: round(v,2) for k,v in totals.items()}

def classify_full_meal(profile, nutrition_list, food_results):
    agg = aggregate_nutrition(nutrition_list)
    score = 0
    reasons = []

    if agg["average_glycemic_index"] > 70:
        score += 2
        reasons.append("High meal GI.")

    if score <= 2:
        classification = "SUITABLE"
    elif score <= 5:
        classification = "MODERATE"
    else:
        classification = "NOT SUITABLE"

    return classification, score, reasons

# ── Main Orchestrator ─────────────────────────────────────────────────────────
def analyze_meal(profile: dict, foods: list[dict]) -> dict:
    nutrition_list = []
    food_results = []

    for food_item in foods:
        name = food_item["food_name"]

        if "quantity_g" in food_item:
            qty = food_item["quantity_g"]
            unit = "g"
        else:
            qty = food_item["quantity"]
            unit = food_item["unit"]

        nutrition = get_food_nutrition_llm(name, qty, unit)
        result = classify_food_item(profile, nutrition)

        nutrition_list.append(nutrition)
        food_results.append(result)

    total_nutrition = aggregate_nutrition(nutrition_list)
    overall_classification, overall_risk_score, overall_reasons = classify_full_meal(profile, nutrition_list, food_results)

    return {
        "foods": [asdict(r) for r in food_results],
        "total_meal_nutrition": total_nutrition,
        "overall_classification": overall_classification,
        "overall_risk_score": overall_risk_score,
        "overall_reasons": overall_reasons
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_profile = {
        "age": 40,
        "gender": "Female",
        "height_cm": 165.0,
        "weight_kg": 68.0,
        "bmi": 24.9,
        "diabetes_type": "Type2",
        "diabetes_duration_years": 4.0,
        "hba1c_percent": 6.5,
        "fasting_glucose_mg_dl": 130.0,
        "post_prandial_glucose_mg_dl": 175.0,
        "hypoglycemia_history": False,
        "triglycerides_mg_dl": 140.0,
        "ldl_cholesterol_mg_dl": 110.0,
        "hdl_cholesterol_mg_dl": 60.0,
        "systolic_bp_mmHg": 122,
        "diastolic_bp_mmHg": 80,
        "creatinine_mg_dl": 0.9,
        "eGFR": 98.0,
        "physical_activity_level": "Highly Active",
        "sleep_hours": 8.0,
    }

    # Works with BOTH formats now
    foods = [
        {"food_name": "idly", "quantity": 2, "unit": "pieces"},
        {"food_name": "sambar", "quantity": 1, "unit": "cup"},
        {"food_name": "beans poriyal", "quantity": 100, "unit": "g"},
        {"food_name": "egg", "quantity": 1, "unit": "piece"},
    ]

    result = analyze_meal(user_profile, foods)

    print("\n" + "="*60)
    print("  FINAL MEAL SUITABILITY REPORT")
    print("="*60)
    print(json.dumps(result, indent=2))

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    for food in result["foods"]:
        status_symbol = "✔" if food["classification"] == "SUITABLE" else (
            "⚠" if food["classification"] == "MODERATE" else "✘"
        )
        print(f"  {status_symbol} {food['food_name']}: {food['classification']} (Risk Score: {food['risk_score']})")

    overall_symbol = "✔" if result["overall_classification"] == "SUITABLE" else (
        "⚠" if result["overall_classification"] == "MODERATE" else "✘"
    )
    print(f"\n  {overall_symbol} OVERALL MEAL: {result['overall_classification']} (Risk Score: {result['overall_risk_score']})")
    print("="*60)