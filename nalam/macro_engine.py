from typing import Dict


def get_macro_targets() -> Dict:
    return {
        "daily": {
            "calories": 2200,
            "carbs_g": 250,
            "protein_g": 95,
            "fat_g": 85,
            "fiber_g": 35,
        },
        "distribution": {
            "breakfast": 0.25,
            "lunch": 0.35,
            "snacks": 0.15,
            "dinner": 0.25,
        },
    }


def get_meal_macro_split(meal_type: str) -> Dict:
    macros = get_macro_targets()
    ratio = macros["distribution"][meal_type]
    daily = macros["daily"]

    return {
        "meal": meal_type,
        "calories": daily["calories"] * ratio,
        "carbs_g": daily["carbs_g"] * ratio,
        "protein_g": daily["protein_g"] * ratio,
        "fat_g": daily["fat_g"] * ratio,
        "fiber_g": daily["fiber_g"] * ratio,
    }

