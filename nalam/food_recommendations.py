from typing import Dict, Optional


def mock_food_recommendation(meal_type: str) -> Optional[Dict]:
    meals = {
        "breakfast": {
            "meal": "breakfast",
            "foods": [
                {
                    "name": "Oats",
                    "quantity": "60g",
                    "calories": 230,
                    "carbs_g": 38,
                    "protein_g": 8,
                    "fat_g": 4,
                    "fiber_g": 6,
                }
            ],
        },
        "lunch": {
            "meal": "lunch",
            "foods": [
                {
                    "name": "Grilled chicken + brown rice + salad",
                    "quantity": "1 plate",
                    "calories": 520,
                    "carbs_g": 55,
                    "protein_g": 38,
                    "fat_g": 14,
                    "fiber_g": 10,
                }
            ],
        },
        "snacks": {
            "meal": "snacks",
            "foods": [
                {
                    "name": "Greek yogurt + berries",
                    "quantity": "1 bowl",
                    "calories": 180,
                    "carbs_g": 20,
                    "protein_g": 15,
                    "fat_g": 4,
                    "fiber_g": 5,
                }
            ],
        },
        "dinner": {
            "meal": "dinner",
            "foods": [
                {
                    "name": "Paneer/tofu + mixed vegetables",
                    "quantity": "1 bowl",
                    "calories": 420,
                    "carbs_g": 28,
                    "protein_g": 28,
                    "fat_g": 22,
                    "fiber_g": 9,
                }
            ],
        },
    }
    return meals.get(meal_type)

