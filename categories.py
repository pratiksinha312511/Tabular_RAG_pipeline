"""Shared category display-name mapping used by pipeline and visualizations."""

# Raw DB format is "SUBCATEGORY_MAINCATEGORY" (e.g. RENT_HOUSING).
# This map turns them into natural, reader-friendly labels.
CATEGORY_DISPLAY: dict[str, str] = {
    "CASHBACK_INCOME":          "Cashback",
    "CLOTHING_SHOPPING":        "Clothing",
    "COFFEE_FOOD":              "Coffee & Cafes",
    "COURSES_EDUCATION":        "Courses & Education",
    "DOCTOR_HEALTH":            "Doctor & Medical",
    "ELECTRONICS_SHOPPING":     "Electronics",
    "FASTFOOD_FOOD":            "Fast Food",
    "FLIGHTS_TRAVEL":           "Flights",
    "FREELANCE_INCOME":         "Freelance Income",
    "FUEL_TRANSPORT":           "Fuel & Gas",
    "GENERAL_SHOPPING":         "General Shopping",
    "GROCERIES_FOOD":           "Groceries",
    "GYM_HEALTH":               "Gym & Fitness",
    "HOTELS_TRAVEL":            "Hotels & Travel",
    "INSURANCE_FINANCE":        "Insurance",
    "INTERNET_HOUSING":         "Internet & Utilities",
    "MOVIES_ENTERTAINMENT":     "Movies & Entertainment",
    "PHARMACY_HEALTH":          "Pharmacy",
    "REFUND_INCOME":            "Refunds",
    "RENT_HOUSING":             "Rent & Housing",
    "RESTAURANT_FOOD":          "Restaurants & Dining",
    "RIDESHARE_TRANSPORT":      "Rideshare & Taxi",
    "SALARY_INCOME":            "Salary",
    "STREAMING_ENTERTAINMENT":  "Streaming Services",
    "SUBSCRIPTION_FINANCE":     "Subscriptions",
    "SUPPLIES_PETS":            "Pet Supplies",
    "UTILITIES_HOUSING":        "Utilities",
}


def friendly_category(raw: str) -> str:
    """Convert a raw DB category like RENT_HOUSING → 'Rent & Housing'."""
    if raw in CATEGORY_DISPLAY:
        return CATEGORY_DISPLAY[raw]
    parts = raw.split("_")
    if len(parts) == 2:
        return f"{parts[0].capitalize()} ({parts[1].capitalize()})"
    return raw.replace("_", " ").title()
