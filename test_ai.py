from ai_insights import generate_insight

result = generate_insight(
    runway=7.5,
    stress=60,
    risk="Moderate",
    factors=[
        "Operating expenses exceed revenue",
        "Negative cash flow trend"
    ]
)

print(result)
