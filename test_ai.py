from ai_insights import generate_insight

test_cases = [

    {
        "name": "Healthy Startup",
        "runway": 24,
        "stress": 15,
        "risk": "Low",
        "factors": []
    },

    {
        "name": "Moderate Risk Startup",
        "runway": 7.5,
        "stress": 60,
        "risk": "Moderate",
        "factors": [
            "Operating expenses exceed revenue",
            "Negative cash flow trend"
        ]
    },

    {
        "name": "Critical Startup",
        "runway": 1.8,
        "stress": 95,
        "risk": "Critical",
        "factors": [
            "Runway below 3 months",
            "High debt burden",
            "Negative cash flow trend"
        ]
    }

]


print("\n" + "=" * 60)
print("REDSTONE AI INSIGHTS TEST")
print("=" * 60)

for i, case in enumerate(test_cases, start=1):

    print(f"\nTEST CASE {i}: {case['name']}")
    print("-" * 60)

    result = generate_insight(
        runway=case["runway"],
        stress=case["stress"],
        risk=case["risk"],
        factors=case["factors"]
    )

    print(f"Runway: {case['runway']} months")
    print(f"Stress Probability: {case['stress']}%")
    print(f"Risk Level: {case['risk']}")

    if case["factors"]:
        print("\nRisk Factors:")
        for factor in case["factors"]:
            print(f"• {factor}")

    print("\nAI Insight:")
    print(result)

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
