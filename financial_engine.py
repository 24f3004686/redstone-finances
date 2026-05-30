def cash_health_score(runway):

    score = min(
        (runway / 24) * 100,
        100
    )

    return round(score, 2)


def stress_probability(runway):

    stress = max(
        0,
        100 - (runway * 4)
    )

    return round(stress, 2)


def risk_level(runway):

    if runway < 3:
        return "Critical"

    elif runway < 6:
        return "High"

    elif runway < 12:
        return "Moderate"

    elif runway < 24:
        return "Stable"

    return "Excellent"