from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_insight(
    runway,
    stress,
    risk,
    factors
):

    prompt = f"""
You are an experienced CFO advisor.

Analyze the following company metrics.

Runway: {runway:.2f} months
Stress Probability: {stress}%
Risk Level: {risk}

Risk Factors:
{', '.join(factors)}

Provide:

1. Financial Insight
2. Risk Explanation
3. Recommendation

Keep the response professional and under 120 words.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
