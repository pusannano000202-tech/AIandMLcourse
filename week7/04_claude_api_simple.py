"""
04. Claude API 간단 데모 (API 키 없이도 실행 가능)
Claude API 개념 이해

실제 API 호출 대신 시뮬레이션으로 학습:
- API 요청/응답 구조
- 프롬프트 엔지니어링 기초
- Temperature, Max tokens 등 파라미터
- 실제 사용 시나리오

학습 목표:
1. LLM API 사용법 이해
2. 프롬프트 작성 방법
3. 응답 파싱과 처리
4. 물리학 문제 해결 예시
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os
import json

output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Claude API Concepts and Simulation")
print("="*70)

def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'Malgun Gothic' in font_list:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif 'Gulim' in font_list:
        plt.rcParams['font.family'] = 'Gulim'
    elif 'Batang' in font_list:
        plt.rcParams['font.family'] = 'Batang'
    elif 'AppleGothic' in font_list:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ============================================================================
# API Request/Response Structure
# ============================================================================

print("\n1. Understanding API Structure...")

# Example API request
api_request = {
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1024,
    "temperature": 0.7,
    "messages": [
        {
            "role": "user",
            "content": "Explain quantum tunneling in simple terms."
        }
    ]
}

# Simulated API response
api_response = {
    "id": "msg_01XYZ123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Quantum tunneling is a phenomenon where particles can pass through energy barriers that classical physics says they shouldn't be able to cross. Imagine throwing a ball at a hill - classically, if the ball doesn't have enough energy, it rolls back. But in quantum mechanics, there's a small probability the ball could appear on the other side of the hill, as if it tunneled through!"
        }
    ],
    "model": "claude-3-sonnet-20240229",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 87
    }
}

print("\n   API Request Structure:")
print(json.dumps(api_request, indent=2))
print("\n   API Response Structure:")
print(json.dumps(api_response, indent=2))

# ============================================================================
# Parameter Effects
# ============================================================================

print("\n2. Simulating Parameter Effects...")

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# (a) Temperature effect on output diversity
ax1 = fig1.add_subplot(gs[0, 0])

temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
diversity_scores = [10, 30, 60, 85, 95]  # simulated
coherence_scores = [95, 90, 75, 60, 40]  # simulated

ax1.plot(temperatures, diversity_scores, marker='o', linewidth=2,
        markersize=8, label='Diversity', color='blue')
ax1.plot(temperatures, coherence_scores, marker='s', linewidth=2,
        markersize=8, label='Coherence', color='red')

ax1.set_xlabel('Temperature', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('(a) Temperature Effect', fontsize=13, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])

# Add annotations
ax1.annotate('Deterministic\n(factual tasks)', xy=(0.0, 95),
            xytext=(0.3, 100), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='green'))
ax1.annotate('Creative\n(storytelling)', xy=(1.5, 95),
            xytext=(1.2, 80), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='purple'))

# (b) Max tokens effect
ax2 = fig1.add_subplot(gs[0, 1])

max_tokens_values = [50, 100, 500, 1000, 4000]
avg_response_lengths = [45, 95, 480, 950, 3800]  # slightly less than max
response_quality = [60, 75, 90, 95, 95]  # plateaus

ax2.bar(range(len(max_tokens_values)), avg_response_lengths,
       color='skyblue', edgecolor='black', linewidth=1.5, label='Avg Response Length')
ax2_twin = ax2.twinx()
ax2_twin.plot(range(len(max_tokens_values)), response_quality,
             marker='o', linewidth=2, markersize=8, color='red', label='Quality')

ax2.set_xlabel('Max Tokens Setting', fontsize=12)
ax2.set_ylabel('Response Length (tokens)', fontsize=12, color='blue')
ax2_twin.set_ylabel('Quality Score', fontsize=12, color='red')
ax2.set_title('(b) Max Tokens Effect', fontsize=13, weight='bold')
ax2.set_xticks(range(len(max_tokens_values)))
ax2.set_xticklabels(max_tokens_values, fontsize=10)
ax2.tick_params(axis='y', labelcolor='blue')
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2.grid(True, alpha=0.3, axis='y')

# (c) Prompt engineering examples
ax3 = fig1.add_subplot(gs[1, :])
ax3.axis('off')

prompt_examples = """
PROMPT ENGINEERING EXAMPLES:

Bad Prompt:
  "Explain physics"
  → Too vague, unclear what aspect

Good Prompt:
  "Explain Newton's second law (F=ma) with a real-world example suitable for high school students."
  → Clear, specific, audience-defined

Better Prompt:
  "You are a physics teacher. Explain Newton's second law (F=ma) to high school students.
   Include: 1) The formula, 2) What each variable means, 3) A real-world example,
   4) Common misconceptions. Use simple language and analogies."
  → Role-playing, structured, comprehensive

Physics Problem Solving:
  "I have a physics problem: A 5kg object is pushed with 20N force. What is its acceleration?
   Please show step-by-step solution and explain each step."
  → Clear task, requests methodology

Code Generation:
  "Write a Python function to calculate the trajectory of a projectile.
   Inputs: initial velocity (m/s), launch angle (degrees), initial height (m)
   Output: list of (x, y) positions at 0.1s intervals
   Include docstring and example usage."
  → Specific requirements, format defined
"""

ax3.text(0.5, 0.5, prompt_examples, fontsize=9, verticalalignment='center',
        horizontalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax3.set_title('(c) Prompt Engineering Best Practices', fontsize=13, weight='bold')

plt.savefig(f'{output_dir}/04_api_parameters.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_api_parameters.png")

# ============================================================================
# Physics Use Cases
# ============================================================================

print("\n3. LLM for Physics Applications...")

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 1, figure=fig2, hspace=0.4)

use_cases = [
    "Code Generation\n(Simulations)",
    "Problem Solving\n(Step-by-step)",
    "Concept Explanation\n(Teaching)",
    "Data Analysis\n(Interpretation)",
    "Literature Review\n(Summarization)",
    "LaTeX Generation\n(Documentation)"
]

usefulness = [85, 90, 95, 75, 80, 90]
difficulty = [30, 20, 15, 40, 35, 25]

# (a) Usefulness rating
ax1 = fig2.add_subplot(gs[0, 0])
bars = ax1.barh(use_cases, usefulness, color='lightgreen', edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Usefulness Score', fontsize=12)
ax1.set_title('(a) LLM Usefulness for Physics Tasks', fontsize=13, weight='bold')
ax1.set_xlim([0, 105])
ax1.grid(True, alpha=0.3, axis='x')

for bar, score in zip(bars, usefulness):
    width = bar.get_width()
    ax1.text(width + 2, bar.get_y() + bar.get_height()/2.,
            f'{score}%', ha='left', va='center', fontsize=10, weight='bold')

# (b) Implementation difficulty
ax2 = fig2.add_subplot(gs[1, 0])
bars2 = ax2.barh(use_cases, difficulty, color='lightcoral', edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Implementation Difficulty', fontsize=12)
ax2.set_title('(b) Difficulty Level (lower is easier)', fontsize=13, weight='bold')
ax2.set_xlim([0, 105])
ax2.grid(True, alpha=0.3, axis='x')

for bar, diff in zip(bars2, difficulty):
    width = bar.get_width()
    ax2.text(width + 2, bar.get_y() + bar.get_height()/2.,
            f'{diff}%', ha='left', va='center', fontsize=10, weight='bold')

# (c) Example outputs
ax3 = fig2.add_subplot(gs[2, 0])
ax3.axis('off')

example_outputs = """
EXAMPLE: Physics Problem Solving

User Prompt:
  "A ball is thrown upward with initial velocity 20 m/s. Calculate max height and time to reach it.
   Use g = 9.8 m/s². Show all steps."

Claude Response (simulated):
  Let me solve this step by step:

  Given:
  - Initial velocity (v₀) = 20 m/s (upward)
  - Acceleration (a) = -9.8 m/s² (gravity, downward)
  - Final velocity at max height (v) = 0 m/s

  Step 1: Find time to reach maximum height
  Using: v = v₀ + at
  0 = 20 + (-9.8)t
  t = 20/9.8 = 2.04 seconds

  Step 2: Calculate maximum height
  Using: h = v₀t + ½at²
  h = 20(2.04) + ½(-9.8)(2.04)²
  h = 40.8 - 20.4
  h = 20.4 meters

  Answer: The ball reaches a maximum height of 20.4 meters after 2.04 seconds.
"""

ax3.text(0.5, 0.5, example_outputs, fontsize=9, verticalalignment='center',
        horizontalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax3.set_title('(c) Example: Step-by-Step Problem Solving', fontsize=13, weight='bold')

plt.savefig(f'{output_dir}/04_physics_applications.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_physics_applications.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. API Structure:")
print(f"   - Request: model, messages, temperature, max_tokens")
print(f"   - Response: content, usage, stop_reason")
print(f"   - Messages: role (user/assistant), content")
print(f"\n2. Key Parameters:")
print(f"   - Temperature: 0.0 (deterministic) to 2.0 (creative)")
print(f"   - Max tokens: Response length limit")
print(f"   - Top-p: Alternative to temperature")
print(f"\n3. Prompt Engineering:")
print(f"   - Be specific and clear")
print(f"   - Define role and audience")
print(f"   - Structure with numbered points")
print(f"   - Request format/methodology")
print(f"\n4. Physics Applications:")
print(f"   - Problem solving (very useful)")
print(f"   - Code generation (useful)")
print(f"   - Concept explanation (excellent)")
print(f"   - Always verify outputs!")
print(f"\n5. Best Practices:")
print(f"   - Start with low temperature for factual tasks")
print(f"   - Use system prompts for consistent behavior")
print(f"   - Break complex tasks into steps")
print(f"   - Validate physics/math with independent checks")
print(f"\nNote: This is a simulation. Actual API requires:")
print(f"      1. API key from Anthropic")
print(f"      2. pip install anthropic")
print(f"      3. Handle rate limits and errors")
print("="*70)

plt.show()
