"""
03. Pre-training and Fine-tuning
사전 학습과 미세 조정

LLM 학습의 두 단계:
- Pre-training: 대규모 데이터로 언어 이해
- Fine-tuning: 특정 작업에 맞게 조정
- Transfer Learning의 힘
- RLHF (Reinforcement Learning from Human Feedback)

학습 목표:
1. Pre-training의 목적과 방법
2. Fine-tuning 전략
3. Few-shot vs Zero-shot Learning
4. Instruction Tuning과 RLHF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os

output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Pre-training and Fine-tuning")
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
# Training Pipeline
# ============================================================================

print("\n1. LLM Training Pipeline...")

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 2, figure=fig1, hspace=0.4, wspace=0.3)

# (a) Data volume comparison
ax1 = fig1.add_subplot(gs[0, 0])
stages = ['Pre-training', 'Fine-tuning', 'RLHF']
data_sizes = [300e9, 1e6, 10e3]  # tokens
colors_stage = ['lightblue', 'lightgreen', 'lightcoral']

bars = ax1.bar(stages, data_sizes, color=colors_stage, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Training Data (tokens)', fontsize=12)
ax1.set_title('(a) Data Requirements', fontsize=13, weight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')

for bar, size in zip(bars, data_sizes):
    height = bar.get_height()
    if size >= 1e9:
        label = f'{size/1e9:.0f}B'
    elif size >= 1e6:
        label = f'{size/1e6:.0f}M'
    else:
        label = f'{size/1e3:.0f}K'
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom', fontsize=10, weight='bold')

# (b) Training time comparison
ax2 = fig1.add_subplot(gs[0, 1])
training_days = [90, 3, 1]  # days

bars2 = ax2.bar(stages, training_days, color=colors_stage, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Training Time (days)', fontsize=12)
ax2.set_title('(b) Training Duration', fontsize=13, weight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

for bar, days in zip(bars2, training_days):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{days}d', ha='center', va='bottom', fontsize=10, weight='bold')

# (c) Cost comparison
ax3 = fig1.add_subplot(gs[1, 0])
costs_million = [10, 0.1, 0.01]  # million USD

bars3 = ax3.bar(stages, costs_million, color=colors_stage, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Cost (Million USD)', fontsize=12)
ax3.set_title('(c) Training Cost (estimated)', fontsize=13, weight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')

for bar, cost in zip(bars3, costs_million):
    height = bar.get_height()
    if cost >= 1:
        label = f'${cost:.0f}M'
    else:
        label = f'${cost*1000:.0f}K'
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom', fontsize=10, weight='bold')

# (d) Performance improvement
ax4 = fig1.add_subplot(gs[1, 1])
performance = [60, 85, 95]  # % accuracy on benchmark

ax4.plot(stages, performance, marker='o', linewidth=3, markersize=12,
        color='green', markerfacecolor='lightgreen', markeredgecolor='black', markeredgewidth=2)
ax4.set_ylabel('Performance Score', fontsize=12)
ax4.set_title('(d) Performance Progression', fontsize=13, weight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([50, 100])

for stage, perf in zip(stages, performance):
    ax4.annotate(f'{perf}%', xy=(stage, perf), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=11, weight='bold')

# (e) Learning objectives
ax5 = fig1.add_subplot(gs[2, :])
ax5.axis('off')

objectives_text = """
TRAINING OBJECTIVES:

Pre-training:
  • Task: Next token prediction (GPT) or Masked language modeling (BERT)
  • Data: Web scrapes, books, Wikipedia, code repositories
  • Goal: Learn general language understanding and world knowledge
  • Output: Base model with broad capabilities

Fine-tuning:
  • Task: Specific downstream tasks (QA, summarization, classification)
  • Data: Task-specific labeled datasets
  • Goal: Adapt model to particular use case
  • Output: Specialized model for specific domain

RLHF (Reinforcement Learning from Human Feedback):
  • Task: Align model behavior with human preferences
  • Data: Human rankings of model outputs
  • Goal: Make model helpful, harmless, and honest
  • Output: Assistant-like model (ChatGPT, Claude)
"""

ax5.text(0.5, 0.5, objectives_text, fontsize=10, verticalalignment='center',
        horizontalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax5.set_title('(e) Training Objectives at Each Stage', fontsize=13, weight='bold')

plt.savefig(f'{output_dir}/03_training_pipeline.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_training_pipeline.png")

# ============================================================================
# Learning Paradigms
# ============================================================================

print("\n2. Comparing Learning Paradigms...")

fig2 = plt.figure(figsize=(15, 8))
gs = GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.3)

# (a) Performance vs data
ax1 = fig2.add_subplot(gs[0, 0])

data_amounts = np.logspace(0, 4, 50)  # 1 to 10K examples
zero_shot = np.ones_like(data_amounts) * 60
few_shot = 60 + 20 * (1 - np.exp(-data_amounts / 100))
fine_tune = 60 + 35 * (1 - np.exp(-data_amounts / 1000))

ax1.plot(data_amounts, zero_shot, label='Zero-shot', linewidth=2, linestyle='--')
ax1.plot(data_amounts, few_shot, label='Few-shot', linewidth=2)
ax1.plot(data_amounts, fine_tune, label='Fine-tuning', linewidth=2)

ax1.set_xlabel('Number of Examples', fontsize=12)
ax1.set_ylabel('Performance (%)', fontsize=12)
ax1.set_title('(a) Performance vs Training Data', fontsize=13, weight='bold')
ax1.set_xscale('log')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([50, 100])

# (b) Learning paradigms comparison
ax2 = fig2.add_subplot(gs[0, 1])
paradigms = ['Zero-shot', 'Few-shot\n(5-10 ex)', 'Fine-tuning\n(1000+ ex)']
setup_time = [0, 1, 60]  # minutes
accuracy = [60, 75, 95]  # %

x = np.arange(len(paradigms))
width = 0.35

bars1 = ax2.bar(x - width/2, setup_time, width, label='Setup Time (min)',
               color='skyblue', edgecolor='black')
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, accuracy, width, label='Accuracy (%)',
                    color='coral', edgecolor='black')

ax2.set_xlabel('Learning Paradigm', fontsize=12)
ax2.set_ylabel('Setup Time (min)', fontsize=12, color='blue')
ax2_twin.set_ylabel('Accuracy (%)', fontsize=12, color='red')
ax2.set_title('(b) Trade-offs', fontsize=13, weight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(paradigms, fontsize=10)
ax2.tick_params(axis='y', labelcolor='blue')
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2.grid(True, alpha=0.3, axis='y')

# (c) RLHF process
ax3 = fig2.add_subplot(gs[1, :])
ax3.axis('off')

rlhf_text = """
RLHF PROCESS (Reinforcement Learning from Human Feedback):

Step 1: Supervised Fine-tuning (SFT)
  └─> Train on high-quality human demonstrations
  └─> Creates initial helpful assistant

Step 2: Reward Model Training
  └─> Humans rank multiple model outputs for same prompt
  └─> Train reward model to predict human preferences
  └─> Learns what "good" outputs look like

Step 3: Reinforcement Learning (PPO)
  └─> Generate outputs and get rewards from reward model
  └─> Update policy to maximize expected reward
  └─> Balance between reward and staying close to SFT model (KL divergence)

Result: Model that is:
  ✓ Helpful (follows instructions)
  ✓ Harmless (avoids harmful content)
  ✓ Honest (admits uncertainty)
"""

ax3.text(0.5, 0.5, rlhf_text, fontsize=10, verticalalignment='center',
        horizontalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax3.set_title('(c) RLHF: Making LLMs into Assistants', fontsize=13, weight='bold')

plt.savefig(f'{output_dir}/03_learning_paradigms.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_learning_paradigms.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Pre-training:")
print(f"   - Massive datasets (300B+ tokens)")
print(f"   - Expensive (millions of dollars)")
print(f"   - Creates foundation model")
print(f"   - Done once, used many times")
print(f"\n2. Fine-tuning:")
print(f"   - Smaller datasets (1K-1M examples)")
print(f"   - Much cheaper (thousands of dollars)")
print(f"   - Task-specific adaptation")
print(f"   - Quick iteration")
print(f"\n3. Learning Paradigms:")
print(f"   - Zero-shot: No training data needed")
print(f"   - Few-shot: 5-10 examples in prompt")
print(f"   - Fine-tuning: 1000+ examples for training")
print(f"\n4. RLHF:")
print(f"   - Aligns model with human values")
print(f"   - Uses human feedback as reward")
print(f"   - Critical for assistant behavior")
print(f"   - Makes models helpful and safe")
print(f"\n5. Modern Approach (2024):")
print(f"   - Pre-train on diverse data")
print(f"   - Instruction tuning (SFT)")
print(f"   - RLHF for alignment")
print(f"   - Continuous improvement from user feedback")
print("="*70)

plt.show()
