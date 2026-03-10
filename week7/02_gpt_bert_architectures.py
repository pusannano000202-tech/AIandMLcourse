"""
02. GPT vs BERT Architectures
GPT와 BERT 아키텍처 비교

Transformer 기반의 두 가지 주요 접근 방식:
- GPT: Autoregressive (다음 단어 예측)
- BERT: Masked Language Model (빈칸 채우기)
- Encoder vs Decoder
- Bidirectional vs Unidirectional

학습 목표:
1. GPT와 BERT의 핵심 차이
2. Decoder-only vs Encoder-only
3. 각 모델의 강점과 약점
4. 사용 사례 비교
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
print("GPT vs BERT Architecture Comparison")
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
# Architecture Comparison
# ============================================================================

print("\n1. Comparing Architectures...")

# Model specifications
models = {
    'GPT-2': {
        'type': 'Decoder-only',
        'layers': 48,
        'd_model': 1600,
        'heads': 25,
        'params': 1.5e9,
        'context': 1024,
        'training': 'Autoregressive'
    },
    'BERT-Large': {
        'type': 'Encoder-only',
        'layers': 24,
        'd_model': 1024,
        'heads': 16,
        'params': 340e6,
        'context': 512,
        'training': 'Masked LM'
    },
    'GPT-3': {
        'type': 'Decoder-only',
        'layers': 96,
        'd_model': 12288,
        'heads': 96,
        'params': 175e9,
        'context': 2048,
        'training': 'Autoregressive'
    }
}

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.4)

# (a) Parameters comparison
ax1 = fig1.add_subplot(gs[0, 0])
model_names = list(models.keys())
param_counts = [models[m]['params'] / 1e9 for m in model_names]
colors = ['skyblue', 'coral', 'lightgreen']

bars = ax1.bar(model_names, param_counts, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Parameters (Billions)', fontsize=12)
ax1.set_title('(a) Model Size Comparison', fontsize=13, weight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_yscale('log')

for bar, count in zip(bars, param_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:.1f}B', ha='center', va='bottom', fontsize=10)

# (b) Architecture depth
ax2 = fig1.add_subplot(gs[0, 1])
layer_counts = [models[m]['layers'] for m in model_names]
bars2 = ax2.bar(model_names, layer_counts, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Layers', fontsize=12)
ax2.set_title('(b) Model Depth', fontsize=13, weight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars2, layer_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}', ha='center', va='bottom', fontsize=10)

# (c) Model width (d_model)
ax3 = fig1.add_subplot(gs[0, 2])
d_models = [models[m]['d_model'] for m in model_names]
bars3 = ax3.bar(model_names, d_models, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Embedding Dimension', fontsize=12)
ax3.set_title('(c) Model Width (d_model)', fontsize=13, weight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar, d in zip(bars3, d_models):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{d}', ha='center', va='bottom', fontsize=10)

# (d) Context window
ax4 = fig1.add_subplot(gs[1, 0])
contexts = [models[m]['context'] for m in model_names]
bars4 = ax4.bar(model_names, contexts, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Context Window (tokens)', fontsize=12)
ax4.set_title('(d) Context Length', fontsize=13, weight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar, ctx in zip(bars4, contexts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{ctx}', ha='center', va='bottom', fontsize=10)

# (e) Architecture type pie chart
ax5 = fig1.add_subplot(gs[1, 1])
arch_types = [models[m]['type'] for m in model_names]
type_counts = {}
for t in arch_types:
    type_counts[t] = type_counts.get(t, 0) + 1

ax5.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.0f%%',
       colors=['skyblue', 'coral'], startangle=90)
ax5.set_title('(e) Architecture Types', fontsize=13, weight='bold')

# (f) Training objective
ax6 = fig1.add_subplot(gs[1, 2])
ax6.axis('off')

info_text = "Training Objectives:\n\n"
info_text += "GPT (Decoder-only):\n"
info_text += "• Autoregressive\n"
info_text += "• Predict next token\n"
info_text += "• Unidirectional (left→right)\n"
info_text += "• Great for generation\n\n"
info_text += "BERT (Encoder-only):\n"
info_text += "• Masked Language Model\n"
info_text += "• Fill in [MASK] tokens\n"
info_text += "• Bidirectional context\n"
info_text += "• Great for understanding"

ax6.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax6.set_title('(f) Training Methods', fontsize=13, weight='bold')

plt.savefig(f'{output_dir}/02_architecture_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_architecture_comparison.png")

# ============================================================================
# Attention Masking
# ============================================================================

print("\n2. Visualizing Attention Masking...")

seq_len = 8

# GPT: Causal masking (lower triangular)
gpt_mask = np.tril(np.ones((seq_len, seq_len)))

# BERT: No masking (full attention)
bert_mask = np.ones((seq_len, seq_len))

fig2 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig2, wspace=0.3)

# (a) GPT causal mask
ax1 = fig2.add_subplot(gs[0, 0])
im1 = ax1.imshow(gpt_mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax1.set_xlabel('Key Position', fontsize=11)
ax1.set_ylabel('Query Position', fontsize=11)
ax1.set_title('(a) GPT Causal Mask\n(Can only see previous tokens)', fontsize=12, weight='bold')

for i in range(seq_len):
    for j in range(seq_len):
        text = ax1.text(j, i, 'See' if gpt_mask[i, j] else 'X',
                       ha="center", va="center",
                       color="black" if gpt_mask[i, j] else "red", fontsize=9)

plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) BERT bidirectional mask
ax2 = fig2.add_subplot(gs[0, 1])
im2 = ax2.imshow(bert_mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax2.set_xlabel('Key Position', fontsize=11)
ax2.set_ylabel('Query Position', fontsize=11)
ax2.set_title('(b) BERT Full Attention\n(Can see all tokens)', fontsize=12, weight='bold')

for i in range(seq_len):
    for j in range(seq_len):
        text = ax2.text(j, i, 'See',
                       ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# (c) Information flow
ax3 = fig2.add_subplot(gs[0, 2])
ax3.axis('off')

flow_text = "Information Flow:\n\n"
flow_text += "GPT (Autoregressive):\n"
flow_text += "  Token 0: sees [0]\n"
flow_text += "  Token 1: sees [0,1]\n"
flow_text += "  Token 2: sees [0,1,2]\n"
flow_text += "  ...\n"
flow_text += "  Token 7: sees [0..7]\n\n"
flow_text += "BERT (Bidirectional):\n"
flow_text += "  All tokens: see all\n"
flow_text += "  Token 0: sees [0..7]\n"
flow_text += "  Token 3: sees [0..7]\n"
flow_text += "  Token 7: sees [0..7]\n"

ax3.text(0.1, 0.5, flow_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax3.set_title('(c) Attention Patterns', fontsize=12, weight='bold')

plt.savefig(f'{output_dir}/02_attention_masking.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_attention_masking.png")

# ============================================================================
# Use Cases
# ============================================================================

print("\n3. Comparing Use Cases...")

fig3 = plt.figure(figsize=(15, 8))
gs = GridSpec(2, 2, figure=fig3, hspace=0.4, wspace=0.3)

# Task performance (simulated scores)
tasks = ['Text\nGeneration', 'Question\nAnswering', 'Sentiment\nAnalysis',
         'Named Entity\nRecognition', 'Summarization']
gpt_scores = [95, 70, 75, 65, 90]
bert_scores = [50, 90, 92, 95, 70]

x = np.arange(len(tasks))
width = 0.35

# (a) Task performance
ax1 = fig3.add_subplot(gs[0, :])
bars1 = ax1.bar(x - width/2, gpt_scores, width, label='GPT', color='skyblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, bert_scores, width, label='BERT', color='coral', edgecolor='black')

ax1.set_ylabel('Performance Score', fontsize=12)
ax1.set_title('(a) Task Performance Comparison', fontsize=13, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 105])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)

# (b) Strengths
ax2 = fig3.add_subplot(gs[1, 0])
ax2.axis('off')

strengths = "STRENGTHS:\n\n"
strengths += "GPT:\n"
strengths += "✓ Excellent at generation\n"
strengths += "✓ Creative writing\n"
strengths += "✓ Code generation\n"
strengths += "✓ Few-shot learning\n"
strengths += "✓ Dialogue systems\n\n"
strengths += "BERT:\n"
strengths += "✓ Deep understanding\n"
strengths += "✓ Classification tasks\n"
strengths += "✓ Named entity recognition\n"
strengths += "✓ Question answering\n"
strengths += "✓ Semantic search"

ax2.text(0.1, 0.5, strengths, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2.set_title('(b) Model Strengths', fontsize=12, weight='bold')

# (c) Limitations
ax3 = fig3.add_subplot(gs[1, 1])
ax3.axis('off')

limitations = "LIMITATIONS:\n\n"
limitations += "GPT:\n"
limitations += "✗ No bidirectional context\n"
limitations += "✗ Weaker at classification\n"
limitations += "✗ May generate nonsense\n"
limitations += "✗ Expensive inference\n"
limitations += "✗ Large model size\n\n"
limitations += "BERT:\n"
limitations += "✗ Cannot generate text\n"
limitations += "✗ Fixed context window\n"
limitations += "✗ Needs fine-tuning\n"
limitations += "✗ Not for creative tasks\n"
limitations += "✗ No streaming output"

ax3.text(0.1, 0.5, limitations, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
ax3.set_title('(c) Model Limitations', fontsize=12, weight='bold')

plt.savefig(f'{output_dir}/02_use_cases.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_use_cases.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Architecture Differences:")
print(f"   - GPT: Decoder-only, Autoregressive")
print(f"   - BERT: Encoder-only, Masked LM")
print(f"\n2. Attention Mechanism:")
print(f"   - GPT: Causal (lower triangular mask)")
print(f"   - BERT: Bidirectional (full attention)")
print(f"\n3. Best Use Cases:")
print(f"   - GPT: Text generation, dialogue, few-shot")
print(f"   - BERT: Classification, NER, QA, search")
print(f"\n4. Model Evolution:")
print(f"   - GPT-2 (2019): 1.5B params")
print(f"   - GPT-3 (2020): 175B params")
print(f"   - GPT-4 (2023): ~1.76T params (rumored)")
print(f"   - BERT-Large: 340M params")
print(f"\n5. Modern Trend:")
print(f"   - Decoder-only models dominating (GPT, Claude, Llama)")
print(f"   - Instruction tuning crucial")
print(f"   - RLHF for alignment")
print("="*70)

plt.show()
