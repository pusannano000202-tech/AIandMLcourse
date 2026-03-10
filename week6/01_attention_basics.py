"""
01. Attention Mechanism Basics
어텐션 메커니즘의 기초

RNN의 한계를 극복하기 위한 Attention의 기본 원리:
- Query, Key, Value 개념
- Dot-product attention 계산
- Attention weights 시각화
- 간단한 예제로 직관 이해

학습 목표:
1. Attention이 왜 필요한지 이해
2. Query-Key-Value 메커니즘 이해
3. Attention weights의 의미
4. Softmax의 역할
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Attention Mechanism Basics")
print("="*70)

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트를 설정합니다."""
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
# Attention Functions
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention을 계산합니다.

    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    Parameters:
    -----------
    Q : array (n_queries, d_k)
        Query 벡터들
    K : array (n_keys, d_k)
        Key 벡터들
    V : array (n_values, d_v)
        Value 벡터들
    mask : array (n_queries, n_keys), optional
        Masking (0으로 마스킹할 위치)

    Returns:
    --------
    output : array (n_queries, d_v)
        Attention 출력
    attention_weights : array (n_queries, n_keys)
        Attention 가중치
    """
    d_k = Q.shape[-1]

    # Q·K^T 계산
    scores = np.dot(Q, K.T)  # (n_queries, n_keys)

    # Scaling (sqrt(d_k)로 나눔)
    scores = scores / np.sqrt(d_k)

    # Masking (옵션)
    if mask is not None:
        scores = scores + (mask * -1e9)

    # Softmax로 가중치 계산
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = np.dot(attention_weights, V)

    return output, attention_weights

def softmax(x, axis=-1):
    """
    Softmax 함수를 계산합니다.

    수치 안정성을 위해 최댓값을 뺍니다.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# ============================================================================
# Example 1: Simple Attention on Sentence
# ============================================================================

print("\n1. Simple Attention Example...")

# 간단한 문장: "I love deep learning"
# 각 단어를 벡터로 표현 (임베딩)
np.random.seed(42)

n_words = 4
d_model = 8  # 임베딩 차원

# 단어 임베딩 (실제로는 학습되지만, 여기서는 랜덤)
words = ["I", "love", "deep", "learning"]
embeddings = np.random.randn(n_words, d_model) * 0.5

print(f"\n   Input: {' '.join(words)}")
print(f"   Embedding dimension: {d_model}")
print(f"   Number of words: {n_words}")

# Query, Key, Value 변환 행렬 (간단히 identity로 시작)
W_q = np.eye(d_model) + np.random.randn(d_model, d_model) * 0.1
W_k = np.eye(d_model) + np.random.randn(d_model, d_model) * 0.1
W_v = np.eye(d_model) + np.random.randn(d_model, d_model) * 0.1

# Q, K, V 계산
Q = np.dot(embeddings, W_q)
K = np.dot(embeddings, W_k)
V = np.dot(embeddings, W_v)

# Attention 계산
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\n   Attention output shape: {output.shape}")
print(f"   Attention weights shape: {attention_weights.shape}")

# ============================================================================
# Visualization 1: Attention Weights Heatmap
# ============================================================================

print("\n2. Visualizing Attention Weights...")

fig1 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig1, wspace=0.3)

# (a) Attention weights heatmap
ax1 = fig1.add_subplot(gs[0, 0])
im1 = ax1.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(n_words))
ax1.set_yticks(range(n_words))
ax1.set_xticklabels(words, fontsize=11)
ax1.set_yticklabels(words, fontsize=11)
ax1.set_xlabel('Key (from)', fontsize=12, weight='bold')
ax1.set_ylabel('Query (to)', fontsize=12, weight='bold')
ax1.set_title('(a) Attention Weights', fontsize=13, weight='bold')

# 각 셀에 값 표시
for i in range(n_words):
    for j in range(n_words):
        text = ax1.text(j, i, f'{attention_weights[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Attention Weight')

# (b) 특정 단어("love")에 대한 attention
ax2 = fig1.add_subplot(gs[0, 1])
word_idx = 1  # "love"
attn_for_word = attention_weights[word_idx]
bars = ax2.bar(words, attn_for_word, color='coral', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Attention Weight', fontsize=12)
ax2.set_title(f'(b) Attention for "{words[word_idx]}"', fontsize=13, weight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

# 각 막대 위에 값 표시
for i, (word, val) in enumerate(zip(words, attn_for_word)):
    ax2.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=10)

# (c) Row-wise sum (should be 1.0)
ax3 = fig1.add_subplot(gs[0, 2])
row_sums = np.sum(attention_weights, axis=1)
ax3.bar(words, row_sums, color='skyblue', edgecolor='black', linewidth=1.5)
ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected (1.0)')
ax3.set_ylabel('Sum of Attention Weights', fontsize=12)
ax3.set_title('(c) Softmax Property Check', fontsize=13, weight='bold')
ax3.set_ylim([0.95, 1.05])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

print(f"   Row sums (should be 1.0): {row_sums}")

plt.savefig(f'{output_dir}/01_attention_weights.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_attention_weights.png")

# ============================================================================
# Example 2: Effect of Scaling Factor
# ============================================================================

print("\n3. Analyzing Scaling Factor (sqrt(d_k))...")

# Query와 Key의 dot product
scores_unscaled = np.dot(Q, K.T)
scores_scaled = scores_unscaled / np.sqrt(d_model)

# Softmax 적용
attn_unscaled = softmax(scores_unscaled, axis=-1)
attn_scaled = softmax(scores_scaled, axis=-1)

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

# (a) Raw scores (unscaled)
ax1 = fig2.add_subplot(gs[0, 0])
im1 = ax1.imshow(scores_unscaled, cmap='RdBu_r', aspect='auto')
ax1.set_xticks(range(n_words))
ax1.set_yticks(range(n_words))
ax1.set_xticklabels(words, fontsize=10)
ax1.set_yticklabels(words, fontsize=10)
ax1.set_title('(a) Q·K^T (Unscaled)', fontsize=12, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) Scaled scores
ax2 = fig2.add_subplot(gs[0, 1])
im2 = ax2.imshow(scores_scaled, cmap='RdBu_r', aspect='auto')
ax2.set_xticks(range(n_words))
ax2.set_yticks(range(n_words))
ax2.set_xticklabels(words, fontsize=10)
ax2.set_yticklabels(words, fontsize=10)
ax2.set_title(f'(b) Q·K^T / sqrt({d_model})', fontsize=12, weight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# (c) Comparison of distributions
ax3 = fig2.add_subplot(gs[0, 2])
ax3.hist(scores_unscaled.flatten(), bins=20, alpha=0.7, label='Unscaled', color='red', edgecolor='black')
ax3.hist(scores_scaled.flatten(), bins=20, alpha=0.7, label='Scaled', color='blue', edgecolor='black')
ax3.set_xlabel('Score Value', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('(c) Score Distributions', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

print(f"   Unscaled scores - mean: {np.mean(scores_unscaled):.3f}, std: {np.std(scores_unscaled):.3f}")
print(f"   Scaled scores   - mean: {np.mean(scores_scaled):.3f}, std: {np.std(scores_scaled):.3f}")

# (d) Attention weights (unscaled)
ax4 = fig2.add_subplot(gs[1, 0])
im4 = ax4.imshow(attn_unscaled, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax4.set_xticks(range(n_words))
ax4.set_yticks(range(n_words))
ax4.set_xticklabels(words, fontsize=10)
ax4.set_yticklabels(words, fontsize=10)
ax4.set_title('(d) Attention (Unscaled)', fontsize=12, weight='bold')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# (e) Attention weights (scaled)
ax5 = fig2.add_subplot(gs[1, 1])
im5 = ax5.imshow(attn_scaled, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax5.set_xticks(range(n_words))
ax5.set_yticks(range(n_words))
ax5.set_xticklabels(words, fontsize=10)
ax5.set_yticklabels(words, fontsize=10)
ax5.set_title('(e) Attention (Scaled)', fontsize=12, weight='bold')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# (f) Entropy comparison
ax6 = fig2.add_subplot(gs[1, 2])

def entropy(p):
    """Calculate entropy: -sum(p * log(p))"""
    p = p + 1e-10  # numerical stability
    return -np.sum(p * np.log(p), axis=-1)

entropy_unscaled = entropy(attn_unscaled)
entropy_scaled = entropy(attn_scaled)

x = np.arange(n_words)
width = 0.35
ax6.bar(x - width/2, entropy_unscaled, width, label='Unscaled', color='red', alpha=0.7, edgecolor='black')
ax6.bar(x + width/2, entropy_scaled, width, label='Scaled', color='blue', alpha=0.7, edgecolor='black')
ax6.set_xticks(x)
ax6.set_xticklabels(words, fontsize=10)
ax6.set_ylabel('Entropy', fontsize=11)
ax6.set_title('(f) Attention Entropy', fontsize=12, weight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

print(f"   Mean entropy (unscaled): {np.mean(entropy_unscaled):.3f}")
print(f"   Mean entropy (scaled):   {np.mean(entropy_scaled):.3f}")
print(f"   Higher entropy = more uniform distribution")

plt.savefig(f'{output_dir}/01_scaling_effect.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_scaling_effect.png")

# ============================================================================
# Example 3: Longer Sequence
# ============================================================================

print("\n4. Attention on Longer Sequence...")

# 더 긴 문장
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_words_long = len(sentence)

# 임베딩
embeddings_long = np.random.randn(n_words_long, d_model) * 0.5

# 특별히 "cat"와 "sat"이 유사하도록, "the" 두 개가 유사하도록 설정
embeddings_long[1] = np.random.randn(d_model) * 0.3  # cat
embeddings_long[2] = embeddings_long[1] + np.random.randn(d_model) * 0.1  # sat (similar to cat)
embeddings_long[0] = np.random.randn(d_model) * 0.3  # the
embeddings_long[4] = embeddings_long[0] + np.random.randn(d_model) * 0.1  # the (similar)

# Q, K, V 계산
Q_long = np.dot(embeddings_long, W_q)
K_long = np.dot(embeddings_long, W_k)
V_long = np.dot(embeddings_long, W_v)

# Attention
output_long, attn_long = scaled_dot_product_attention(Q_long, K_long, V_long)

fig3 = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 2, figure=fig3, wspace=0.3)

# (a) Full attention matrix
ax1 = fig3.add_subplot(gs[0, 0])
im1 = ax1.imshow(attn_long, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(n_words_long))
ax1.set_yticks(range(n_words_long))
ax1.set_xticklabels(sentence, fontsize=11)
ax1.set_yticklabels(sentence, fontsize=11)
ax1.set_xlabel('Key', fontsize=12, weight='bold')
ax1.set_ylabel('Query', fontsize=12, weight='bold')
ax1.set_title('(a) Attention Matrix for Full Sentence', fontsize=13, weight='bold')

# 값 표시
for i in range(n_words_long):
    for j in range(n_words_long):
        text = ax1.text(j, i, f'{attn_long[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if attn_long[i, j] > 0.5 else "black",
                       fontsize=9)

plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Attention')

# (b) Attention patterns for each word
ax2 = fig3.add_subplot(gs[0, 1])
for i in range(n_words_long):
    ax2.plot(range(n_words_long), attn_long[i], marker='o',
             label=sentence[i], linewidth=2, markersize=6)

ax2.set_xticks(range(n_words_long))
ax2.set_xticklabels(sentence, fontsize=11)
ax2.set_xlabel('Attending to', fontsize=12, weight='bold')
ax2.set_ylabel('Attention Weight', fontsize=12, weight='bold')
ax2.set_title('(b) Attention Patterns Per Word', fontsize=13, weight='bold')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.savefig(f'{output_dir}/01_longer_sequence.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_longer_sequence.png")

# 분석
print(f"\n   Sentence: {' '.join(sentence)}")
print(f"\n   Observations:")
for i, word in enumerate(sentence):
    top_attend = np.argmax(attn_long[i])
    print(f"   '{word}' attends most to '{sentence[top_attend]}' (weight: {attn_long[i, top_attend]:.3f})")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Attention Mechanism:")
print(f"   - Query: 'What am I looking for?'")
print(f"   - Key: 'What do I contain?'")
print(f"   - Value: 'What do I actually provide?'")
print(f"   - Attention = weighted sum of Values")
print(f"\n2. Scaled Dot-Product:")
print(f"   - Score = Q·K^T / sqrt(d_k)")
print(f"   - Scaling prevents vanishing gradients in softmax")
print(f"   - Keeps scores in reasonable range")
print(f"\n3. Softmax Properties:")
print(f"   - Converts scores to probabilities (sum = 1)")
print(f"   - High scores → high attention weights")
print(f"   - Differentiable (can backpropagate)")
print(f"\n4. Why Attention?")
print(f"   - Direct connection between any two positions")
print(f"   - Parallel computation (unlike RNN)")
print(f"   - Can attend to relevant information regardless of distance")
print("="*70)

plt.show()
