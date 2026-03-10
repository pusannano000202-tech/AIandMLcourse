"""
02. Self-Attention and Multi-Head Attention
셀프 어텐션과 멀티 헤드 어텐션

Self-Attention의 핵심 개념과 Multi-Head의 이점:
- Self-Attention 메커니즘
- Multi-Head Attention 구현
- RNN과의 계산 복잡도 비교
- 병렬 처리의 장점

학습 목표:
1. Self-Attention의 작동 원리
2. Multi-Head가 왜 필요한지
3. RNN 대비 장점 이해
4. 계산 복잡도 O(n²) vs O(n) 비교
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os
import time

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Self-Attention and Multi-Head Attention")
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
# Self-Attention Functions
# ============================================================================

def softmax(x, axis=-1):
    """Softmax with numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v, mask=None):
    """
    Self-Attention: Q, K, V all come from same input X

    Parameters:
    -----------
    X : array (seq_len, d_model)
        Input sequence
    W_q, W_k, W_v : arrays (d_model, d_k)
        Projection matrices
    mask : array (seq_len, seq_len), optional
        Attention mask

    Returns:
    --------
    output : array (seq_len, d_k)
        Attention output
    attention_weights : array (seq_len, seq_len)
        Attention matrix
    """
    # Project to Q, K, V
    Q = np.dot(X, W_q)  # (seq_len, d_k)
    K = np.dot(X, W_k)  # (seq_len, d_k)
    V = np.dot(X, W_v)  # (seq_len, d_k)

    d_k = Q.shape[-1]

    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)

    # Apply mask if provided
    if mask is not None:
        scores = scores + (mask * -1e9)

    # Softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = np.dot(attention_weights, V)

    return output, attention_weights, Q, K, V

def multi_head_attention(X, W_q_heads, W_k_heads, W_v_heads, W_o, n_heads):
    """
    Multi-Head Attention

    Parameters:
    -----------
    X : array (seq_len, d_model)
        Input sequence
    W_q_heads, W_k_heads, W_v_heads : list of arrays
        Projection matrices for each head
    W_o : array (n_heads * d_k, d_model)
        Output projection
    n_heads : int
        Number of attention heads

    Returns:
    --------
    output : array (seq_len, d_model)
        Multi-head attention output
    all_attention_weights : list of arrays
        Attention weights for each head
    """
    all_outputs = []
    all_attention_weights = []

    # Process each head
    for h in range(n_heads):
        output_h, attn_h, _, _, _ = self_attention(X, W_q_heads[h], W_k_heads[h], W_v_heads[h])
        all_outputs.append(output_h)
        all_attention_weights.append(attn_h)

    # Concatenate all heads
    concat_output = np.concatenate(all_outputs, axis=-1)  # (seq_len, n_heads * d_k)

    # Final linear projection
    output = np.dot(concat_output, W_o)  # (seq_len, d_model)

    return output, all_attention_weights

# ============================================================================
# Example 1: Self-Attention Visualization
# ============================================================================

print("\n1. Self-Attention Example...")

np.random.seed(42)

# Input sequence
sentence = ["The", "quick", "brown", "fox", "jumps"]
seq_len = len(sentence)
d_model = 16
d_k = 8

# Create embeddings
X = np.random.randn(seq_len, d_model) * 0.5

# Make "quick brown" and "fox jumps" more similar
X[1] = np.random.randn(d_model) * 0.3
X[2] = X[1] + np.random.randn(d_model) * 0.1
X[3] = np.random.randn(d_model) * 0.3
X[4] = X[3] + np.random.randn(d_model) * 0.1

print(f"\n   Input sequence: {' '.join(sentence)}")
print(f"   Sequence length: {seq_len}")
print(f"   Embedding dimension: {d_model}")
print(f"   Attention dimension: {d_k}")

# Initialize projection matrices with He initialization
limit = np.sqrt(2.0 / d_model)
W_q = np.random.randn(d_model, d_k) * limit
W_k = np.random.randn(d_model, d_k) * limit
W_v = np.random.randn(d_model, d_k) * limit

# Compute self-attention
output, attention_weights, Q, K, V = self_attention(X, W_q, W_k, W_v)

print(f"\n   Output shape: {output.shape}")
print(f"   Attention weights shape: {attention_weights.shape}")

# ============================================================================
# Visualization 1: Self-Attention Components
# ============================================================================

print("\n2. Visualizing Self-Attention Components...")

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)

# (a) Input embeddings (first few dimensions)
ax1 = fig1.add_subplot(gs[0, 0])
im1 = ax1.imshow(X[:, :8].T, cmap='RdBu_r', aspect='auto')
ax1.set_yticks(range(8))
ax1.set_yticklabels([f'd{i}' for i in range(8)], fontsize=9)
ax1.set_xticks(range(seq_len))
ax1.set_xticklabels(sentence, fontsize=10)
ax1.set_ylabel('Dimension', fontsize=11)
ax1.set_title('(a) Input Embeddings X', fontsize=12, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) Query matrix Q
ax2 = fig1.add_subplot(gs[0, 1])
im2 = ax2.imshow(Q.T, cmap='RdBu_r', aspect='auto')
ax2.set_yticks(range(d_k))
ax2.set_yticklabels([f'q{i}' for i in range(d_k)], fontsize=9)
ax2.set_xticks(range(seq_len))
ax2.set_xticklabels(sentence, fontsize=10)
ax2.set_ylabel('Dimension', fontsize=11)
ax2.set_title('(b) Query Q = X·W_q', fontsize=12, weight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# (c) Key matrix K
ax3 = fig1.add_subplot(gs[0, 2])
im3 = ax3.imshow(K.T, cmap='RdBu_r', aspect='auto')
ax3.set_yticks(range(d_k))
ax3.set_yticklabels([f'k{i}' for i in range(d_k)], fontsize=9)
ax3.set_xticks(range(seq_len))
ax3.set_xticklabels(sentence, fontsize=10)
ax3.set_ylabel('Dimension', fontsize=11)
ax3.set_title('(c) Key K = X·W_k', fontsize=12, weight='bold')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# (d) Attention weights
ax4 = fig1.add_subplot(gs[1, 0])
im4 = ax4.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax4.set_yticks(range(seq_len))
ax4.set_yticklabels(sentence, fontsize=10)
ax4.set_xticks(range(seq_len))
ax4.set_xticklabels(sentence, fontsize=10, rotation=45)
ax4.set_ylabel('Query (from)', fontsize=11)
ax4.set_xlabel('Key (to)', fontsize=11)
ax4.set_title('(d) Attention Weights', fontsize=12, weight='bold')

# Add values
for i in range(seq_len):
    for j in range(seq_len):
        color = "white" if attention_weights[i, j] > 0.5 else "black"
        ax4.text(j, i, f'{attention_weights[i, j]:.2f}',
                ha="center", va="center", color=color, fontsize=9)

plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Weight')

# (e) Value matrix V
ax5 = fig1.add_subplot(gs[1, 1])
im5 = ax5.imshow(V.T, cmap='RdBu_r', aspect='auto')
ax5.set_yticks(range(d_k))
ax5.set_yticklabels([f'v{i}' for i in range(d_k)], fontsize=9)
ax5.set_xticks(range(seq_len))
ax5.set_xticklabels(sentence, fontsize=10)
ax5.set_ylabel('Dimension', fontsize=11)
ax5.set_title('(e) Value V = X·W_v', fontsize=12, weight='bold')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# (f) Output
ax6 = fig1.add_subplot(gs[1, 2])
im6 = ax6.imshow(output.T, cmap='RdBu_r', aspect='auto')
ax6.set_yticks(range(d_k))
ax6.set_yticklabels([f'o{i}' for i in range(d_k)], fontsize=9)
ax6.set_xticks(range(seq_len))
ax6.set_xticklabels(sentence, fontsize=10)
ax6.set_ylabel('Dimension', fontsize=11)
ax6.set_title('(f) Output = Attn·V', fontsize=12, weight='bold')
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

plt.savefig(f'{output_dir}/02_self_attention_components.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_self_attention_components.png")

# ============================================================================
# Example 2: Multi-Head Attention
# ============================================================================

print("\n3. Multi-Head Attention Example...")

n_heads = 4
d_k_per_head = d_model // n_heads  # 16 // 4 = 4

print(f"\n   Number of heads: {n_heads}")
print(f"   Dimension per head: {d_k_per_head}")

# Initialize projection matrices for each head
W_q_heads = []
W_k_heads = []
W_v_heads = []

limit = np.sqrt(2.0 / d_model)
for _ in range(n_heads):
    W_q_heads.append(np.random.randn(d_model, d_k_per_head) * limit)
    W_k_heads.append(np.random.randn(d_model, d_k_per_head) * limit)
    W_v_heads.append(np.random.randn(d_model, d_k_per_head) * limit)

# Output projection
W_o = np.random.randn(n_heads * d_k_per_head, d_model) * limit

# Compute multi-head attention
mh_output, all_attn_weights = multi_head_attention(X, W_q_heads, W_k_heads, W_v_heads, W_o, n_heads)

print(f"\n   Multi-head output shape: {mh_output.shape}")

# ============================================================================
# Visualization 2: Multi-Head Attention Patterns
# ============================================================================

print("\n4. Visualizing Multi-Head Attention...")

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 4, figure=fig2, hspace=0.3, wspace=0.4)

# Plot each head's attention
for h in range(n_heads):
    row = h // 2
    col = h % 2

    ax = fig2.add_subplot(gs[row, col])
    im = ax.imshow(all_attn_weights[h], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(sentence, fontsize=9)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(sentence, fontsize=9, rotation=45)
    ax.set_title(f'Head {h+1}', fontsize=12, weight='bold')

    if col == 0:
        ax.set_ylabel('Query', fontsize=10)
    if row == 1:
        ax.set_xlabel('Key', fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Average attention across all heads
ax_avg = fig2.add_subplot(gs[0, 2])
avg_attn = np.mean(all_attn_weights, axis=0)
im_avg = ax_avg.imshow(avg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax_avg.set_yticks(range(seq_len))
ax_avg.set_yticklabels(sentence, fontsize=9)
ax_avg.set_xticks(range(seq_len))
ax_avg.set_xticklabels(sentence, fontsize=9, rotation=45)
ax_avg.set_ylabel('Query', fontsize=10)
ax_avg.set_title('Average Attention', fontsize=12, weight='bold')
plt.colorbar(im_avg, ax=ax_avg, fraction=0.046, pad=0.04)

# Attention diversity (variance across heads)
ax_var = fig2.add_subplot(gs[0, 3])
var_attn = np.var(all_attn_weights, axis=0)
im_var = ax_var.imshow(var_attn, cmap='viridis', aspect='auto')
ax_var.set_yticks(range(seq_len))
ax_var.set_yticklabels(sentence, fontsize=9)
ax_var.set_xticks(range(seq_len))
ax_var.set_xticklabels(sentence, fontsize=9, rotation=45)
ax_var.set_title('Attention Variance', fontsize=12, weight='bold')
plt.colorbar(im_var, ax=ax_var, fraction=0.046, pad=0.04)

# Attention patterns for specific word across heads
ax_word = fig2.add_subplot(gs[1, 2])
word_idx = 2  # "brown"
for h in range(n_heads):
    ax_word.plot(range(seq_len), all_attn_weights[h][word_idx],
                marker='o', label=f'Head {h+1}', linewidth=2, markersize=6)
ax_word.set_xticks(range(seq_len))
ax_word.set_xticklabels(sentence, fontsize=10)
ax_word.set_xlabel('Attending to', fontsize=11)
ax_word.set_ylabel('Attention Weight', fontsize=11)
ax_word.set_title(f'"{sentence[word_idx]}" Attention per Head', fontsize=12, weight='bold')
ax_word.legend(loc='best', fontsize=9)
ax_word.grid(True, alpha=0.3)
ax_word.set_ylim([0, 1])

# Entropy per head
ax_ent = fig2.add_subplot(gs[1, 3])
def entropy(p):
    p = p + 1e-10
    return -np.sum(p * np.log(p), axis=-1)

entropies = []
for h in range(n_heads):
    ent_h = entropy(all_attn_weights[h])
    entropies.append(np.mean(ent_h))

ax_ent.bar(range(1, n_heads+1), entropies, color='skyblue', edgecolor='black', linewidth=1.5)
ax_ent.set_xlabel('Head', fontsize=11)
ax_ent.set_ylabel('Average Entropy', fontsize=11)
ax_ent.set_title('Attention Entropy per Head', fontsize=12, weight='bold')
ax_ent.set_xticks(range(1, n_heads+1))
ax_ent.grid(True, alpha=0.3, axis='y')

print(f"\n   Average entropy per head: {entropies}")
print(f"   Higher entropy = more uniform attention")

plt.savefig(f'{output_dir}/02_multi_head_attention.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_multi_head_attention.png")

# ============================================================================
# Example 3: RNN vs Self-Attention Complexity
# ============================================================================

print("\n5. Comparing RNN vs Self-Attention Complexity...")

def simple_rnn(X, W_h, b_h):
    """Simple RNN forward pass (sequential)"""
    seq_len, d_model = X.shape
    d_hidden = W_h.shape[1]

    h = np.zeros((seq_len, d_hidden))
    h_prev = np.zeros(d_hidden)

    for t in range(seq_len):
        h[t] = np.tanh(np.dot(X[t], W_h) + np.dot(h_prev, W_h) + b_h)
        h_prev = h[t]

    return h

# Test different sequence lengths
seq_lengths = [10, 20, 50, 100, 200]
rnn_times = []
attn_times = []

d_hidden = 16

for seq_len_test in seq_lengths:
    # Generate random input
    X_test = np.random.randn(seq_len_test, d_model)

    # RNN parameters
    W_h_rnn = np.random.randn(d_model, d_hidden) * 0.1
    b_h_rnn = np.zeros(d_hidden)

    # Time RNN
    start = time.time()
    for _ in range(10):  # Multiple runs for averaging
        _ = simple_rnn(X_test, W_h_rnn, b_h_rnn)
    rnn_time = (time.time() - start) / 10
    rnn_times.append(rnn_time)

    # Attention parameters
    W_q_test = np.random.randn(d_model, d_k) * 0.1
    W_k_test = np.random.randn(d_model, d_k) * 0.1
    W_v_test = np.random.randn(d_model, d_k) * 0.1

    # Time Self-Attention
    start = time.time()
    for _ in range(10):
        _ = self_attention(X_test, W_q_test, W_k_test, W_v_test)
    attn_time = (time.time() - start) / 10
    attn_times.append(attn_time)

print(f"\n   Timing results:")
for seq_len_test, rnn_t, attn_t in zip(seq_lengths, rnn_times, attn_times):
    print(f"   Seq length {seq_len_test:3d}: RNN {rnn_t*1000:.2f}ms, Attention {attn_t*1000:.2f}ms")

# Visualization
fig3 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig3, wspace=0.3)

# (a) Computation time comparison
ax1 = fig3.add_subplot(gs[0, 0])
ax1.plot(seq_lengths, np.array(rnn_times)*1000, marker='o', linewidth=2,
         markersize=8, label='RNN (Sequential)', color='red')
ax1.plot(seq_lengths, np.array(attn_times)*1000, marker='s', linewidth=2,
         markersize=8, label='Self-Attention (Parallel)', color='blue')
ax1.set_xlabel('Sequence Length', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('(a) Computation Time', fontsize=13, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# (b) Theoretical complexity
ax2 = fig3.add_subplot(gs[0, 1])
n_range = np.array(seq_lengths)
# RNN: O(n) but sequential
rnn_ops = n_range * d_model * d_hidden
# Attention: O(n²·d) but parallel
attn_ops = n_range**2 * d_k + n_range * d_k * d_model

ax2.plot(n_range, rnn_ops / 1000, marker='o', linewidth=2, markersize=8,
         label='RNN: O(n·d²)', color='red')
ax2.plot(n_range, attn_ops / 1000, marker='s', linewidth=2, markersize=8,
         label='Attention: O(n²·d)', color='blue')
ax2.set_xlabel('Sequence Length n', fontsize=12)
ax2.set_ylabel('Operations (×1000)', fontsize=12)
ax2.set_title('(b) Theoretical Complexity', fontsize=13, weight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# (c) Path length comparison
ax3 = fig3.add_subplot(gs[0, 2])
max_seq = 20
positions = np.arange(max_seq)

# RNN: linear path length
rnn_path = positions

# Attention: constant path (O(1))
attn_path = np.ones(max_seq)

ax3.plot(positions, rnn_path, marker='o', linewidth=2, markersize=6,
        label='RNN: O(n)', color='red')
ax3.plot(positions, attn_path, marker='s', linewidth=2, markersize=6,
        label='Attention: O(1)', color='blue')
ax3.set_xlabel('Distance Between Tokens', fontsize=12)
ax3.set_ylabel('Path Length', fontsize=12)
ax3.set_title('(c) Maximum Path Length', fontsize=13, weight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, max_seq])

plt.savefig(f'{output_dir}/02_rnn_vs_attention.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/02_rnn_vs_attention.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Self-Attention:")
print(f"   - Q, K, V all derived from same input")
print(f"   - Each position attends to all positions")
print(f"   - Captures relationships within sequence")
print(f"\n2. Multi-Head Attention:")
print(f"   - Multiple attention mechanisms in parallel")
print(f"   - Each head can learn different patterns")
print(f"   - Increases model capacity and diversity")
print(f"   - Concatenate + project to combine heads")
print(f"\n3. RNN vs Self-Attention:")
print(f"   - RNN: Sequential (O(n) steps), path length O(n)")
print(f"   - Attention: Parallel (O(1) steps), path length O(1)")
print(f"   - Attention: Direct connections between any positions")
print(f"   - RNN: Better for very long sequences (memory)")
print(f"   - Attention: Better for capturing long-range dependencies")
print(f"\n4. Complexity Trade-offs:")
print(f"   - RNN: O(n·d²) operations, sequential")
print(f"   - Attention: O(n²·d) operations, parallel")
print(f"   - For n < d, Attention is faster")
print(f"   - For n >> d, RNN may be more efficient")
print("="*70)

plt.show()
