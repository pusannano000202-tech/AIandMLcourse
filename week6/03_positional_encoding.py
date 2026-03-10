"""
03. Positional Encoding
위치 인코딩

Self-Attention은 순서 정보가 없음 - 위치 인코딩으로 해결:
- Sinusoidal positional encoding
- 학습 가능한 positional embedding과 비교
- 위치 정보의 중요성
- 다양한 주파수로 위치 표현

학습 목표:
1. 왜 위치 정보가 필요한지
2. Sinusoidal encoding의 수학적 원리
3. 다른 위치 인코딩 방법들
4. 위치 인코딩의 시각화
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
print("Positional Encoding")
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
# Positional Encoding Functions
# ============================================================================

def get_positional_encoding_sinusoidal(seq_len, d_model):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017)

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters:
    -----------
    seq_len : int
        Sequence length
    d_model : int
        Embedding dimension (must be even)

    Returns:
    --------
    pos_encoding : array (seq_len, d_model)
        Positional encoding
    """
    pos_encoding = np.zeros((seq_len, d_model))

    # Position indices
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)

    # Dimension indices
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Apply sin to even indices
    pos_encoding[:, 0::2] = np.sin(position * div_term)

    # Apply cos to odd indices
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding

def get_positional_encoding_learned(seq_len, d_model):
    """
    Learned positional embeddings (random initialization)

    Parameters:
    -----------
    seq_len : int
        Maximum sequence length
    d_model : int
        Embedding dimension

    Returns:
    --------
    pos_encoding : array (seq_len, d_model)
        Learned positional encoding
    """
    # Random initialization (would be learned during training)
    limit = np.sqrt(3.0 / d_model)
    pos_encoding = np.random.uniform(-limit, limit, (seq_len, d_model))

    return pos_encoding

def get_positional_encoding_linear(seq_len, d_model):
    """
    Simple linear positional encoding

    Parameters:
    -----------
    seq_len : int
        Sequence length
    d_model : int
        Embedding dimension

    Returns:
    --------
    pos_encoding : array (seq_len, d_model)
        Linear positional encoding
    """
    pos_encoding = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    # Broadcast to (seq_len, d_model) by repeating across dimensions
    pos_encoding = np.tile(position / seq_len, (1, d_model))

    return pos_encoding

# ============================================================================
# Example 1: Sinusoidal Positional Encoding
# ============================================================================

print("\n1. Generating Sinusoidal Positional Encoding...")

seq_len = 100
d_model = 128

pos_encoding_sin = get_positional_encoding_sinusoidal(seq_len, d_model)

print(f"\n   Sequence length: {seq_len}")
print(f"   Model dimension: {d_model}")
print(f"   Positional encoding shape: {pos_encoding_sin.shape}")

# ============================================================================
# Visualization 1: Positional Encoding Heatmap
# ============================================================================

print("\n2. Visualizing Positional Encoding...")

fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# (a) Full positional encoding heatmap
ax1 = fig1.add_subplot(gs[0, :])
im1 = ax1.imshow(pos_encoding_sin.T, cmap='RdBu_r', aspect='auto')
ax1.set_xlabel('Position', fontsize=12, weight='bold')
ax1.set_ylabel('Dimension', fontsize=12, weight='bold')
ax1.set_title('(a) Sinusoidal Positional Encoding Heatmap', fontsize=13, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.01, pad=0.04, label='Value')

# (b) Encoding for first few positions
ax2 = fig1.add_subplot(gs[1, 0])
positions_to_show = [0, 1, 5, 10, 20, 50]
for pos in positions_to_show:
    ax2.plot(range(d_model), pos_encoding_sin[pos], label=f'Pos {pos}', linewidth=1.5)
ax2.set_xlabel('Dimension', fontsize=12)
ax2.set_ylabel('Encoding Value', fontsize=12)
ax2.set_title('(b) Encoding at Different Positions', fontsize=13, weight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) Encoding for specific dimensions across positions
ax3 = fig1.add_subplot(gs[1, 1])
dims_to_show = [0, 1, 10, 20, 50, 100]
for dim in dims_to_show:
    ax3.plot(range(seq_len), pos_encoding_sin[:, dim], label=f'Dim {dim}', linewidth=1.5)
ax3.set_xlabel('Position', fontsize=12)
ax3.set_ylabel('Encoding Value', fontsize=12)
ax3.set_title('(c) Different Dimensions Across Positions', fontsize=13, weight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/03_positional_encoding_sinusoidal.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_positional_encoding_sinusoidal.png")

# ============================================================================
# Example 2: Comparing Different Encodings
# ============================================================================

print("\n3. Comparing Different Positional Encoding Methods...")

seq_len_compare = 50
d_model_compare = 64

pos_sin = get_positional_encoding_sinusoidal(seq_len_compare, d_model_compare)
pos_learned = get_positional_encoding_learned(seq_len_compare, d_model_compare)
pos_linear = get_positional_encoding_linear(seq_len_compare, d_model_compare)

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 3, figure=fig2, hspace=0.35, wspace=0.3)

encodings = [
    (pos_sin, 'Sinusoidal', 'RdBu_r'),
    (pos_learned, 'Learned (Random Init)', 'viridis'),
    (pos_linear, 'Linear', 'plasma')
]

for idx, (encoding, name, cmap) in enumerate(encodings):
    # Heatmap
    ax_heat = fig2.add_subplot(gs[idx, 0])
    im = ax_heat.imshow(encoding.T, cmap=cmap, aspect='auto')
    ax_heat.set_ylabel('Dimension', fontsize=11)
    if idx == 2:
        ax_heat.set_xlabel('Position', fontsize=11)
    ax_heat.set_title(f'{name}', fontsize=12, weight='bold')
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Sample positions
    ax_pos = fig2.add_subplot(gs[idx, 1])
    sample_positions = [0, 10, 25, 40]
    for pos in sample_positions:
        ax_pos.plot(range(min(32, d_model_compare)), encoding[pos, :32],
                   label=f'Pos {pos}', linewidth=1.5, marker='o', markersize=3)
    ax_pos.set_ylabel('Value', fontsize=11)
    if idx == 2:
        ax_pos.set_xlabel('Dimension (first 32)', fontsize=11)
    ax_pos.set_title(f'Sample Positions', fontsize=11)
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    # Sample dimensions
    ax_dim = fig2.add_subplot(gs[idx, 2])
    sample_dims = [0, 5, 15, 30]
    for dim in sample_dims:
        if dim < d_model_compare:
            ax_dim.plot(range(seq_len_compare), encoding[:, dim],
                       label=f'Dim {dim}', linewidth=1.5)
    ax_dim.set_ylabel('Value', fontsize=11)
    if idx == 2:
        ax_dim.set_xlabel('Position', fontsize=11)
    ax_dim.set_title(f'Sample Dimensions', fontsize=11)
    ax_dim.legend(fontsize=8)
    ax_dim.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/03_encoding_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_encoding_comparison.png")

# ============================================================================
# Example 3: Relative Position Property
# ============================================================================

print("\n4. Analyzing Relative Position Property...")

# Sinusoidal encoding has the property that PE(pos+k) can be represented
# as a linear function of PE(pos)

seq_len_rel = 50
d_model_rel = 32
pos_enc = get_positional_encoding_sinusoidal(seq_len_rel, d_model_rel)

# Compute similarity (dot product) between positions
similarity_matrix = np.dot(pos_enc, pos_enc.T)

# Normalize to get cosine similarity
norms = np.linalg.norm(pos_enc, axis=1, keepdims=True)
cosine_similarity = similarity_matrix / (norms * norms.T)

fig3 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig3, wspace=0.3)

# (a) Dot product similarity
ax1 = fig3.add_subplot(gs[0, 0])
im1 = ax1.imshow(similarity_matrix, cmap='RdBu_r', aspect='auto')
ax1.set_xlabel('Position', fontsize=12)
ax1.set_ylabel('Position', fontsize=12)
ax1.set_title('(a) Dot Product Similarity', fontsize=13, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) Cosine similarity
ax2 = fig3.add_subplot(gs[0, 1])
im2 = ax2.imshow(cosine_similarity, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax2.set_xlabel('Position', fontsize=12)
ax2.set_ylabel('Position', fontsize=12)
ax2.set_title('(b) Cosine Similarity', fontsize=13, weight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# (c) Similarity vs distance
ax3 = fig3.add_subplot(gs[0, 2])

# For different reference positions
ref_positions = [0, 10, 25, 40]
for ref_pos in ref_positions:
    distances = np.abs(np.arange(seq_len_rel) - ref_pos)
    similarities = cosine_similarity[ref_pos]
    ax3.scatter(distances, similarities, alpha=0.6, s=30, label=f'Ref pos {ref_pos}')

ax3.set_xlabel('Distance from Reference', fontsize=12)
ax3.set_ylabel('Cosine Similarity', fontsize=12)
ax3.set_title('(c) Similarity vs Distance', fontsize=13, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

print(f"\n   Observations:")
print(f"   - Diagonal has highest similarity (same position)")
print(f"   - Similarity decreases with distance")
print(f"   - Pattern is consistent across different reference positions")

plt.savefig(f'{output_dir}/03_relative_position.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_relative_position.png")

# ============================================================================
# Example 4: Effect on Attention (With vs Without Position)
# ============================================================================

print("\n5. Impact on Attention (With vs Without Positional Encoding)...")

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Create a simple sequence
np.random.seed(42)
sentence = ["I", "love", "machine", "learning", "very", "much"]
seq_len_sent = len(sentence)
d_model_sent = 32

# Word embeddings (content only)
word_embeddings = np.random.randn(seq_len_sent, d_model_sent) * 0.5

# Make "love" and "learning" similar (should attend to each other)
word_embeddings[1] = np.random.randn(d_model_sent) * 0.3
word_embeddings[3] = word_embeddings[1] + np.random.randn(d_model_sent) * 0.1

# Get positional encoding
pos_enc_sent = get_positional_encoding_sinusoidal(seq_len_sent, d_model_sent)

# Input with and without positional encoding
input_no_pos = word_embeddings
input_with_pos = word_embeddings + pos_enc_sent

# Simple attention (Q = K = V = input)
def simple_attention(X):
    scores = np.dot(X, X.T) / np.sqrt(d_model_sent)
    attention_weights = softmax(scores, axis=-1)
    return attention_weights

attn_no_pos = simple_attention(input_no_pos)
attn_with_pos = simple_attention(input_with_pos)

fig4 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig4, hspace=0.3, wspace=0.3)

# (a) Word embeddings only
ax1 = fig4.add_subplot(gs[0, 0])
im1 = ax1.imshow(word_embeddings.T, cmap='RdBu_r', aspect='auto')
ax1.set_xticks(range(seq_len_sent))
ax1.set_xticklabels(sentence, fontsize=10)
ax1.set_ylabel('Dimension', fontsize=11)
ax1.set_title('(a) Word Embeddings Only', fontsize=12, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) Positional encoding
ax2 = fig4.add_subplot(gs[0, 1])
im2 = ax2.imshow(pos_enc_sent.T, cmap='RdBu_r', aspect='auto')
ax2.set_xticks(range(seq_len_sent))
ax2.set_xticklabels(sentence, fontsize=10)
ax2.set_ylabel('Dimension', fontsize=11)
ax2.set_title('(b) Positional Encoding', fontsize=12, weight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# (c) Combined (word + position)
ax3 = fig4.add_subplot(gs[0, 2])
im3 = ax3.imshow(input_with_pos.T, cmap='RdBu_r', aspect='auto')
ax3.set_xticks(range(seq_len_sent))
ax3.set_xticklabels(sentence, fontsize=10)
ax3.set_ylabel('Dimension', fontsize=11)
ax3.set_title('(c) Word + Position', fontsize=12, weight='bold')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# (d) Attention without position
ax4 = fig4.add_subplot(gs[1, 0])
im4 = ax4.imshow(attn_no_pos, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax4.set_xticks(range(seq_len_sent))
ax4.set_xticklabels(sentence, fontsize=10, rotation=45)
ax4.set_yticks(range(seq_len_sent))
ax4.set_yticklabels(sentence, fontsize=10)
ax4.set_title('(d) Attention (No Position)', fontsize=12, weight='bold')

for i in range(seq_len_sent):
    for j in range(seq_len_sent):
        color = "white" if attn_no_pos[i, j] > 0.5 else "black"
        ax4.text(j, i, f'{attn_no_pos[i, j]:.2f}',
                ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# (e) Attention with position
ax5 = fig4.add_subplot(gs[1, 1])
im5 = ax5.imshow(attn_with_pos, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax5.set_xticks(range(seq_len_sent))
ax5.set_xticklabels(sentence, fontsize=10, rotation=45)
ax5.set_yticks(range(seq_len_sent))
ax5.set_yticklabels(sentence, fontsize=10)
ax5.set_title('(e) Attention (With Position)', fontsize=12, weight='bold')

for i in range(seq_len_sent):
    for j in range(seq_len_sent):
        color = "white" if attn_with_pos[i, j] > 0.5 else "black"
        ax5.text(j, i, f'{attn_with_pos[i, j]:.2f}',
                ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# (f) Difference
ax6 = fig4.add_subplot(gs[1, 2])
diff = attn_with_pos - attn_no_pos
im6 = ax6.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax6.set_xticks(range(seq_len_sent))
ax6.set_xticklabels(sentence, fontsize=10, rotation=45)
ax6.set_yticks(range(seq_len_sent))
ax6.set_yticklabels(sentence, fontsize=10)
ax6.set_title('(f) Difference (With - Without)', fontsize=12, weight='bold')
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Δ Attention')

print(f"\n   Input sentence: {' '.join(sentence)}")
print(f"\n   Without position: Content-based attention only")
print(f"   With position: Both content and position information")

plt.savefig(f'{output_dir}/03_position_effect.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/03_position_effect.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Why Positional Encoding?")
print(f"   - Self-attention is permutation-invariant")
print(f"   - 'I love you' = 'you love I' without position info")
print(f"   - Position encoding adds order information")
print(f"\n2. Sinusoidal Encoding:")
print(f"   - PE(pos, 2i)   = sin(pos / 10000^(2i/d))")
print(f"   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))")
print(f"   - Different frequencies for different dimensions")
print(f"   - Can represent relative positions")
print(f"\n3. Properties:")
print(f"   - No learning required (deterministic)")
print(f"   - Can extrapolate to longer sequences")
print(f"   - Smooth and continuous")
print(f"   - PE(pos+k) is linear function of PE(pos)")
print(f"\n4. Alternatives:")
print(f"   - Learned embeddings: flexible but limited to training length")
print(f"   - Relative encodings: directly encode distances")
print(f"   - RoPE, ALiBi: modern alternatives")
print(f"\n5. Impact:")
print(f"   - Changes attention patterns significantly")
print(f"   - Allows model to use both content and position")
print(f"   - Essential for sequence tasks")
print("="*70)

plt.show()
