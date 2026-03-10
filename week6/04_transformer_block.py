"""
04. Complete Transformer Block
완전한 트랜스포머 블록

모든 요소를 결합한 완전한 Transformer Encoder Block:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network
- Layer Normalization
- Residual Connections
- 각 컴포넌트의 역할

학습 목표:
1. Transformer block의 전체 구조
2. Residual connection의 중요성
3. Layer Normalization의 역할
4. Feed-Forward Network의 필요성
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
print("Complete Transformer Encoder Block")
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
# Helper Functions
# ============================================================================

def softmax(x, axis=-1):
    """Softmax with numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-6):
    """
    Layer Normalization

    LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta

    Parameters:
    -----------
    x : array (..., d_model)
        Input
    eps : float
        Small constant for numerical stability

    Returns:
    --------
    normalized : array
        Normalized output
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)

    # gamma and beta are learnable parameters (set to 1 and 0 here)
    gamma = 1.0
    beta = 0.0

    return gamma * normalized + beta

def gelu(x):
    """
    GELU activation function (approximation)

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

# ============================================================================
# Transformer Components
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    attention_weights = softmax(scores, axis=-1)
    output = np.dot(attention_weights, V)

    return output, attention_weights

def multi_head_attention(X, W_q_heads, W_k_heads, W_v_heads, W_o, n_heads):
    """
    Multi-Head Attention

    Parameters:
    -----------
    X : array (seq_len, d_model)
        Input
    W_q_heads, W_k_heads, W_v_heads : list of arrays
        Query, Key, Value projections for each head
    W_o : array (n_heads * d_k, d_model)
        Output projection
    n_heads : int
        Number of heads

    Returns:
    --------
    output : array (seq_len, d_model)
        Multi-head attention output
    all_attention : list
        Attention weights for each head
    """
    all_outputs = []
    all_attention = []

    for h in range(n_heads):
        Q = np.dot(X, W_q_heads[h])
        K = np.dot(X, W_k_heads[h])
        V = np.dot(X, W_v_heads[h])

        output_h, attn_h = scaled_dot_product_attention(Q, K, V)
        all_outputs.append(output_h)
        all_attention.append(attn_h)

    # Concatenate heads
    concat_output = np.concatenate(all_outputs, axis=-1)

    # Final projection
    output = np.dot(concat_output, W_o)

    return output, all_attention

def feed_forward_network(x, W1, b1, W2, b2, activation='gelu'):
    """
    Position-wise Feed-Forward Network

    FFN(x) = activation(x·W1 + b1)·W2 + b2

    Parameters:
    -----------
    x : array (seq_len, d_model)
        Input
    W1 : array (d_model, d_ff)
        First layer weights
    b1 : array (d_ff,)
        First layer bias
    W2 : array (d_ff, d_model)
        Second layer weights
    b2 : array (d_model,)
        Second layer bias
    activation : str
        'relu' or 'gelu'

    Returns:
    --------
    output : array (seq_len, d_model)
        FFN output
    """
    # First layer
    hidden = np.dot(x, W1) + b1

    # Activation
    if activation == 'relu':
        hidden = relu(hidden)
    elif activation == 'gelu':
        hidden = gelu(hidden)

    # Second layer
    output = np.dot(hidden, W2) + b2

    return output

def transformer_encoder_block(X, mha_params, ffn_params, n_heads):
    """
    Complete Transformer Encoder Block

    Block(X) = LayerNorm(X + MultiHeadAttention(X))
               + LayerNorm(X + FFN(X))

    Parameters:
    -----------
    X : array (seq_len, d_model)
        Input
    mha_params : dict
        Multi-head attention parameters
    ffn_params : dict
        Feed-forward network parameters
    n_heads : int
        Number of attention heads

    Returns:
    --------
    output : array (seq_len, d_model)
        Block output
    intermediates : dict
        Intermediate values for visualization
    """
    intermediates = {}

    # Multi-Head Self-Attention
    attn_output, attn_weights = multi_head_attention(
        X,
        mha_params['W_q_heads'],
        mha_params['W_k_heads'],
        mha_params['W_v_heads'],
        mha_params['W_o'],
        n_heads
    )
    intermediates['attention_output'] = attn_output
    intermediates['attention_weights'] = attn_weights

    # Add & Norm (first residual connection)
    attn_output_residual = X + attn_output
    intermediates['after_residual_1'] = attn_output_residual

    attn_output_norm = layer_norm(attn_output_residual)
    intermediates['after_norm_1'] = attn_output_norm

    # Feed-Forward Network
    ffn_output = feed_forward_network(
        attn_output_norm,
        ffn_params['W1'],
        ffn_params['b1'],
        ffn_params['W2'],
        ffn_params['b2']
    )
    intermediates['ffn_output'] = ffn_output

    # Add & Norm (second residual connection)
    ffn_output_residual = attn_output_norm + ffn_output
    intermediates['after_residual_2'] = ffn_output_residual

    output = layer_norm(ffn_output_residual)
    intermediates['final_output'] = output

    return output, intermediates

# ============================================================================
# Example 1: Complete Transformer Block
# ============================================================================

print("\n1. Building Complete Transformer Encoder Block...")

np.random.seed(42)

# Parameters
sentence = ["Transformers", "are", "very", "powerful"]
seq_len = len(sentence)
d_model = 64
n_heads = 4
d_k = d_model // n_heads  # 16
d_ff = d_model * 4  # 256 (typically 4x d_model)

print(f"\n   Input: {' '.join(sentence)}")
print(f"   Sequence length: {seq_len}")
print(f"   Model dimension: {d_model}")
print(f"   Number of heads: {n_heads}")
print(f"   Dimension per head: {d_k}")
print(f"   FFN hidden dimension: {d_ff}")

# Create input (word embeddings + positional encoding)
X = np.random.randn(seq_len, d_model) * 0.5

# Initialize Multi-Head Attention parameters
mha_params = {}
limit = np.sqrt(2.0 / d_model)

W_q_heads = []
W_k_heads = []
W_v_heads = []

for _ in range(n_heads):
    W_q_heads.append(np.random.randn(d_model, d_k) * limit)
    W_k_heads.append(np.random.randn(d_model, d_k) * limit)
    W_v_heads.append(np.random.randn(d_model, d_k) * limit)

W_o = np.random.randn(n_heads * d_k, d_model) * limit

mha_params['W_q_heads'] = W_q_heads
mha_params['W_k_heads'] = W_k_heads
mha_params['W_v_heads'] = W_v_heads
mha_params['W_o'] = W_o

# Initialize FFN parameters
ffn_params = {}
limit_ff1 = np.sqrt(2.0 / d_model)
limit_ff2 = np.sqrt(2.0 / d_ff)

ffn_params['W1'] = np.random.randn(d_model, d_ff) * limit_ff1
ffn_params['b1'] = np.zeros(d_ff)
ffn_params['W2'] = np.random.randn(d_ff, d_model) * limit_ff2
ffn_params['b2'] = np.zeros(d_model)

# Forward pass through transformer block
output, intermediates = transformer_encoder_block(X, mha_params, ffn_params, n_heads)

print(f"\n   Output shape: {output.shape}")
print(f"   Intermediate stages captured: {len(intermediates)}")

# ============================================================================
# Visualization 1: Data Flow Through Block
# ============================================================================

print("\n2. Visualizing Data Flow Through Transformer Block...")

fig1 = plt.figure(figsize=(15, 12))
gs = GridSpec(4, 3, figure=fig1, hspace=0.4, wspace=0.3)

# Show first few dimensions for clarity
dims_to_show = 32

stages = [
    ('Input X', X[:, :dims_to_show].T),
    ('After Attention', intermediates['attention_output'][:, :dims_to_show].T),
    ('After Residual 1', intermediates['after_residual_1'][:, :dims_to_show].T),
    ('After Norm 1', intermediates['after_norm_1'][:, :dims_to_show].T),
    ('After FFN', intermediates['ffn_output'][:, :dims_to_show].T),
    ('After Residual 2', intermediates['after_residual_2'][:, :dims_to_show].T),
    ('After Norm 2 (Output)', output[:, :dims_to_show].T),
]

for idx, (stage_name, data) in enumerate(stages):
    if idx >= 9:  # Only 9 subplots
        break

    row = idx // 3
    col = idx % 3

    ax = fig1.add_subplot(gs[row, col])
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(sentence, fontsize=9)
    ax.set_ylabel('Dimension', fontsize=10)
    ax.set_title(f'{stage_name}', fontsize=11, weight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
           transform=ax.transAxes, va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.savefig(f'{output_dir}/04_transformer_dataflow.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_transformer_dataflow.png")

# ============================================================================
# Visualization 2: Attention Patterns
# ============================================================================

print("\n3. Analyzing Multi-Head Attention Patterns...")

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

attn_weights = intermediates['attention_weights']

# Plot each head
for h in range(min(n_heads, 4)):
    row = h // 2
    col = h % 2

    ax = fig2.add_subplot(gs[row, col])
    im = ax.imshow(attn_weights[h], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(sentence, fontsize=10, rotation=45)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(sentence, fontsize=10)
    ax.set_title(f'Head {h+1}', fontsize=12, weight='bold')

    # Add values
    for i in range(seq_len):
        for j in range(seq_len):
            color = "white" if attn_weights[h][i, j] > 0.5 else "black"
            ax.text(j, i, f'{attn_weights[h][i, j]:.2f}',
                   ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Average attention
ax_avg = fig2.add_subplot(gs[0, 2])
avg_attn = np.mean(attn_weights, axis=0)
im_avg = ax_avg.imshow(avg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax_avg.set_xticks(range(seq_len))
ax_avg.set_xticklabels(sentence, fontsize=10, rotation=45)
ax_avg.set_yticks(range(seq_len))
ax_avg.set_yticklabels(sentence, fontsize=10)
ax_avg.set_title('Average Attention', fontsize=12, weight='bold')

for i in range(seq_len):
    for j in range(seq_len):
        color = "white" if avg_attn[i, j] > 0.5 else "black"
        ax_avg.text(j, i, f'{avg_attn[i, j]:.2f}',
                   ha="center", va="center", color=color, fontsize=9)

plt.colorbar(im_avg, ax=ax_avg, fraction=0.046, pad=0.04)

# Attention diversity
ax_div = fig2.add_subplot(gs[1, 2])
attn_std = np.std(attn_weights, axis=0)
im_div = ax_div.imshow(attn_std, cmap='viridis', aspect='auto')
ax_div.set_xticks(range(seq_len))
ax_div.set_xticklabels(sentence, fontsize=10, rotation=45)
ax_div.set_yticks(range(seq_len))
ax_div.set_yticklabels(sentence, fontsize=10)
ax_div.set_title('Attention Diversity (Std)', fontsize=12, weight='bold')
plt.colorbar(im_div, ax=ax_div, fraction=0.046, pad=0.04)

plt.savefig(f'{output_dir}/04_attention_patterns.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_attention_patterns.png")

# ============================================================================
# Example 2: Effect of Residual Connections
# ============================================================================

print("\n4. Analyzing Residual Connections...")

# Compare with and without residual connections
def transformer_no_residual(X, mha_params, ffn_params, n_heads):
    """Transformer block WITHOUT residual connections."""
    attn_output, _ = multi_head_attention(
        X, mha_params['W_q_heads'], mha_params['W_k_heads'],
        mha_params['W_v_heads'], mha_params['W_o'], n_heads
    )
    attn_output_norm = layer_norm(attn_output)

    ffn_output = feed_forward_network(
        attn_output_norm,
        ffn_params['W1'], ffn_params['b1'],
        ffn_params['W2'], ffn_params['b2']
    )
    output = layer_norm(ffn_output)

    return output, attn_output_norm, ffn_output

output_no_res, attn_no_res, ffn_no_res = transformer_no_residual(X, mha_params, ffn_params, n_heads)

fig3 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 4, figure=fig3, hspace=0.3, wspace=0.4)

# (a) Input
ax1 = fig3.add_subplot(gs[0, 0])
im1 = ax1.imshow(X[:, :dims_to_show].T, cmap='RdBu_r', aspect='auto')
ax1.set_xticks(range(seq_len))
ax1.set_xticklabels(sentence, fontsize=9)
ax1.set_ylabel('Dim', fontsize=10)
ax1.set_title('(a) Input', fontsize=11, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) With residual - after first Add&Norm
ax2 = fig3.add_subplot(gs[0, 1])
im2 = ax2.imshow(intermediates['after_norm_1'][:, :dims_to_show].T, cmap='RdBu_r', aspect='auto')
ax2.set_xticks(range(seq_len))
ax2.set_xticklabels(sentence, fontsize=9)
ax2.set_ylabel('Dim', fontsize=10)
ax2.set_title('(b) With Residual\n(After Attn+Norm)', fontsize=11, weight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# (c) Without residual - after first Norm
ax3 = fig3.add_subplot(gs[0, 2])
im3 = ax3.imshow(attn_no_res[:, :dims_to_show].T, cmap='RdBu_r', aspect='auto')
ax3.set_xticks(range(seq_len))
ax3.set_xticklabels(sentence, fontsize=9)
ax3.set_ylabel('Dim', fontsize=10)
ax3.set_title('(c) Without Residual\n(After Attn+Norm)', fontsize=11, weight='bold')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# (d) Difference
ax4 = fig3.add_subplot(gs[0, 3])
diff1 = intermediates['after_norm_1'][:, :dims_to_show].T - attn_no_res[:, :dims_to_show].T
im4 = ax4.imshow(diff1, cmap='RdBu_r', aspect='auto')
ax4.set_xticks(range(seq_len))
ax4.set_xticklabels(sentence, fontsize=9)
ax4.set_ylabel('Dim', fontsize=10)
ax4.set_title('(d) Difference', fontsize=11, weight='bold')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# (e) Final output with residual
ax5 = fig3.add_subplot(gs[1, 0])
im5 = ax5.imshow(output[:, :dims_to_show].T, cmap='RdBu_r', aspect='auto')
ax5.set_xticks(range(seq_len))
ax5.set_xticklabels(sentence, fontsize=9)
ax5.set_ylabel('Dim', fontsize=10)
ax5.set_title('(e) Final (With Residual)', fontsize=11, weight='bold')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# (f) Final output without residual
ax6 = fig3.add_subplot(gs[1, 1])
im6 = ax6.imshow(output_no_res[:, :dims_to_show].T, cmap='RdBu_r', aspect='auto')
ax6.set_xticks(range(seq_len))
ax6.set_xticklabels(sentence, fontsize=9)
ax6.set_ylabel('Dim', fontsize=10)
ax6.set_title('(f) Final (Without Residual)', fontsize=11, weight='bold')
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

# (g) Similarity to input
ax7 = fig3.add_subplot(gs[1, 2])
similarity_with_res = []
similarity_no_res = []

for i in range(seq_len):
    # Cosine similarity
    cos_with = np.dot(output[i], X[i]) / (np.linalg.norm(output[i]) * np.linalg.norm(X[i]))
    cos_no = np.dot(output_no_res[i], X[i]) / (np.linalg.norm(output_no_res[i]) * np.linalg.norm(X[i]))
    similarity_with_res.append(cos_with)
    similarity_no_res.append(cos_no)

x_pos = np.arange(seq_len)
width = 0.35
ax7.bar(x_pos - width/2, similarity_with_res, width, label='With Residual',
       color='green', alpha=0.7, edgecolor='black')
ax7.bar(x_pos + width/2, similarity_no_res, width, label='Without Residual',
       color='red', alpha=0.7, edgecolor='black')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(sentence, fontsize=9)
ax7.set_ylabel('Cosine Similarity to Input', fontsize=10)
ax7.set_title('(g) Input Preservation', fontsize=11, weight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# (h) Gradient flow simulation
ax8 = fig3.add_subplot(gs[1, 3])
# Simulate gradient magnitudes through layers
layers = ['Input', 'Attn', 'Norm1', 'FFN', 'Norm2']
grad_with_res = [1.0, 0.9, 0.85, 0.8, 0.75]  # Gradients preserved
grad_no_res = [1.0, 0.7, 0.4, 0.2, 0.1]  # Vanishing gradients

ax8.plot(layers, grad_with_res, marker='o', linewidth=2, markersize=8,
        label='With Residual', color='green')
ax8.plot(layers, grad_no_res, marker='s', linewidth=2, markersize=8,
        label='Without Residual', color='red')
ax8.set_ylabel('Relative Gradient Magnitude', fontsize=10)
ax8.set_title('(h) Gradient Flow (Simulated)', fontsize=11, weight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)
ax8.set_ylim([0, 1.1])

print(f"\n   Average similarity to input:")
print(f"   With residual:    {np.mean(similarity_with_res):.4f}")
print(f"   Without residual: {np.mean(similarity_no_res):.4f}")
print(f"\n   Residual connections preserve input information!")

plt.savefig(f'{output_dir}/04_residual_effect.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/04_residual_effect.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Transformer Encoder Block Structure:")
print(f"   - Multi-Head Self-Attention")
print(f"   - Add & Normalize (residual + layer norm)")
print(f"   - Feed-Forward Network")
print(f"   - Add & Normalize (residual + layer norm)")
print(f"\n2. Multi-Head Self-Attention:")
print(f"   - Parallel attention with {n_heads} heads")
print(f"   - Each head: dimension {d_k}")
print(f"   - Concatenate + linear projection")
print(f"\n3. Feed-Forward Network:")
print(f"   - Two linear layers with activation")
print(f"   - Typically 4x expansion: {d_model} → {d_ff} → {d_model}")
print(f"   - Applied position-wise (same for all positions)")
print(f"   - Adds capacity and non-linearity")
print(f"\n4. Residual Connections:")
print(f"   - Preserve input information")
print(f"   - Enable gradient flow")
print(f"   - Allow training very deep networks")
print(f"   - Output = LayerNorm(Input + Sublayer(Input))")
print(f"\n5. Layer Normalization:")
print(f"   - Normalize across features (not batch)")
print(f"   - Stabilizes training")
print(f"   - Reduces internal covariate shift")
print(f"   - Mean = 0, Variance = 1")
print(f"\n6. Why This Architecture?")
print(f"   - Attention: capture dependencies")
print(f"   - FFN: add non-linear transformations")
print(f"   - Residual: enable deep stacking")
print(f"   - LayerNorm: stabilize learning")
print("="*70)

plt.show()
