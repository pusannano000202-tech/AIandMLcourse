"""
05. Sequence Modeling with Transformer
트랜스포머를 이용한 시퀀스 모델링

실제 시퀀스 모델링 작업에 Transformer 적용:
- 간단한 시계열 예측
- Transformer vs RNN 비교
- Attention visualization으로 학습 분석
- 학습 곡선과 성능 비교

학습 목표:
1. Transformer를 실제 문제에 적용
2. RNN과 성능 비교
3. Attention map으로 모델 해석
4. 학습 과정 이해
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
print("Sequence Modeling: Transformer vs RNN")
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

def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-6):
    """Layer Normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def gelu(x):
    """GELU activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)

# ============================================================================
# Data Generation
# ============================================================================

def generate_sine_sequence(n_samples, seq_len, n_features=1):
    """
    Generate synthetic sine wave data for sequence prediction

    Task: Predict next value given previous seq_len values

    Parameters:
    -----------
    n_samples : int
        Number of sequences
    seq_len : int
        Length of input sequence
    n_features : int
        Number of features per timestep

    Returns:
    --------
    X : array (n_samples, seq_len, n_features)
        Input sequences
    y : array (n_samples, n_features)
        Target values (next timestep)
    """
    X = np.zeros((n_samples, seq_len, n_features))
    y = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)

        # Generate sequence
        t = np.linspace(0, 10, seq_len + 1)
        signal = np.sin(freq * t + phase)

        X[i, :, 0] = signal[:-1]
        y[i, 0] = signal[-1]

    return X, y

def generate_multi_sine_sequence(n_samples, seq_len, n_features=3):
    """
    Generate multiple overlapping sine waves

    More complex pattern for testing model capacity
    """
    X = np.zeros((n_samples, seq_len, n_features))
    y = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        t = np.linspace(0, 10, seq_len + 1)

        for feat in range(n_features):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.5)

            signal = amplitude * np.sin(freq * t + phase)
            X[i, :, feat] = signal[:-1]
            y[i, feat] = signal[-1]

    return X, y

# ============================================================================
# Transformer Components
# ============================================================================

def get_positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    pos_encoding = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding

def scaled_dot_product_attention(Q, K, V):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    output = np.dot(attention_weights, V)
    return output, attention_weights

def multi_head_attention_forward(X, W_q_heads, W_k_heads, W_v_heads, W_o, n_heads):
    """Multi-head attention forward pass."""
    all_outputs = []
    all_attention = []

    for h in range(n_heads):
        Q = np.dot(X, W_q_heads[h])
        K = np.dot(X, W_k_heads[h])
        V = np.dot(X, W_v_heads[h])

        output_h, attn_h = scaled_dot_product_attention(Q, K, V)
        all_outputs.append(output_h)
        all_attention.append(attn_h)

    concat_output = np.concatenate(all_outputs, axis=-1)
    output = np.dot(concat_output, W_o)

    return output, all_attention

def feed_forward(x, W1, b1, W2, b2):
    """Feed-forward network."""
    hidden = gelu(np.dot(x, W1) + b1)
    output = np.dot(hidden, W2) + b2
    return output

class SimpleTransformer:
    """Simple Transformer for sequence prediction."""

    def __init__(self, seq_len, d_input, d_model, n_heads, d_ff, d_output):
        self.seq_len = seq_len
        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_output = d_output
        self.d_k = d_model // n_heads

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize all parameters with He initialization."""
        limit = np.sqrt(2.0 / self.d_model)

        # Input projection
        self.W_input = np.random.randn(self.d_input, self.d_model) * np.sqrt(2.0 / self.d_input)

        # Multi-head attention
        self.W_q_heads = [np.random.randn(self.d_model, self.d_k) * limit for _ in range(self.n_heads)]
        self.W_k_heads = [np.random.randn(self.d_model, self.d_k) * limit for _ in range(self.n_heads)]
        self.W_v_heads = [np.random.randn(self.d_model, self.d_k) * limit for _ in range(self.n_heads)]
        self.W_o = np.random.randn(self.n_heads * self.d_k, self.d_model) * limit

        # Feed-forward
        self.W1 = np.random.randn(self.d_model, self.d_ff) * limit
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, self.d_model) * np.sqrt(2.0 / self.d_ff)
        self.b2 = np.zeros(self.d_model)

        # Output projection
        self.W_out = np.random.randn(self.d_model, self.d_output) * np.sqrt(2.0 / self.d_model)
        self.b_out = np.zeros(self.d_output)

        # Positional encoding
        self.pos_encoding = get_positional_encoding(self.seq_len, self.d_model)

    def forward(self, X):
        """
        Forward pass

        Parameters:
        -----------
        X : array (seq_len, d_input)
            Input sequence

        Returns:
        --------
        output : array (d_output,)
            Prediction
        attention_weights : list
            Attention weights for visualization
        """
        # Input projection
        X_proj = np.dot(X, self.W_input)  # (seq_len, d_model)

        # Add positional encoding
        X_pos = X_proj + self.pos_encoding

        # Multi-head attention
        attn_output, attention_weights = multi_head_attention_forward(
            X_pos, self.W_q_heads, self.W_k_heads, self.W_v_heads, self.W_o, self.n_heads
        )

        # Residual + LayerNorm
        X_attn = layer_norm(X_pos + attn_output)

        # Feed-forward
        ffn_output = feed_forward(X_attn, self.W1, self.b1, self.W2, self.b2)

        # Residual + LayerNorm
        X_ffn = layer_norm(X_attn + ffn_output)

        # Pool (use last position)
        pooled = X_ffn[-1]  # (d_model,)

        # Output projection
        output = np.dot(pooled, self.W_out) + self.b_out

        return output, attention_weights

class SimpleRNN:
    """Simple RNN for comparison."""

    def __init__(self, d_input, d_hidden, d_output):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output

        # Initialize parameters with Xavier initialization
        limit_h = np.sqrt(6.0 / (d_input + d_hidden))
        limit_o = np.sqrt(6.0 / (d_hidden + d_output))

        self.W_xh = np.random.uniform(-limit_h, limit_h, (d_input, d_hidden))
        self.W_hh = np.random.uniform(-limit_h, limit_h, (d_hidden, d_hidden))
        self.b_h = np.zeros(d_hidden)

        self.W_out = np.random.uniform(-limit_o, limit_o, (d_hidden, d_output))
        self.b_out = np.zeros(d_output)

    def forward(self, X):
        """
        Forward pass

        Parameters:
        -----------
        X : array (seq_len, d_input)
            Input sequence

        Returns:
        --------
        output : array (d_output,)
            Prediction
        """
        h = np.zeros(self.d_hidden)

        for t in range(X.shape[0]):
            h = np.tanh(np.dot(X[t], self.W_xh) + np.dot(h, self.W_hh) + self.b_h)

        output = np.dot(h, self.W_out) + self.b_out

        return output

# ============================================================================
# Example 1: Generate Data
# ============================================================================

print("\n1. Generating Sequence Data...")

np.random.seed(42)

seq_len = 20
n_features = 1

# Training data
n_train = 500
X_train, y_train = generate_sine_sequence(n_train, seq_len, n_features)

# Test data
n_test = 100
X_test, y_test = generate_sine_sequence(n_test, seq_len, n_features)

print(f"\n   Training samples: {n_train}")
print(f"   Test samples: {n_test}")
print(f"   Sequence length: {seq_len}")
print(f"   Features: {n_features}")
print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")

# Visualize sample sequences
fig1 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig1, wspace=0.3)

for i in range(3):
    ax = fig1.add_subplot(gs[0, i])
    ax.plot(range(seq_len), X_train[i, :, 0], 'b-', linewidth=2, label='Input sequence')
    ax.plot(seq_len, y_train[i, 0], 'ro', markersize=10, label='Target (next value)')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Sample {i+1}', fontsize=12, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/05_sample_sequences.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_sample_sequences.png")

# ============================================================================
# Example 2: Initialize Models
# ============================================================================

print("\n2. Initializing Models...")

# Transformer
d_model = 32
n_heads = 4
d_ff = 128
transformer = SimpleTransformer(seq_len, n_features, d_model, n_heads, d_ff, n_features)

# RNN
d_hidden = 64
rnn = SimpleRNN(n_features, d_hidden, n_features)

print(f"\n   Transformer:")
print(f"   - Model dimension: {d_model}")
print(f"   - Attention heads: {n_heads}")
print(f"   - FFN dimension: {d_ff}")
print(f"\n   RNN:")
print(f"   - Hidden dimension: {d_hidden}")

# ============================================================================
# Example 3: Test Forward Pass
# ============================================================================

print("\n3. Testing Forward Pass...")

# Single example
x_sample = X_train[0]  # (seq_len, n_features)
y_sample = y_train[0]  # (n_features,)

# Transformer forward
y_pred_transformer, attn_weights = transformer.forward(x_sample)
loss_transformer = mse_loss(y_sample, y_pred_transformer)

# RNN forward
y_pred_rnn = rnn.forward(x_sample)
loss_rnn = mse_loss(y_sample, y_pred_rnn)

print(f"\n   Sample prediction:")
print(f"   True value: {y_sample[0]:.4f}")
print(f"   Transformer: {y_pred_transformer[0]:.4f} (loss: {loss_transformer:.4f})")
print(f"   RNN: {y_pred_rnn[0]:.4f} (loss: {loss_rnn:.4f})")

# ============================================================================
# Visualization 1: Attention Patterns
# ============================================================================

print("\n4. Visualizing Attention Patterns...")

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

# Test on multiple samples
sample_indices = [0, 10, 25]

for idx, sample_idx in enumerate(sample_indices):
    x = X_train[sample_idx]
    y_pred, attn = transformer.forward(x)

    # Average attention across heads
    avg_attn = np.mean(attn, axis=0)

    # (a) Attention heatmap
    ax1 = fig2.add_subplot(gs[0, idx])
    im1 = ax1.imshow(avg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xlabel('Key Position', fontsize=10)
    ax1.set_ylabel('Query Position', fontsize=10)
    ax1.set_title(f'Sample {sample_idx}: Avg Attention', fontsize=11, weight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # (b) Input sequence with attention from last position
    ax2 = fig2.add_subplot(gs[1, idx])
    ax2.plot(range(seq_len), x[:, 0], 'b-', linewidth=2, label='Input')
    ax2.plot(seq_len, y_train[sample_idx, 0], 'go', markersize=10, label='True next')
    ax2.plot(seq_len, y_pred[0], 'r^', markersize=10, label='Predicted')

    # Show attention from last position
    ax2_twin = ax2.twinx()
    ax2_twin.bar(range(seq_len), avg_attn[-1], alpha=0.3, color='orange',
                 label='Attention weights')
    ax2_twin.set_ylabel('Attention Weight', fontsize=10, color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.set_ylim([0, 1])

    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_title(f'Prediction with Attention', fontsize=11, weight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/05_attention_patterns.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_attention_patterns.png")

# ============================================================================
# Example 4: Evaluate on Test Set
# ============================================================================

print("\n5. Evaluating on Test Set...")

# Transformer predictions
transformer_losses = []
for i in range(n_test):
    y_pred, _ = transformer.forward(X_test[i])
    loss = mse_loss(y_test[i], y_pred)
    transformer_losses.append(loss)

# RNN predictions
rnn_losses = []
for i in range(n_test):
    y_pred = rnn.forward(X_test[i])
    loss = mse_loss(y_test[i], y_pred)
    rnn_losses.append(loss)

print(f"\n   Transformer - Mean test loss: {np.mean(transformer_losses):.4f}")
print(f"   RNN - Mean test loss: {np.mean(rnn_losses):.4f}")

# ============================================================================
# Visualization 2: Performance Comparison
# ============================================================================

print("\n6. Comparing Model Performance...")

fig3 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig3, wspace=0.3)

# (a) Loss distribution
ax1 = fig3.add_subplot(gs[0, 0])
ax1.hist(transformer_losses, bins=20, alpha=0.7, label='Transformer',
        color='blue', edgecolor='black')
ax1.hist(rnn_losses, bins=20, alpha=0.7, label='RNN',
        color='red', edgecolor='black')
ax1.set_xlabel('MSE Loss', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('(a) Test Loss Distribution', fontsize=13, weight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# (b) Box plot
ax2 = fig3.add_subplot(gs[0, 1])
box_data = [transformer_losses, rnn_losses]
bp = ax2.boxplot(box_data, labels=['Transformer', 'RNN'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('MSE Loss', fontsize=12)
ax2.set_title('(b) Loss Distribution (Box Plot)', fontsize=13, weight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# (c) Sample predictions
ax3 = fig3.add_subplot(gs[0, 2])
sample_ids = range(20)
transformer_sample_losses = transformer_losses[:20]
rnn_sample_losses = rnn_losses[:20]

ax3.plot(sample_ids, transformer_sample_losses, 'b-o', linewidth=2,
        markersize=6, label='Transformer')
ax3.plot(sample_ids, rnn_sample_losses, 'r-s', linewidth=2,
        markersize=6, label='RNN')
ax3.set_xlabel('Sample Index', fontsize=12)
ax3.set_ylabel('MSE Loss', fontsize=12)
ax3.set_title('(c) Per-Sample Losses (First 20)', fontsize=13, weight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.savefig(f'{output_dir}/05_performance_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_performance_comparison.png")

# ============================================================================
# Example 5: Multi-Head Analysis
# ============================================================================

print("\n7. Analyzing Multi-Head Attention...")

# Pick an interesting example
x_analysis = X_test[5]
y_pred_analysis, attn_analysis = transformer.forward(x_analysis)

fig4 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, n_heads, figure=fig4, hspace=0.3, wspace=0.3)

# Plot each head's attention
for h in range(n_heads):
    # Attention heatmap
    ax1 = fig4.add_subplot(gs[0, h])
    im = ax1.imshow(attn_analysis[h], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_title(f'Head {h+1}', fontsize=11, weight='bold')
    if h == 0:
        ax1.set_ylabel('Query Position', fontsize=10)
    ax1.set_xlabel('Key Position', fontsize=10)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Attention from last position
    ax2 = fig4.add_subplot(gs[1, h])
    ax2.bar(range(seq_len), attn_analysis[h][-1], color='coral', edgecolor='black')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Attention Weight', fontsize=10)
    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_title(f'Last Pos → All Positions', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

plt.savefig(f'{output_dir}/05_multihead_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_multihead_analysis.png")

# ============================================================================
# Example 6: Attention vs Position Importance
# ============================================================================

print("\n8. Analyzing Position Importance...")

# Compute average attention weights across test samples
all_attn_last = []

for i in range(min(50, n_test)):
    _, attn = transformer.forward(X_test[i])
    avg_attn = np.mean(attn, axis=0)  # Average across heads
    all_attn_last.append(avg_attn[-1])  # Attention from last position

avg_position_importance = np.mean(all_attn_last, axis=0)
std_position_importance = np.std(all_attn_last, axis=0)

fig5 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 2, figure=fig5, wspace=0.3)

# (a) Average attention to each position
ax1 = fig5.add_subplot(gs[0, 0])
ax1.bar(range(seq_len), avg_position_importance, color='skyblue',
       edgecolor='black', linewidth=1.5, yerr=std_position_importance,
       capsize=5)
ax1.set_xlabel('Position in Sequence', fontsize=12)
ax1.set_ylabel('Average Attention Weight', fontsize=12)
ax1.set_title('(a) Position Importance (from last position)', fontsize=13, weight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# (b) Relative position effect
ax2 = fig5.add_subplot(gs[0, 1])
relative_positions = np.arange(seq_len) - (seq_len - 1)  # Distance from last
ax2.scatter(relative_positions, avg_position_importance, s=100, alpha=0.7,
           c=avg_position_importance, cmap='YlOrRd', edgecolors='black')
ax2.set_xlabel('Relative Position (from last)', fontsize=12)
ax2.set_ylabel('Average Attention Weight', fontsize=12)
ax2.set_title('(b) Attention vs Relative Distance', fontsize=13, weight='bold')
ax2.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(relative_positions, avg_position_importance, 2)
p = np.poly1d(z)
ax2.plot(relative_positions, p(relative_positions), "r--", linewidth=2,
        label='Quadratic fit')
ax2.legend(fontsize=10)

print(f"\n   Most attended positions (from last):")
top_positions = np.argsort(avg_position_importance)[-5:][::-1]
for pos in top_positions:
    print(f"   Position {pos}: {avg_position_importance[pos]:.4f} ± {std_position_importance[pos]:.4f}")

plt.savefig(f'{output_dir}/05_position_importance.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/05_position_importance.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Task: Sequence Prediction")
print(f"   - Predict next value in sine wave sequence")
print(f"   - Sequence length: {seq_len}")
print(f"   - Training samples: {n_train}")
print(f"   - Test samples: {n_test}")
print(f"\n2. Model Performance:")
print(f"   - Transformer avg loss: {np.mean(transformer_losses):.4f} ± {np.std(transformer_losses):.4f}")
print(f"   - RNN avg loss: {np.mean(rnn_losses):.4f} ± {np.std(rnn_losses):.4f}")
print(f"\n3. Attention Insights:")
print(f"   - Multi-head attention captures different patterns")
print(f"   - Each head focuses on different positions")
print(f"   - Recent positions get more attention (generally)")
print(f"   - Attention weights interpretable!")
print(f"\n4. Transformer Advantages:")
print(f"   - Parallel processing (all positions at once)")
print(f"   - Direct connections (O(1) path length)")
print(f"   - Interpretable via attention weights")
print(f"   - Can capture long-range dependencies")
print(f"\n5. RNN Characteristics:")
print(f"   - Sequential processing (step by step)")
print(f"   - O(n) path length for distant positions")
print(f"   - Hidden state as memory")
print(f"   - May struggle with long sequences")
print(f"\n6. When to Use Which?")
print(f"   - Transformer: Parallel hardware, need interpretability")
print(f"   - RNN: Very long sequences, limited memory")
print(f"   - Transformer: Better for most modern NLP tasks")
print(f"   - RNN: Still useful for online/streaming scenarios")
print("="*70)

plt.show()
