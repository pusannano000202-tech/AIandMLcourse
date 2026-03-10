"""
01. Tokens and Embeddings - LLM의 기초
토큰과 임베딩 - LLM의 기본 구성 요소

Large Language Model이 텍스트를 처리하는 방법:
- Tokenization: 텍스트를 작은 단위로 분할
- Vocabulary: 모든 가능한 토큰의 집합
- Embedding: 토큰을 벡터로 변환
- Subword Tokenization: BPE, WordPiece

학습 목표:
1. Tokenization이 왜 필요한지
2. 다양한 Tokenization 방법
3. Token Embedding의 원리
4. Vocabulary 크기의 중요성
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import os
from collections import Counter
import re

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Tokens and Embeddings - Fundamentals of LLM")
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
# Tokenization Methods
# ============================================================================

def character_tokenization(text):
    """
    Character-level tokenization
    가장 간단한 방법: 각 문자가 하나의 토큰
    """
    return list(text)

def word_tokenization(text):
    """
    Word-level tokenization
    공백과 구두점으로 분리
    """
    # Simple word tokenization
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

def simple_bpe_tokenization(text, num_merges=10):
    """
    Simplified Byte Pair Encoding (BPE)

    BPE는 subword tokenization의 대표적 방법
    - 자주 등장하는 문자 쌍을 병합
    - OOV (Out-of-Vocabulary) 문제 해결

    Parameters:
    -----------
    text : str
        입력 텍스트
    num_merges : int
        병합 횟수

    Returns:
    --------
    tokens : list
        토큰 리스트
    vocab : set
        vocabulary
    """
    # Start with character-level
    tokens = [' '.join(list(word)) + ' </w>' for word in text.lower().split()]

    vocab = set()
    for token in tokens:
        vocab.update(token.split())

    # Perform merges
    for i in range(num_merges):
        # Count pairs
        pairs = Counter()
        for token in tokens:
            symbols = token.split()
            for j in range(len(symbols) - 1):
                pairs[(symbols[j], symbols[j+1])] += 1

        if not pairs:
            break

        # Get most frequent pair
        best_pair = max(pairs, key=pairs.get)

        # Merge the pair
        new_token = ''.join(best_pair)
        tokens = [token.replace(' '.join(best_pair), new_token) for token in tokens]
        vocab.add(new_token)

    # Final tokens
    final_tokens = []
    for token in tokens:
        final_tokens.extend(token.split())

    return final_tokens, vocab

# ============================================================================
# Example 1: Different Tokenization Methods
# ============================================================================

print("\n1. Comparing Tokenization Methods...")

sample_text = "The quick brown fox jumps over the lazy dog. AI is amazing!"

print(f"\n   Original text: '{sample_text}'")
print(f"   Text length: {len(sample_text)} characters")

# Character-level
char_tokens = character_tokenization(sample_text)
print(f"\n   Character-level tokenization:")
print(f"   Tokens: {char_tokens[:20]}...")
print(f"   Total tokens: {len(char_tokens)}")
print(f"   Vocabulary size: {len(set(char_tokens))}")

# Word-level
word_tokens = word_tokenization(sample_text)
print(f"\n   Word-level tokenization:")
print(f"   Tokens: {word_tokens}")
print(f"   Total tokens: {len(word_tokens)}")
print(f"   Vocabulary size: {len(set(word_tokens))}")

# BPE-like
bpe_tokens, bpe_vocab = simple_bpe_tokenization(sample_text, num_merges=5)
print(f"\n   BPE-style tokenization (5 merges):")
print(f"   Tokens: {bpe_tokens[:15]}...")
print(f"   Total tokens: {len(bpe_tokens)}")
print(f"   Vocabulary size: {len(bpe_vocab)}")

# ============================================================================
# Visualization 1: Token Statistics
# ============================================================================

print("\n2. Visualizing Token Statistics...")

fig1 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig1, wspace=0.3)

methods = ['Character', 'Word', 'BPE']
token_counts = [len(char_tokens), len(word_tokens), len(bpe_tokens)]
vocab_sizes = [len(set(char_tokens)), len(set(word_tokens)), len(bpe_vocab)]

# (a) Number of tokens
ax1 = fig1.add_subplot(gs[0, 0])
bars1 = ax1.bar(methods, token_counts, color=['skyblue', 'coral', 'lightgreen'],
               edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Tokens', fontsize=12)
ax1.set_title('(a) Token Count Comparison', fontsize=13, weight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars1, token_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}', ha='center', va='bottom', fontsize=11, weight='bold')

# (b) Vocabulary size
ax2 = fig1.add_subplot(gs[0, 1])
bars2 = ax2.bar(methods, vocab_sizes, color=['skyblue', 'coral', 'lightgreen'],
               edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Vocabulary Size', fontsize=12)
ax2.set_title('(b) Vocabulary Size Comparison', fontsize=13, weight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar, size in zip(bars2, vocab_sizes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{size}', ha='center', va='bottom', fontsize=11, weight='bold')

# (c) Efficiency (tokens per character)
ax3 = fig1.add_subplot(gs[0, 2])
efficiency = [token_counts[i] / len(sample_text) for i in range(3)]
bars3 = ax3.bar(methods, efficiency, color=['skyblue', 'coral', 'lightgreen'],
               edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Tokens per Character', fontsize=12)
ax3.set_title('(c) Tokenization Efficiency', fontsize=13, weight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar, eff in zip(bars3, efficiency):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{eff:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')

plt.savefig(f'{output_dir}/01_tokenization_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_tokenization_comparison.png")

# ============================================================================
# Example 2: Token Embeddings
# ============================================================================

print("\n3. Creating Token Embeddings...")

# Build vocabulary from word tokens
vocab = sorted(set(word_tokens))
vocab_size = len(vocab)
embedding_dim = 8  # Small for visualization

print(f"\n   Vocabulary: {vocab}")
print(f"   Vocabulary size: {vocab_size}")
print(f"   Embedding dimension: {embedding_dim}")

# Create token-to-index mapping
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# Initialize embeddings (random)
np.random.seed(42)
embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

print(f"\n   Embedding matrix shape: {embeddings.shape}")
print(f"   Example - '{vocab[0]}' embedding: {embeddings[0, :4]}...")

# ============================================================================
# Visualization 2: Embedding Space
# ============================================================================

print("\n4. Visualizing Embedding Space...")

# Use PCA-like projection for 2D visualization
# Simple projection: use first 2 dimensions
embedding_2d = embeddings[:, :2]

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

# (a) Embedding matrix heatmap
ax1 = fig2.add_subplot(gs[0, 0])
im1 = ax1.imshow(embeddings.T, cmap='RdBu_r', aspect='auto')
ax1.set_xlabel('Token Index', fontsize=11)
ax1.set_ylabel('Embedding Dimension', fontsize=11)
ax1.set_title('(a) Token Embedding Matrix', fontsize=12, weight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (b) 2D embedding space
ax2 = fig2.add_subplot(gs[0, 1])
ax2.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=100, alpha=0.6,
           c=range(vocab_size), cmap='tab20', edgecolors='black')

for idx, token in enumerate(vocab):
    ax2.annotate(token, (embedding_2d[idx, 0], embedding_2d[idx, 1]),
                fontsize=9, ha='center', va='bottom')

ax2.set_xlabel('Dimension 1', fontsize=11)
ax2.set_ylabel('Dimension 2', fontsize=11)
ax2.set_title('(b) 2D Embedding Space', fontsize=12, weight='bold')
ax2.grid(True, alpha=0.3)

# (c) Sample token embeddings
ax3 = fig2.add_subplot(gs[1, 0])
sample_tokens_idx = [0, 3, 7, 10]
sample_tokens = [vocab[i] for i in sample_tokens_idx if i < vocab_size]

for idx in sample_tokens_idx:
    if idx < vocab_size:
        ax3.plot(range(embedding_dim), embeddings[idx], marker='o',
                linewidth=2, markersize=6, label=vocab[idx])

ax3.set_xlabel('Dimension', fontsize=11)
ax3.set_ylabel('Value', fontsize=11)
ax3.set_title('(c) Sample Token Embeddings', fontsize=12, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# (d) Embedding similarity (cosine)
ax4 = fig2.add_subplot(gs[1, 1])

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_matrix = np.zeros((vocab_size, vocab_size))
for i in range(vocab_size):
    for j in range(vocab_size):
        similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

im4 = ax4.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax4.set_xlabel('Token Index', fontsize=11)
ax4.set_ylabel('Token Index', fontsize=11)
ax4.set_title('(d) Cosine Similarity Matrix', fontsize=12, weight='bold')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Similarity')

plt.savefig(f'{output_dir}/01_token_embeddings.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_token_embeddings.png")

# ============================================================================
# Example 3: Context Window and Token Limits
# ============================================================================

print("\n5. Understanding Context Window...")

# Simulate different context window sizes
context_windows = [8, 16, 32, 64, 128]

long_text = " ".join(["word"] * 200)  # 200 words
long_tokens = word_tokenization(long_text)

print(f"\n   Long text: {len(long_tokens)} tokens")

fig3 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 2, figure=fig3, wspace=0.3)

# (a) Tokens that fit in different context windows
ax1 = fig3.add_subplot(gs[0, 0])
tokens_fit = [min(len(long_tokens), cw) for cw in context_windows]
coverage = [tf / len(long_tokens) * 100 for tf in tokens_fit]

bars = ax1.bar([str(cw) for cw in context_windows], tokens_fit,
              color='steelblue', edgecolor='black', linewidth=1.5)
ax1.axhline(y=len(long_tokens), color='red', linestyle='--',
           linewidth=2, label=f'Total tokens ({len(long_tokens)})')
ax1.set_xlabel('Context Window Size', fontsize=12)
ax1.set_ylabel('Tokens That Fit', fontsize=12)
ax1.set_title('(a) Context Window Capacity', fontsize=13, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

for bar, tf, cov in zip(bars, tokens_fit, coverage):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{tf}\n({cov:.0f}%)', ha='center', va='bottom', fontsize=9)

# (b) Real-world context windows
ax2 = fig3.add_subplot(gs[0, 1])

models = ['GPT-2', 'GPT-3', 'GPT-4\n(8K)', 'GPT-4\n(32K)', 'Claude 2', 'Claude 3']
context_sizes = [1024, 2048, 8192, 32768, 100000, 200000]

bars2 = ax2.barh(models, context_sizes, color='coral', edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Context Window (tokens)', fontsize=12)
ax2.set_title('(b) Real LLM Context Windows', fontsize=13, weight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, axis='x')

for bar, size in zip(bars2, context_sizes):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f' {size:,}', ha='left', va='center', fontsize=10, weight='bold')

plt.savefig(f'{output_dir}/01_context_window.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_context_window.png")

# ============================================================================
# Example 4: Vocabulary Size Trade-offs
# ============================================================================

print("\n6. Analyzing Vocabulary Size Trade-offs...")

# Simulate different vocabulary sizes
vocab_sizes_analysis = [100, 500, 1000, 5000, 10000, 50000, 100000]

# Estimate token counts (inversely related to vocab size for subword)
# Larger vocab → fewer tokens needed
avg_tokens_per_word = [3.0, 2.0, 1.5, 1.2, 1.1, 1.05, 1.02]

# Memory usage (proportional to vocab size)
embedding_dim_large = 768  # Typical for large models
memory_mb = [vs * embedding_dim_large * 4 / 1024 / 1024 for vs in vocab_sizes_analysis]

fig4 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig4, wspace=0.3)

# (a) Tokens per word vs vocab size
ax1 = fig4.add_subplot(gs[0, 0])
ax1.plot(vocab_sizes_analysis, avg_tokens_per_word, marker='o',
        linewidth=2, markersize=8, color='blue')
ax1.set_xlabel('Vocabulary Size', fontsize=12)
ax1.set_ylabel('Avg Tokens per Word', fontsize=12)
ax1.set_title('(a) Tokenization Efficiency', fontsize=13, weight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# (b) Memory usage vs vocab size
ax2 = fig4.add_subplot(gs[0, 1])
ax2.plot(vocab_sizes_analysis, memory_mb, marker='s',
        linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Vocabulary Size', fontsize=12)
ax2.set_ylabel('Embedding Memory (MB)', fontsize=12)
ax2.set_title('(b) Memory Footprint', fontsize=13, weight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# (c) Trade-off visualization
ax3 = fig3.add_subplot(gs[0, 1])
ax3_twin = ax3.twinx()

line1 = ax3.plot(vocab_sizes_analysis, avg_tokens_per_word, marker='o',
                linewidth=2, markersize=8, color='blue', label='Tokens/Word')
line2 = ax3_twin.plot(vocab_sizes_analysis, memory_mb, marker='s',
                     linewidth=2, markersize=8, color='red', label='Memory (MB)')

ax3.set_xlabel('Vocabulary Size', fontsize=12)
ax3.set_ylabel('Avg Tokens per Word', fontsize=12, color='blue')
ax3_twin.set_ylabel('Memory (MB)', fontsize=12, color='red')
ax3.set_title('(c) Vocabulary Size Trade-off', fontsize=13, weight='bold')
ax3.set_xscale('log')
ax3.tick_params(axis='y', labelcolor='blue')
ax3_twin.tick_params(axis='y', labelcolor='red')
ax3.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.savefig(f'{output_dir}/01_vocabulary_tradeoffs.png', dpi=150, bbox_inches='tight')
print(f"\n   Saved: {output_dir}/01_vocabulary_tradeoffs.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Key Results:")
print("="*70)
print(f"1. Tokenization Methods:")
print(f"   - Character: Simple, large vocab for each language")
print(f"   - Word: Intuitive, but huge vocab, OOV problems")
print(f"   - Subword (BPE): Best trade-off, handles rare words")
print(f"\n2. Token Embeddings:")
print(f"   - Convert discrete tokens → continuous vectors")
print(f"   - Learned during training")
print(f"   - Capture semantic similarity")
print(f"   - Typical dim: 768 (BERT), 12288 (GPT-3)")
print(f"\n3. Context Window:")
print(f"   - Maximum number of tokens model can process")
print(f"   - GPT-2: 1K, GPT-4: 8K-128K, Claude 3: 200K")
print(f"   - Longer context = more memory, slower")
print(f"\n4. Vocabulary Size:")
print(f"   - Larger vocab: fewer tokens, more memory")
print(f"   - Smaller vocab: more tokens, less memory")
print(f"   - Common: 30K-50K tokens (GPT-2, BERT)")
print(f"   - GPT-4: ~100K tokens")
print(f"\n5. Why Tokenization Matters:")
print(f"   - Affects model size and speed")
print(f"   - Determines what languages/scripts supported")
print(f"   - Impacts reasoning about characters (e.g., spelling)")
print(f"   - Different for different modalities (text, code, etc.)")
print("="*70)

plt.show()
