import json
import subprocess
import os
import pandas as pd
import numpy as np

# ==================== 配置参数 ====================
TARGET_NUM = 50         # 每类癌症生成数量
RANDOM_SEED = 42          # 随机种子，保证可复现
MIN_LEN = 5               # 最短多肽长度（如果真实样本长度 < 5，则强制为 5）
MAX_LEN = 50              # 最大多肽长度（过滤标准）

# ==================== 1. 读取真实数据并提取长度分布 ====================
real_csv = "D:/Issue/2024_2025/code/data/processed/CancerPPD_merged_filtered.csv"
print("📖 读取真实数据...")
df_real = pd.read_csv(real_csv)

# 验证列名
TISSUE_COL = 'Tissue Affected'
SEQ_COL = 'Sequence'
if TISSUE_COL not in df_real.columns:
    for col in df_real.columns:
        if 'tissue' in col.lower() or 'cancer' in col.lower():
            TISSUE_COL = col
            break
if SEQ_COL not in df_real.columns:
    for col in df_real.columns:
        if 'sequence' in col.lower() or 'seq' in col.lower():
            SEQ_COL = col
            break

print(f"   使用列: Tissue='{TISSUE_COL}', Sequence='{SEQ_COL}'")

# ==================== 2. 按癌症类型提取长度分布（强制 ≥ MIN_LEN） ====================
cancer_lengths = {}
filtered_out = 0

for short_name, group in df_real.groupby(TISSUE_COL):
    # 过滤长度 ≤ MAX_LEN，并将 < MIN_LEN 的修正为 MIN_LEN
    valid_lengths = []
    for seq in group[SEQ_COL]:
        seq_len = len(seq)
        if seq_len <= MAX_LEN:
            # 确保长度至少为 MIN_LEN
            adjusted_len = max(MIN_LEN, seq_len)
            valid_lengths.append(adjusted_len)
    
    n_filtered = len(group) - len(valid_lengths)
    filtered_out += n_filtered
    
    if not valid_lengths:
        # 如果该癌症完全没有有效长度数据，使用后备长度池（15-30 随机）
        print(f"⚠️ {short_name}: 无有效长度数据（全部 > {MAX_LEN} 或为空），使用后备长度池 15-30")
        # 生成一个包含 15-30 的列表作为后备（每个长度出现 10 次，保证有足够多样性）
        fallback_pool = list(np.arange(15, 31)) * 10
        cancer_lengths[short_name] = fallback_pool
        continue
    
    cancer_lengths[short_name] = valid_lengths
    print(f"   {short_name}: 原始 {len(group)} 条, 有效 {len(valid_lengths)} 条 "
          f"(剔除 {n_filtered}), 长度范围 [{min(valid_lengths)}, {max(valid_lengths)}], "
          f"平均 {np.mean(valid_lengths):.1f}")

print(f"\n✅ 全局过滤统计：共剔除 {filtered_out} 条长度 > {MAX_LEN} 的序列")
print(f"   所有有效长度已确保 ≥ {MIN_LEN}")

# ==================== 3. 读取癌症标签映射 ====================
with open('cancer_to_idx.json', 'r', encoding='utf-8') as f:
    cancer_to_idx = json.load(f)

cancer_types = sorted(cancer_to_idx.keys())
print(f"\n✅ 检测到 {len(cancer_types)} 种癌症类型: {cancer_types}")

# ==================== 4. 过滤：只保留在 cancer_to_idx 中的癌种 ====================
for short_name in list(cancer_lengths.keys()):
    if short_name not in cancer_to_idx:
        print(f"⚠️ 跳过 '{short_name}'（不在 cancer_to_idx.json 中）")
        del cancer_lengths[short_name]

# ==================== 5. 批量生成 ====================
output_dir = "generated_peptides_length_matched"
os.makedirs(output_dir, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)

for short_name in cancer_types:
    # 检查该癌症是否有长度数据
    if short_name not in cancer_lengths:
        print(f"⚠️ 跳过 {short_name}：无长度数据")
        continue
    
    lengths_pool = cancer_lengths[short_name]
    
    # 从真实长度分布中采样（确保所有长度 ≥ MIN_LEN）
    if len(lengths_pool) >= TARGET_NUM:
        sampled_lengths = rng.choice(lengths_pool, size=TARGET_NUM, replace=False).tolist()
        sample_info = f"无放回 (池大小 {len(lengths_pool)})"
    else:
        sampled_lengths = rng.choice(lengths_pool, size=TARGET_NUM, replace=True).tolist()
        sample_info = f"有放回 (池大小 {len(lengths_pool)})"
    
    # 额外安全：检查是否所有长度都 ≥ MIN_LEN
    if min(sampled_lengths) < MIN_LEN:
        # 如果不满足，强制修正（理论上不会发生，因为长度池已经保证）
        sampled_lengths = [max(MIN_LEN, l) for l in sampled_lengths]
        print(f"   ⚠️ {short_name}: 修正了部分小于 {MIN_LEN} 的长度")
    
    # 随机打乱长度顺序
    rng.shuffle(sampled_lengths)
    lengths_str = ','.join(map(str, sampled_lengths))

    print(f"\n🚀 正在为 {short_name} 生成 {TARGET_NUM} 条序列...")
    print(f"   长度采样: {sample_info}, 范围 [{min(sampled_lengths)}, {max(sampled_lengths)}], "
          f"平均 {np.mean(sampled_lengths):.1f}")

    command = [
        "python", "generate_conditional_peptides.py",
        "--cancer", short_name,
        "--lengths", lengths_str,
        "--ckpt", "models/prefix_tuned/last.ckpt",
        "--guidance_scale", "3",
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"✅ {short_name} 生成完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ {short_name} 生成失败: {e}")

print(f"\n🎉 全部生成完成！文件保存在 {output_dir}/ 目录下")
print(f"   检查点: models/prefix_tuned/last.ckpt")
print(f"   CFG guidance_scale: 3.0")
print(f"   最短长度限制: {MIN_LEN} (所有生成长度 ≥ {MIN_LEN})")