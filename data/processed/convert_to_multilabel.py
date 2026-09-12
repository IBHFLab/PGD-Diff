import re
from collections import defaultdict

# ==================== 配置 ====================
INPUT_FILE = "prefix_training_data.txt"
OUTPUT_FILE = "prefix_training_data_multilabel.txt"

# ==================== 读取数据 ====================
# 建立 序列 → {癌症标签集合} 的映射
seq_to_cancers = defaultdict(set)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        # 提取标签和序列
        start = line.find('[')
        end = line.find(']')
        if start == -1 or end == -1:
            print(f"⚠️ 第 {line_num} 行格式异常（缺少括号），跳过: {line}")
            continue
        
        label_str = line[start+1:end].strip()
        seq = line[end+1:].strip().upper()
        
        if not seq:
            print(f"⚠️ 第 {line_num} 行序列为空，跳过")
            continue
        
        # 处理标签（可能已有逗号，但原始数据是单标签，这里兼容处理）
        for cancer in label_str.split(','):
            cancer = cancer.strip()
            if cancer:
                seq_to_cancers[seq].add(cancer)

print(f"✅ 读取完成：共 {len(seq_to_cancers)} 条唯一序列")

# ==================== 统计跨标签分布 ====================
label_counts = defaultdict(int)
for cancers in seq_to_cancers.values():
    label_counts[len(cancers)] += 1

print("\n【序列标签分布】")
for k in sorted(label_counts.keys()):
    print(f"  {k} 个标签: {label_counts[k]} 条序列")

# ==================== 输出多标签文件 ====================
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    # 按序列长度分组输出（可选），这里按序列字母序
    for seq, cancers in sorted(seq_to_cancers.items(), key=lambda x: len(x[0])):
        # 标签按字母序排列，便于阅读
        sorted_cancers = sorted(cancers)
        label_str = ','.join(sorted_cancers)
        f.write(f"[{label_str}] {seq}\n")

print(f"\n✅ 多标签数据已保存至: {OUTPUT_FILE}")

# ==================== 统计每个癌种的序列数（多标签去重后） ====================
cancer_seq_count = defaultdict(int)
for seq, cancers in seq_to_cancers.items():
    for cancer in cancers:
        cancer_seq_count[cancer] += 1

print("\n【各癌种关联的序列数量（多标签去重后）】")
print(f"{'癌种':<12} {'序列数':<10}")
print("-" * 25)
for cancer, count in sorted(cancer_seq_count.items(), key=lambda x: -x[1]):
    print(f"{cancer:<12} {count:<10}")

print(f"\n总序列数（去重后）: {len(seq_to_cancers)}")