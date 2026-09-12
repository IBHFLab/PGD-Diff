import re
import pandas as pd
from collections import defaultdict
from Levenshtein import distance as levenshtein_dist

# 1. 读取数据，按癌症类型分组
file_path = "prefix_training_data.txt"

cancer_groups = defaultdict(list)
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 匹配格式：[Cancer] Sequence
        match = re.match(r'\[([^\]]+)\]\s*(.+)', line)
        if match:
            cancer_type = match.group(1).strip()
            sequence = match.group(2).strip()
            cancer_groups[cancer_type].append(sequence)

# 获取所有癌症类型并排序，保证矩阵顺序固定
cancer_types = sorted(cancer_groups.keys())
print(f"检测到 {len(cancer_types)} 种癌症类型: {cancer_types}")
print("-" * 50)

# 2. 计算完全相同序列重叠矩阵 (Exact Match Matrix)
# 先将每个类型的序列转为集合，方便快速查找
cancer_sets = {c: set(seqs) for c, seqs in cancer_groups.items()}

exact_matrix = pd.DataFrame(0, index=cancer_types, columns=cancer_types)
for c1 in cancer_types:
    for c2 in cancer_types:
        # 计算 c1 中的序列有多少条也出现在 c2 中
        overlap = len(cancer_sets[c1].intersection(cancer_sets[c2]))
        exact_matrix.loc[c1, c2] = overlap

print("【完全相同序列重叠矩阵 (Exact Match)】")
print(exact_matrix.to_string())
print("\n" + "-" * 50)

# 3. 计算高同源序列重叠矩阵 (High Similarity >= 80%)
# 注意：如果两个序列长度相差很大，编辑距离会较大，这里采用归一化编辑距离。
def is_similar(seq1, seq2, threshold=0.8):
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return True
    # 归一化编辑距离 = 1 - (编辑距离 / 最长长度)
    sim = 1 - (levenshtein_dist(seq1, seq2) / max_len)
    return sim >= threshold

similar_matrix = pd.DataFrame(0, index=cancer_types, columns=cancer_types)
# 为了加速，只对非完全相同的序列对进行相似度计算，并将结果复制到对称位置
for i, c1 in enumerate(cancer_types):
    for j, c2 in enumerate(cancer_types):
        if i > j:  # 对称，直接复制
            similar_matrix.loc[c1, c2] = similar_matrix.loc[c2, c1]
            continue
        
        count_sim = 0
        # 只计算不同类之间的相似度，同类之间跳过（或者你可以算同类内高度相似，但审稿人主要关注跨癌种）
        if c1 == c2:
            # 对角线放该类型的总序列数（去重后）
            count_sim = len(cancer_sets[c1])
        else:
            # 为了避免 O(N^2) 过慢，可以采样或全量计算（数据量几千条完全没问题）
            list1 = list(cancer_sets[c1])
            list2 = list(cancer_sets[c2])
            # 快速过滤：如果长度差超过 20%，直接跳过相似度计算
            for seq1 in list1:
                len1 = len(seq1)
                for seq2 in list2:
                    len2 = len(seq2)
                    if abs(len1 - len2) > 0.2 * max(len1, len2):
                        continue
                    if is_similar(seq1, seq2, threshold=0.8):
                        count_sim += 1
                        # 避免重复计数：每条 c1 中的序列只计一次（这里计了与所有 c2 的匹配数）
                        # 但为了矩阵对称，我们只记录“存在匹配的序列对数量”
                        # 注意：这种算法会导致非对称，因为 len1 和 len2 不同。
                        # 更好的办法：统计“至少与对方某条序列相似的序列数量”
                        # 这里我简单统计所有满足条件的 seq1-seq2 对的数量，并让矩阵对称（取平均或最大）
                        # 为了严谨，我们统计“满足相似条件的序列对数”
                        pass
            # 由于双层循环计数，这里采取更标准的做法：统计 c1 中有多少条序列在 c2 中存在相似序列
            # 重新实现：
            sim_count = 0
            for seq1 in list1:
                len1 = len(seq1)
                found = False
                for seq2 in list2:
                    len2 = len(seq2)
                    if abs(len1 - len2) > 0.2 * max(len1, len2):
                        continue
                    if is_similar(seq1, seq2, 0.8):
                        found = True
                        break
                if found:
                    sim_count += 1
            count_sim = sim_count
        
        similar_matrix.loc[c1, c2] = count_sim

# 为了保证对称性（因为上述算法是非对称的，c1 找 c2 和 c2 找 c1 可能不一致）
# 我们取两者平均或者取最大值，通常在论文中我们展示对称矩阵，用交集计数更合适。
# 更严谨的写法（交集）：数量 = 在 c1 中存在且能在 c2 中找到相似序列的序列数量。
# 由于上面算法已经计算了 c1->c2 的计数，我们直接赋值给矩阵，但让矩阵对称会比较困难。
# 最简单的做法：对每对癌症 (c1, c2) 重新计算，找到“同时满足”的数量。
# 我这里重新写一下对称计算：
for i, c1 in enumerate(cancer_types):
    for j, c2 in enumerate(cancer_types):
        if i <= j:
            continue
        list1 = list(cancer_sets[c1])
        list2 = list(cancer_sets[c2])
        count_both = 0
        for seq1 in list1:
            len1 = len(seq1)
            for seq2 in list2:
                len2 = len(seq2)
                if abs(len1 - len2) > 0.2 * max(len1, len2):
                    continue
                if is_similar(seq1, seq2, 0.8):
                    count_both += 1
                    break  # seq1 找到第一个匹配就停止，避免重复计数
        # 对称赋值
        similar_matrix.loc[c1, c2] = count_both
        similar_matrix.loc[c2, c1] = count_both

# 对角线保持不变（该类总序列数）
for c in cancer_types:
    similar_matrix.loc[c, c] = len(cancer_sets[c])

print("\n【高同源序列重叠矩阵 (Similarity >= 80%)】")
print(similar_matrix.to_string())

# 保存为 CSV，方便在 Excel 中画热图
exact_matrix.to_csv("exact_overlap_matrix.csv")
similar_matrix.to_csv("similar_overlap_matrix.csv")
print("\n结果已保存为 exact_overlap_matrix.csv 和 similar_overlap_matrix.csv")