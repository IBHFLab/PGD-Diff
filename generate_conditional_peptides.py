# generate_conditional_peptides.py (最终修正版 - 支持分层长度采样)
# 修改点：
# 1. 新增 length_sampling_strategy 参数，支持 'uniform' (默认) 和 'stratified' (分层比例)。
# 2. 实现 stratified 采样逻辑：5-15(1/6), 15-25(2/6), 25-35(2/6), 35-50(1/6)。
# 3. 从检查点提取 hyper_parameters，用其构造模型。
# 4. 自动适配检查点的类别数，从 cancer_to_idx.json 中截取对应数量的类别。

import torch
import json
import numpy as np
from pathlib import Path
from models.prefix_tuned.PGD_Diff_condition import PGD_Diff
import warnings
import inspect

def parse_lengths(lengths_str):
    return [int(x) for x in lengths_str.split(',')]

def sample_stratified_lengths(n_seq, min_len=5, max_len=50, seed=None):
    """
    按照 1:2:2:1 的比例在不同长度区间采样
    区间定义:
    1. [5, 15]
    2. [15, 25]
    3. [25, 35]
    4. [35, 50]
    """
    rng = np.random.default_rng(seed)
    
    # 定义区间边界 (左闭右开或左闭右闭需统一，这里采用左闭右闭，注意重叠点处理)
    # 为了避免边界重复，我们定义互斥区间:
    # Range 1: 5-14
    # Range 2: 15-24
    # Range 3: 25-34
    # Range 4: 35-50
    # 但用户要求是 5-15, 15-25... 通常意味着包含边界。
    # 为了简单且符合直觉，我们使用以下互斥分段：
    # R1: [5, 14] (10个值) -> 比例 1
    # R2: [15, 24] (10个值) -> 比例 2
    # R3: [25, 34] (10个值) -> 比例 2
    # R4: [35, 50] (16个值) -> 比例 1
    
    # 如果用户坚持 5-15, 15-25 这种重叠写法，通常在编程中处理为 [5,15), [15,25)...
    # 这里我们采用更自然的生物序列长度分布逻辑：
    # Bucket 1: 5-15 (含15)
    # Bucket 2: 16-25
    # Bucket 3: 26-35
    # Bucket 4: 36-50
    
    buckets = [
        (5, 15),   # 1份
        (16, 25),  # 2份
        (26, 35),  # 2份
        (36, 50)   # 1份
    ]
    ratios = [1, 2, 2, 1]
    total_ratio = sum(ratios)
    
    lengths = []
    remaining_samples = n_seq
    remaining_ratio = total_ratio
    
    for i, (low, high) in enumerate(buckets):
        ratio = ratios[i]
        # 计算当前桶应分配的样本数，使用向下取整，最后余数补到最后一个桶或随机分配
        count = int(np.round(n_seq * ratio / total_ratio))
        
        # 修正最后一个桶的数量以匹配总数
        if i == len(buckets) - 1:
            count = n_seq - len(lengths)
            
        if count > 0:
            # 在 [low, high] 之间均匀采样
            sampled = rng.integers(low, high + 1, size=count).tolist()
            lengths.extend(sampled)
            
    # 打乱顺序，避免相同长度的序列聚集在一起
    rng.shuffle(lengths)
    return lengths

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="models/prefix_tuned/last.ckpt")
    parser.add_argument('--cancer', type=str, required=True)
    parser.add_argument('--n_seq', type=int, default=None)
    parser.add_argument('--lengths', type=str, default=None)
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--fasta_out', action='store_true')
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--temperature', type=float, default=0.8)
    # 新增参数：长度采样策略
    parser.add_argument('--length_strategy', type=str, default='stratified', 
                        choices=['uniform', 'stratified'],
                        help="长度采样策略: uniform(均匀随机), stratified(1:2:2:1分层)")
    args = parser.parse_args()

    # 确定长度列表
    if args.lengths:
        lengths = parse_lengths(args.lengths)
        n_seq = len(lengths)
    else:
        if args.n_seq is None:
            raise ValueError("请提供 --n_seq 或 --lengths")
        n_seq = args.n_seq
        
        if args.length_strategy == 'stratified':
            print(f"📏 使用分层长度采样策略 (5-15:1, 16-25:2, 26-35:2, 36-50:1)")
            lengths = sample_stratified_lengths(n_seq, min_len=args.min_len, max_len=args.max_len)
        else:
            print(f"📏 使用均匀长度采样策略 ({args.min_len}-{args.max_len})")
            rng = np.random.default_rng()
            lengths = rng.integers(args.min_len, args.max_len + 1, size=n_seq).tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 1. 加载检查点并提取超参数 ==========
    print(f"📦 加载检查点: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})
    # 确保 num_cancer_types 存在
    if 'num_cancer_types' not in hparams:
        state_dict = ckpt['state_dict']
        if 'cancer_embedding.weight' in state_dict:
            hparams['num_cancer_types'] = state_dict['cancer_embedding.weight'].shape[0]
        elif 'transformer.blocks.0.prefix_k' in state_dict:
            hparams['num_cancer_types'] = state_dict['transformer.blocks.0.prefix_k'].shape[0]
        else:
            raise ValueError("无法从检查点推断 num_cancer_types")
    model_num_classes = hparams['num_cancer_types']
    print(f"📌 检查点中的癌症类别数: {model_num_classes}")

    # ========== 2. 读取映射文件，截取前 model_num_classes 个类别 ==========
    ckpt_dir = Path(args.ckpt).parent
    mapping_file = ckpt_dir / "cancer_to_idx.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"未找到映射文件: {mapping_file}")
    with open(mapping_file, 'r') as f:
        full_cancer_to_idx = json.load(f)
    all_cancer_names = sorted(full_cancer_to_idx.keys())
    if len(all_cancer_names) < model_num_classes:
        raise ValueError(f"mapping 文件只有 {len(all_cancer_names)} 个类别，少于检查点的 {model_num_classes}")
    elif len(all_cancer_names) > model_num_classes:
        warnings.warn(f"mapping 文件包含 {len(all_cancer_names)} 个类别，但检查点只有 {model_num_classes} 个。"
                      f"将只使用前 {model_num_classes} 个类别: {all_cancer_names[:model_num_classes]}")
        cancer_names = all_cancer_names[:model_num_classes]
    else:
        cancer_names = all_cancer_names
    cancer_to_idx = {name: i for i, name in enumerate(cancer_names)}
    print(f"✅ 模型使用的癌症类别: {cancer_names}")

    # ========== 3. 解析用户指定的癌症名称，构建多热向量 ==========
    user_cancers = [c.strip() for c in args.cancer.split(',')]
    multi_hot = torch.zeros(model_num_classes, dtype=torch.float32)
    unknown = []
    for name in user_cancers:
        if name in cancer_to_idx:
            multi_hot[cancer_to_idx[name]] = 1.0
        else:
            unknown.append(name)
    if unknown:
        print(f"⚠️ 未知癌症: {unknown}")
        print(f"   可用: {cancer_names}")
        return
    multi_hot_batch = multi_hot.unsqueeze(0).repeat(n_seq, 1)
    print(f"✅ 生成针对 [{args.cancer}] 的 {n_seq} 条序列")
    print(f"   多热向量: {multi_hot.tolist()}")

    # ========== 4. 使用检查点中的超参数实例化模型 ==========
    print("🔄 使用检查点超参数实例化模型...")
    # 获取 PGD_Diff 的默认参数
    sig = inspect.signature(PGD_Diff.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    # 合并，hparams 覆盖默认值
    merged_params = {**defaults, **hparams}
    # 创建模型
    model = PGD_Diff(**merged_params)
    # 加载权重
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    model.to(device)
    print("✅ 模型加载成功")

    # ========== 5. 生成 ==========
    print(f"🚀 开始生成... (guidance_scale={args.guidance_scale})")
    sequences, _, _ = model.denoise_seq_sample_with_prefix(
        cancer_multi_hot=multi_hot_batch,
        n_seq=n_seq,
        seq_length=lengths,
        fasta_out_statue=args.fasta_out,
        guidance_scale=args.guidance_scale
    )

    # ========== 6. 保存 ==========
    safe_name = args.cancer.lower().replace(' ', '_').replace(',', '_')
    out_file = ckpt_dir / f"generated_{safe_name}_peptides.txt"
    with open(out_file, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')
    print(f"✅ 序列已保存至 {out_file}")
    
    # 统计长度分布
    from collections import Counter
    len_counts = Counter(lengths)
    print("\n📊 生成长度分布统计:")
    for l in sorted(len_counts.keys()):
        print(f"   Length {l}: {len_counts[l]} sequences")

if __name__ == '__main__':
    main()
