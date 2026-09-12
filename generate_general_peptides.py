#!/usr/bin/env python
"""
抗癌肽序列生成脚本（优化版）
功能：动态长度生成 + 分组评估
作者：AI助手
日期：2025-06-26
"""
import os
import sys
import torch
import random
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from models.PGD_Diff import PGD_Diff  # 导入您的模型类

# 配置参数（根据您的实际需求调整）
CONFIG = {
    "num_sequences": 10000,       # 总生成序列数
    "min_length": 5,             # 最小长度
    "max_length": 50,             # 最大长度
    "target_distribution": {
        "Peptides_5-15": (5, 15, 0.25),   # 短肽组: min, max, 占比
        "Peptides_16-25": (16, 25, 0.25), # 中短肽组
        "Peptides_26-35": (26, 35, 0.25), # 中长肽组
        "Peptides_36-50": (36, 50, 0.25)  # 长肽组
    },
    "checkpoint_path": "data/output/pgd_diff/both_dual_encoder/last.ckpt",
    "output_dir": "data/output/test",
    "output_file": "pgd_diff_10000.fasta",
    "seed": 2026,                # 随机种子
    "model_config": {
        'n_timestep': 200,
        'beta_schedule': 'linear',
        'beta_start': 1e-7,
        'beta_end': 0.02,
        'temperature': 0.01,
        'learning_rate_struct': 1e-3,
        'learning_rate_seq': 1e-3,
        'learning_rate_cont': 1e-3
    }
}

def set_seed(seed):
    """设置随机种子确保结果可复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"🔧 随机种子设置为: {seed}")

def generate_lengths(distribution, num_sequences):
    """根据目标分布生成长度列表"""
    group_counts = {}
    length_list = []
    
    # 计算每组应得的序列数量（平均分配）
    groups = list(distribution.keys())
    base_count = num_sequences // len(groups)  # 每组基础数量
    remainder = num_sequences % len(groups)    # 余数
    
    for i, (group, (min_len, max_len, _)) in enumerate(distribution.items()):
        # 基础数量加上余数分配（前remainder组各多一个）
        count = base_count + (1 if i < remainder else 0)
        group_counts[group] = {
            "count": count,
            "min": min_len,
            "max": max_len
        }
    
    # 生成长度列表
    for group, info in group_counts.items():
        for _ in range(info["count"]):
            length = random.randint(info["min"], info["max"])
            length_list.append(length)
    
    # 打乱顺序
    random.shuffle(length_list)
    
    # 打印分布信息
    print("📊 长度分布设置:")
    for group, info in group_counts.items():
        print(f" - {group}组 ({info['min']}-{info['max']}残基): {info['count']}条序列")
    
    return length_list


def load_model(checkpoint_path, model_config):
    """加载预训练模型，忽略额外参数"""
    try:
        print(f"⏳ 从 {checkpoint_path} 加载模型...")
        
        # 创建模型实例
        model = PGD_Diff(**model_config)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # 创建新的状态字典，只包含当前模型需要的键
        model_state_dict = model.state_dict()
        new_state_dict = {}
        
        # 匹配模型参数
        for key in model_state_dict.keys():
            if key in state_dict:
                # 检查形状是否匹配
                if state_dict[key].shape == model_state_dict[key].shape:
                    new_state_dict[key] = state_dict[key]
                else:
                    print(f"⚠️ 形状不匹配: {key} (检查点: {state_dict[key].shape}, 当前: {model_state_dict[key].shape})")
                    # 使用当前模型初始化
                    new_state_dict[key] = model_state_dict[key]
            else:
                print(f"⚠️ 键 {key} 不在检查点中，使用当前模型初始化")
                new_state_dict[key] = model_state_dict[key]
        
        # 加载新状态字典
        model.load_state_dict(new_state_dict)
        model.eval()
        model.freeze()
        
        print("✅ 模型加载成功！")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        return None

def generate_sequences(model, num_sequences, length_list):
    """使用MMCD模型生成抗癌肽序列"""
    print(f"\n🧬 开始使用MMCD模型生成 {num_sequences} 条肽序列...")
    
    try:
        # 调用模型的denoise_seq_sample方法
        sequences, _, _ = model.denoise_seq_sample(
            n_seq=num_sequences,
            seq_length=length_list,
            fasta_out_statue=False
        )
        
        # 显示部分生成的序列
        print("\n🔬 部分生成序列示例:")
        for i, seq in enumerate(sequences[:10]):
            print(f"肽 {i+1}: {seq[:10]}... (长度: {len(seq)})")
        
        # 计算实际长度分布
        length_stats = {}
        for seq in sequences:
            length = len(seq)
            length_stats[length] = length_stats.get(length, 0) + 1
        
        print("\n📊 实际长度分布:")
        for length, count in sorted(length_stats.items()):
            print(f" - {length}个残基: {count}条序列")
        
        return sequences
    
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        raise RuntimeError("肽序列生成失败，无法继续") from e

def save_sequences(sequences, output_dir, output_file):
    """保存序列到FASTA文件"""
    if not sequences:
        print("\n⚠️ 没有序列需要保存")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    records = []
    for i, seq in enumerate(sequences):
        records.append(SeqRecord(
            Seq(seq), 
            id=f"MMCD_{i+1:04d}", 
            description=f"Length={len(seq)}"
        ))
    
    SeqIO.write(records, output_path, "fasta")
    print(f"\n💾 序列已保存至: {os.path.abspath(output_path)}")
    return output_path

def main():
    """主函数"""
    config = CONFIG
    
    print("\n" + "=" * 60)
    print("抗癌肽序列生成系统 - 动态长度优化版")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
    
    # 设置随机种子
    set_seed(config["seed"])
    
    # 检查模型文件是否存在
    if not os.path.exists(config["checkpoint_path"]):
        print(f"\n❌ 错误: 检查点文件 {config['checkpoint_path']} 不存在")
        return
    
    # 生成长度分布
    length_list = generate_lengths(config["target_distribution"], config["num_sequences"])
    
    # 加载模型
    model = load_model(config["checkpoint_path"], config["model_config"])
    if not model:
        return
    
    # 生成序列
    sequences = generate_sequences(model, config["num_sequences"], length_list)
    
    # 保存序列
    save_sequences(sequences, config["output_dir"], config["output_file"])
    
    print("\n🎉 序列生成完成！准备进行分组评估")

if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        input("\n按回车键退出...")