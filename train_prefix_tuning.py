# train_prefix_tuning_hybrid_fixed.py
#
# ==================== 核心改动 ====================
# 1. 只使用训练数据
# 2. 解冻整个 Transformer 和 SeqFFN 主干
# 3. 差异化学习率
# 4. 【修复】使用更强大的梯度监控 Callback，直接调用模型内部方法
# =================================================

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
import json
import os
import numpy as np
from models.prefix_tuned.PGD_Diff_condition import PGD_Diff
from util.embed.sequence import onehot_encoding
from pytorch_lightning.callbacks import EarlyStopping, Callback


class PrefixPeptideDataset(Dataset):
    def __init__(self, txt_path, cancer_to_idx, max_seq_len=50):
        self.data = []
        self.cancer_to_idx = cancer_to_idx
        self.num_cancers = len(cancer_to_idx)
        self.max_seq_len = max_seq_len

        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                start = line.find('[')
                end = line.find(']')
                if start == -1 or end == -1:
                    continue
                label_str = line[start + 1:end].strip()
                seq = line[end + 1:].strip().upper()
                if not seq:
                    continue
                cancer_indices = []
                for cancer in label_str.split(','):
                    cancer = cancer.strip()
                    if cancer in cancer_to_idx:
                        cancer_indices.append(cancer_to_idx[cancer])
                if not cancer_indices:
                    continue
                multi_hot = torch.zeros(self.num_cancers, dtype=torch.float32)
                for idx in cancer_indices:
                    multi_hot[idx] = 1.0
                if len(seq) > max_seq_len:
                    seq = seq[:max_seq_len]
                logit = onehot_encoding(seq)
                pos = np.zeros((len(seq), 4, 3), dtype=np.float32)
                self.data.append({
                    'fasta': seq,
                    'logit': torch.tensor(logit, dtype=torch.float32),
                    'pos': torch.tensor(pos, dtype=torch.float32),
                    'cancer_multi_hot': multi_hot,
                    'length': len(seq),
                })
        print(f"✅ 加载数据完成，共 {len(self.data)} 条序列，{self.num_cancers} 个癌症类型")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    batch.sort(key=lambda x: x['length'], reverse=True)
    logits, poses, fastas, multi_hots, lengths = [], [], [], [], []
    for item in batch:
        logits.append(item['logit'])
        poses.append(item['pos'])
        fastas.append(item['fasta'])
        multi_hots.append(item['cancer_multi_hot'])
        lengths.append(item['length'])
    logit_cat = torch.cat(logits, dim=0)
    pos_cat = torch.cat(poses, dim=0)
    batch_index = []
    for i, L in enumerate(lengths):
        batch_index.extend([i] * L)
    batch_index = torch.tensor(batch_index, dtype=torch.long)
    multi_hots_tensor = torch.stack(multi_hots, dim=0)
    return SimpleNamespace(
        x=torch.zeros(len(batch_index), 46),
        pos=pos_cat,
        fasta=fastas,
        logit=logit_cat,
        cancer_multi_hot=multi_hots_tensor,
        batch_index=batch_index,
        lengths=lengths,
        nonamp_fasta=[],
        nonamp_logit=torch.empty(0, 20),
        nonamp_pos=torch.empty(0, 4, 3),
    )


def build_cancer_mapping(txt_path):
    cancers = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start = line.find('[')
            end = line.find(']')
            if start == -1 or end == -1:
                continue
            label_str = line[start + 1:end].strip()
            for cancer in label_str.split(','):
                cancer = cancer.strip()
                if cancer:
                    cancers.add(cancer)
    cancer_list = sorted(cancers)
    cancer_to_idx = {c: i for i, c in enumerate(cancer_list)}
    print(f"✅ 检测到 {len(cancer_list)} 个癌症类型: {cancer_list}")
    return cancer_to_idx


# ==================== 增强版梯度监控 Callback ====================
class AdvancedPrefixMonitor(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 每 50 步检查一次
        if trainer.global_step % 50 == 0:
            try:
                # 尝试找到 transformer 模块
                # 注意：根据你的模型结构，可能需要调整路径，例如 pl_module.model.transformer
                if hasattr(pl_module, 'transformer'):
                    transformer = pl_module.transformer
                elif hasattr(pl_module, 'seq_pred') and hasattr(pl_module.seq_pred, 'transformer'):
                    transformer = pl_module.seq_pred.transformer
                else:
                    return
                
                # 调用我们在 transformer_prefix.py 中定义的方法
                if hasattr(transformer, 'check_prefix_grads'):
                    transformer.check_prefix_grads()
            except Exception as e:
                print(f"⚠️ Gradient Monitor Error: {e}")


def main():
    TRAIN_PATH = "data/processed/prefix_training_data_final_augmented.txt"
    pretrained_ckpt = r"data/output/pgd_diff/both_dual_encoder/last.ckpt"
    output_dir = "models/prefix_tuned_hybrid_v2"
    os.makedirs(output_dir, exist_ok=True)

    cancer_to_idx = build_cancer_mapping(TRAIN_PATH)
    num_cancers = len(cancer_to_idx)
    print(f"癌症类别数: {num_cancers}")

    NEG_LOSS_WEIGHT = 1.0   
    model = PGD_Diff.load_from_checkpoint(
        pretrained_ckpt,
        num_cancer_types=num_cancers,
        condition_mode='hybrid',
        prefix_len=10,
        loss_weight=1.0,
        neg_loss_weight=NEG_LOSS_WEIGHT,
        strict=False,
    )
    print(f"✅ 模型加载成功")

    # ========== 冻结所有参数，然后选择性解冻 ==========
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_keywords = [
        'prefix_k', 'prefix_v',
        'cancer_embedding', 'condition_mlp', 'uncond_embedding',
        'classifier', '_aa_embedding',
        'transformer',
        'seq_ffn',
    ]

    thawed_count = 0
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True
            thawed_count += 1

    print(f"✅ 已解冻 {thawed_count} 组参数")

    # ========== 定义 configure_optimizers ==========
    def configure_optimizers():
        prefix_params = []
        condition_params = []
        transformer_params = []
        seq_ffn_params = []
        other_trainable_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'prefix_k' in name or 'prefix_v' in name:
                prefix_params.append(param)
            elif 'condition_mlp' in name or 'cancer_embedding' in name:
                condition_params.append(param)
            elif 'transformer' in name and 'prefix' not in name:
                transformer_params.append(param)
            elif 'seq_ffn' in name:
                seq_ffn_params.append(param)
            else:
                other_trainable_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': prefix_params, 'lr': 1e-3, 'weight_decay': 0.0}, 
            {'params': condition_params, 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': transformer_params, 'lr': 5e-5, 'weight_decay': 1e-4},
            {'params': seq_ffn_params, 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': other_trainable_params, 'lr': 1e-4, 'weight_decay': 1e-4},
        ])
        return optimizer

    model.configure_optimizers = configure_optimizers

    train_dataset = PrefixPeptideDataset(TRAIN_PATH, cancer_to_idx, max_seq_len=50)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
    print(f"\n📦 训练集: {len(train_dataset)} 条")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor='total_loss',      
        mode='min',
        filename='epoch_{epoch:02d}_loss_{total_loss:.4f}',
        save_top_k=10,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='total_loss',      
        patience=20,
        mode='min',
        verbose=True,
    )

    # 🔥 使用新的监控 Callback
    gradient_monitor = AdvancedPrefixMonitor()

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stop_callback, gradient_monitor],  
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    print("\n" + "="*60)
    print("🚀 开始 Hybrid Prefix-Tuning 训练 (带全层梯度监控)")
    print("="*60)

    trainer.fit(model, train_loader)

    with open(os.path.join(output_dir, "cancer_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(cancer_to_idx, f, ensure_ascii=False, indent=2)
    print(f"✅ 标签映射已保存至: {os.path.join(output_dir, 'cancer_to_idx.json')}")
    print(f"\n✅ 训练完成！")


if __name__ == '__main__':
    main()


