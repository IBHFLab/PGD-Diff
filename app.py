import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import json
import torch
import random
from pathlib import Path

# ================= 1. 添加项目根目录到 Python 路径 =================
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ================= 2. 导入现有生成脚本的函数 =================
# 通用模型：直接复用 generate_general_peptides.py 中的 load_model 和 CONFIG
from generate_general_peptides import load_model as load_general_model_raw, CONFIG

# 条件模型：直接复用 generate_conditional_peptides.py 中的逻辑
from models.prefix_tuned.PGD_Diff_condition import PGD_Diff as PGD_Diff_Condition

# ================= 3. Global Page Configuration =================
st.set_page_config(
    page_title="PGD-Diff: Targeted Anticancer Peptide Generation",
    page_icon="🧬",
    layout="wide"
)

# ================= 4. Custom CSS Styling =================
def add_custom_css():
    custom_css = """
    <style>
    .stButton > button {
        background: linear-gradient(135deg, #00CED1 0%, #4A90E2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 206, 209, 0.4);
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #444;
        background-color: #262730;
        color: #E0E0E0;
    }
    [data-testid="stDataFrame"] {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #333;
    }
    [data-testid="stSidebar"] {
        background-color: #181818;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 28px !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label div {
        font-size: 18px !important;
        line-height: 2.5 !important;
        padding-left: 5px !important;
    }
    [data-testid="stSidebar"] label p {
        font-size: 16px !important;
        margin-bottom: 8px !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()

# ================= 5. 加载癌症映射 =================
@st.cache_data
def load_cancer_mapping():
    """加载癌症类型到索引的映射"""
    map_path = PROJECT_ROOT / "models" / "prefix_tuned" / "cancer_to_idx.json"
    if map_path.exists():
        with open(map_path, 'r') as f:
            return json.load(f)
    map_path2 = PROJECT_ROOT / "cancer_to_idx.json"
    if map_path2.exists():
        with open(map_path2, 'r') as f:
            return json.load(f)
    return {
        "Breast": 0,
        "Lung": 1,
        "Colorectal": 2,
        "Liver": 3,
        "Leukemia": 4
    }

# ================= 6. 模型加载 =================
@st.cache_resource
def load_general_model():
    """
    加载通用生成模型
    直接复用 generate_general_peptides.py 中的 load_model 函数
    """
    try:
        ckpt_path = PROJECT_ROOT / CONFIG["checkpoint_path"]
        
        if not ckpt_path.exists():
            st.error(f"通用模型 checkpoint 未找到: {ckpt_path}")
            return None
        
        # 直接调用已验证的 load_model 函数
        model = load_general_model_raw(str(ckpt_path), CONFIG["model_config"])
        
        if model is not None:
            print("✅ 通用模型加载成功！")
            return model
        else:
            st.error("通用模型加载失败（load_model 返回 None）")
            return None
            
    except Exception as e:
        st.error(f"加载通用模型失败: {e}")
        return None

@st.cache_resource
def load_conditional_model():
    """
    加载条件生成模型（Prefix-Tuned）
    手动加载 checkpoint，过滤形状不匹配的键（容错）
    """
    try:
        from models.prefix_tuned.PGD_Diff_condition import PGD_Diff
        
        ckpt_path = PROJECT_ROOT / "models" / "prefix_tuned" / "last.ckpt"
        if not ckpt_path.exists():
            st.error(f"条件模型 checkpoint 未找到: {ckpt_path}")
            return None
        
        cancer_map = load_cancer_mapping()
        num_cancer_types = len(cancer_map)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. 创建模型实例
        model = PGD_Diff(
            use_prefix_condition=True,
            num_cancer_types=num_cancer_types,
            condition_mode='cross_attn',
            prefix_len=20,
            loss_weight=1.0
        )
        
        # 2. 加载 checkpoint
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        state_dict = checkpoint['state_dict']
        
        # 3. 过滤：只保留形状匹配的键（和通用模型一样的逻辑）
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not in model)")
        
        if skipped_keys:
            print(f"⚠️ 跳过了 {len(skipped_keys)} 个键")
        
        # 4. 加载过滤后的状态字典
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to(device)
        
        print("✅ 条件模型加载成功！")
        return model
        
    except Exception as e:
        st.error(f"加载条件模型失败: {e}")
        return None

# ================= 7. 生成函数 =================
def generate_general_peptides(model, length, num_sequences):
    """通用生成：调用 denoise_seq_sample"""
    if model is None:
        return None
    try:
        length_list = [length] * num_sequences
        sequences, _, _ = model.denoise_seq_sample(
            n_seq=num_sequences,
            seq_length=length_list,
            fasta_out_statue=False
        )
        return sequences
    except Exception as e:
        st.error(f"通用生成失败: {e}")
        return None

def generate_conditional_peptides(model, cancer_type, length, num_sequences, cancer_map):
    """条件生成：调用 denoise_seq_sample_with_prefix"""
    if model is None:
        return None
    try:
        cancer_idx = cancer_map.get(cancer_type, 0)
        length_list = [length] * num_sequences
        sequences, _, _ = model.denoise_seq_sample_with_prefix(
            cancer_idx=cancer_idx,
            n_seq=num_sequences,
            seq_length=length_list,
            fasta_out_statue=False
        )
        return sequences
    except Exception as e:
        st.error(f"条件生成失败: {e}")
        return None


# ================= 理化性质计算工具 =================
def calculate_peptide_properties(sequence):
    """
    计算肽段的理化性质
    返回字典：长度、分子量、净电荷(pH7)、平均疏水性、等电点
    """
    # 氨基酸分子量 (平均残基质量，单位 Da)
    aa_masses = {
        'A': 71.0788, 'R': 156.1875, 'N': 114.1038, 'D': 115.0886,
        'C': 103.1388, 'E': 129.1155, 'Q': 128.1307, 'G': 57.0519,
        'H': 137.1411, 'I': 113.1594, 'L': 113.1594, 'K': 128.1741,
        'M': 131.1926, 'F': 147.1766, 'P': 97.1167, 'S': 87.0782,
        'T': 101.1051, 'W': 186.2132, 'Y': 163.1760, 'V': 99.1326
    }
    
    # Kyte-Doolittle 疏水性标度
    kyte_doolittle = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # 带电氨基酸 pKa (用于 pH 7.0 电荷计算)
    # 侧链 pKa: D=3.9, E=4.3, H=6.0, C=8.3, K=10.5, R=12.5, Y=10.1
    # N-terminus pKa=9.6, C-terminus pKa=2.3
    seq = sequence.upper().strip()
    
    # 过滤非标准氨基酸
    valid_aa = set(aa_masses.keys())
    if not all(aa in valid_aa for aa in seq):
        invalid = [aa for aa in seq if aa not in valid_aa]
        return {"error": f"包含非标准氨基酸: {', '.join(set(invalid))}"}
    
    length = len(seq)
    
    # 1. 分子量 (加水分子 H2O = 18.015)
    mass = sum(aa_masses[aa] for aa in seq) + 18.015
    
    # 2. 净电荷 (pH 7.0)
    # 使用 Henderson-Hasselbalch 方程: 带电荷比例 = 1 / (1 + 10^(pH - pKa))
    pH = 7.0
    # 侧链带电基团
    charge_dict = {
        'D': (3.9, -1),  # 酸性，带负电
        'E': (4.3, -1),
        'H': (6.0, 1),   # 碱性，带正电
        'C': (8.3, -1),
        'K': (10.5, 1),
        'R': (12.5, 1),
        'Y': (10.1, 0),  # 酪氨酸在 pH7 基本不带电，忽略
    }
    net_charge = 0.0
    # N-terminus (正电)
    if length > 0:
        net_charge += 1 / (1 + 10**(pH - 9.6))
    # C-terminus (负电)
    if length > 0:
        net_charge -= 1 / (1 + 10**(9.6 - pH))
    # 侧链
    for aa in seq:
        if aa in charge_dict:
            pKa, val = charge_dict[aa]
            if val == 1:
                net_charge += 1 / (1 + 10**(pH - pKa))
            elif val == -1:
                net_charge -= 1 / (1 + 10**(pKa - pH))
    
    # 3. 平均疏水性 (Kyte-Doolittle)
    hydrophobicity = sum(kyte_doolittle.get(aa, 0) for aa in seq) / length
    
    # 4. 等电点 pI (粗略估计)
    pI = 7.0
    if net_charge > 0.1:
        pI = 7.0 + min(net_charge * 1.5, 4.0)
    elif net_charge < -0.1:
        pI = 7.0 - min(abs(net_charge) * 1.5, 4.0)
    pI = max(3.0, min(11.0, pI))
    
    return {
        "length": length,
        "molecular_weight": mass,
        "net_charge": net_charge,
        "hydrophobicity": hydrophobicity,
        "isoelectric_point": pI
    }


# ================= 8. 加载模型 =================
cancer_map = load_cancer_mapping()
general_model = load_general_model()
conditional_model = load_conditional_model()

# 显示加载状态
if general_model is not None:
    st.sidebar.success("✅ 通用模型已加载")
else:
    st.sidebar.warning("⚠️ 通用模型未加载")

if conditional_model is not None:
    st.sidebar.success("✅ 条件模型已加载")
else:
    st.sidebar.warning("⚠️ 条件模型未加载")

# ================= 9. Sidebar Navigation =================
st.sidebar.title("🧬 PGD-Diff")
page = st.sidebar.radio(
    "Select Module:",
    ["🎯 Broad-Spectrum Generation", "🎯 Cancer-Targeted Generation", 
     "📊 Physicochemical Visualization", "📖 About Model", 
     "📥 Dataset Download", "📞 Contact Us"]
)

# ================= 10. Page Content Routing =================

# --- 10.1 Broad-Spectrum Generation ---
if page == "🎯 Broad-Spectrum Generation":
    st.title("🎯 Broad-Spectrum Anticancer Peptide Generation")
    st.write("Input parameters to generate peptide sequences with broad-spectrum anticancer activity.")
    
    col1, col2 = st.columns(2)
    with col1:
        length = st.slider("Peptide Length", min_value=5, max_value=50, value=15)
        num = st.slider("Number of Sequences", min_value=1, max_value=20, value=5)
        
    if st.button("🚀 Generate Sequences"):
        with st.spinner("Generating sequences..."):
            if general_model is None:
                st.error("❌ 通用模型未加载，无法生成序列。请检查模型文件或重启应用。")
                st.stop()
            
            seqs = generate_general_peptides(general_model, length, num)
            if seqs is None:
                st.error("❌ 生成失败，请检查模型或稍后重试。")
                st.stop()
            
            st.success(f"Successfully generated {num} sequences!")
            st.dataframe(pd.DataFrame({"Sequence ID": range(1, num+1), "Amino Acid Sequence": seqs}))

# --- 10.2 Cancer-Targeted Generation ---
elif page == "🎯 Cancer-Targeted Generation":
    st.title("🎯 Cancer-Tissue Targeted Generation")
    st.write("Targeted generation of highly specific ACPs for specific cancer tissue microenvironments.")
    
    col1, col2 = st.columns(2)
    with col1:
        cancer_options = list(cancer_map.keys())
        cancer_type = st.selectbox("Select Cancer Tissue Type", cancer_options)
    with col2:
        length = st.slider("Peptide Length", min_value=5, max_value=50, value=18)
        
    num = st.slider("Number of Sequences", min_value=1, max_value=20, value=5)
    
    if st.button("🚀 Generate Targeted Sequences"):
        with st.spinner(f"Generating sequences for {cancer_type}..."):
            if conditional_model is None:
                st.error("❌ 条件模型未加载，无法生成序列。请检查模型文件或重启应用。")
                st.stop()
            
            seqs = generate_conditional_peptides(
                conditional_model, cancer_type, length, num, cancer_map
            )
            if seqs is None:
                st.error("❌ 生成失败，请检查模型或稍后重试。")
                st.stop()
            
            st.success(f"Successfully generated {num} sequences targeting {cancer_type}!")
            st.dataframe(pd.DataFrame({"Sequence ID": range(1, num+1), "Amino Acid Sequence": seqs}))

# --- 10.3 Physicochemical Visualization ---
elif page == "📊 Physicochemical Visualization":
    st.title("📊 Physicochemical Property Visualization")
    st.write("Enter one or more peptide sequences (one per line). The tool will calculate and compare their properties.")
    
    user_seq = st.text_area(
        "Enter Amino Acid Sequences (one per line, e.g., ACDEFGHIKLMNPQRSTVWY)",
        "ACDEFGHIKLMNPQRSTVWY\nKALGGGGIKVK",
        height=200
    )
    
    if st.button("🔬 Analyze Sequences", type="primary"):
        # 解析输入：按行分割，去除空行
        lines = [line.strip() for line in user_seq.splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one peptide sequence.")
        else:
            # 对每条序列进行计算
            results = []
            error_seqs = []
            for seq in lines:
                result = calculate_peptide_properties(seq)
                if "error" in result:
                    error_seqs.append((seq, result["error"]))
                else:
                    # 保存原始序列和计算结果
                    result["sequence"] = seq
                    results.append(result)
            
            if not results:
                st.error("No valid sequences to analyze. Please check your input.")
                if error_seqs:
                    with st.expander("Show errors"):
                        for seq, err in error_seqs:
                            st.write(f"❌ {seq} → {err}")
                st.stop()
            
            # 如果有错误，显示警告
            if error_seqs:
                st.warning(f"⚠️ {len(error_seqs)} sequence(s) contained non-standard amino acids and were skipped.")
                with st.expander("Show skipped sequences"):
                    for seq, err in error_seqs:
                        st.write(f"❌ {seq} → {err}")
            
            # 构建 DataFrame（每条肽一行）
            df_prop = pd.DataFrame([
                {
                    "Sequence": r["sequence"][:30] + "..." if len(r["sequence"]) > 30 else r["sequence"],
                    "Length": r["length"],
                    "Mol. Weight (Da)": round(r["molecular_weight"], 2),
                    "Net Charge (pH 7)": round(r["net_charge"], 3),
                    "Avg Hydrophobicity": round(r["hydrophobicity"], 3),
                    "pI": round(r["isoelectric_point"], 2)
                }
                for r in results
            ])
            
            # 1. 显示表格
            st.subheader("📊 Property Summary")
            st.dataframe(df_prop, use_container_width=True, hide_index=True)
            
            # 2. 雷达图（多条肽，每条一条轨迹）
            if len(results) > 1:
                st.subheader("📈 Multi-Peptide Radar Chart")
                # 准备雷达图数据：归一化值
                radar_rows = []
                for i, r in enumerate(results):
                    radar_rows.append({
                        "Peptide": f"Peptide {i+1}",
                        "Property": "Mol. Weight",
                        "Value": r["molecular_weight"] / 5000
                    })
                    radar_rows.append({
                        "Peptide": f"Peptide {i+1}",
                        "Property": "Net Charge",
                        "Value": (r["net_charge"] + 5) / 10
                    })
                    radar_rows.append({
                        "Peptide": f"Peptide {i+1}",
                        "Property": "Hydrophobicity",
                        "Value": (r["hydrophobicity"] + 5) / 10
                    })
                    radar_rows.append({
                        "Peptide": f"Peptide {i+1}",
                        "Property": "pI",
                        "Value": r["isoelectric_point"] / 14
                    })
                df_radar = pd.DataFrame(radar_rows)
                
                fig_radar = px.line_polar(
                    df_radar,
                    r="Value",
                    theta="Property",
                    line_close=True,
                    color="Peptide",
                    title="Normalized Property Comparison",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_radar.update_traces(fill="toself", opacity=0.2)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                # 只有一条肽时，显示单条雷达图（与原一致）
                st.subheader("📈 Property Radar Chart")
                r = results[0]
                radar_data = {
                    "Property": ["Mol. Weight", "Net Charge", "Hydrophobicity", "pI"],
                    "Value": [
                        r["molecular_weight"] / 5000,
                        (r["net_charge"] + 5) / 10,
                        (r["hydrophobicity"] + 5) / 10,
                        r["isoelectric_point"] / 14
                    ]
                }
                df_radar_single = pd.DataFrame(radar_data)
                fig_radar_single = px.line_polar(
                    df_radar_single,
                    r="Value",
                    theta="Property",
                    line_close=True,
                    title="Normalized Property Radar Chart",
                    color_discrete_sequence=["#4A90E2"]
                )
                fig_radar_single.update_traces(fill="toself", opacity=0.3)
                st.plotly_chart(fig_radar_single, use_container_width=True)
            
            # 3. 全局氨基酸组成（所有肽汇总）
            st.subheader("🧬 Overall Amino Acid Composition")
            all_seq = "".join(r["sequence"].upper() for r in results)
            aa_counts = pd.Series(list(all_seq)).value_counts()
            df_aa = pd.DataFrame({
                "Amino Acid": aa_counts.index,
                "Count": aa_counts.values,
                "Percentage": (aa_counts.values / len(all_seq) * 100).round(1)
            })
            st.dataframe(df_aa, use_container_width=True, hide_index=True)
            
            fig_pie = px.pie(
                df_aa,
                values="Count",
                names="Amino Acid",
                title="Overall Amino Acid Composition",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 4. 说明
            with st.expander("📖 About These Properties"):
                st.markdown("""
                - **Molecular Weight**: Calculated from amino acid residue masses + H₂O
                - **Net Charge**: Estimated at pH 7.0 using side-chain pKa values
                - **Avg Hydrophobicity**: Kyte-Doolittle scale averaged over sequence
                - **Isoelectric Point (pI)**: Approximate pH where net charge is zero
                - **Radar Chart**: Normalized values for visual comparison (scaled to 0-1 range)
                - For multi-peptide input, each peptide appears as a separate curve in the radar chart.
                """)

# --- 10.4 About Model ---
elif page == "📖 About Model":
    st.title("📖 About PGD-Diff Model")
    st.markdown("""
    ### Model Overview
    We propose **Prefix-Guided Dual-path Diffusion (PGD-Diff)**, a multimodal diffusion framework that integrates a dual-path encoder with conditional prefix tuning.
    
    **1. Dual-Path Sequence and Structure Encoders:**
    For the sequence modality, we employ Transformer blocks with parallel global self-attention and local convolutions to simultaneously capture long-range residue dependencies and local motif information. For the structure modality, we design a hybrid encoder combining equivariant Graph Neural Networks (EGNN) with global Transformers.
    
    **2. Prefix-Guided Conditional Generation:**
    Building upon a pre-trained multimodal diffusion model, we introduce cancer type as conditional information. A lightweight prefix-tuning module injects type embeddings into the sequence denoising process, enabling cancer-type-specific ACP generation.
    """)
    
    st.subheader("Model Architecture")
    image_path = PROJECT_ROOT / "images" / "Figure 1.tif"
    try:
        if image_path.exists():
            st.image(str(image_path), caption="Architecture of the PGD-Diff Framework", width=800)
        else:
            st.warning(f"Image not found at: {image_path}")
    except Exception as e:
        st.error(f"Failed to load the image. Error: {e}")

# --- 10.5 Dataset Download ---
elif page == "📥 Dataset Download":
    st.title("📥 Dataset Download")
    st.write("Access high-quality datasets used for model training and validation, as well as generated peptide sequences from our model.")
    
    st.markdown("---")
    
    # ========== 训练数据集 ==========
    st.subheader("📊 Training Datasets")
    st.write("Balanced datasets of anticancer peptides (ACPs) and non-ACPs used for model training.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        acp_path = PROJECT_ROOT / "data" / "source" / "fasta" / "ACP.fasta"
        if acp_path.exists():
            with open(acp_path, 'r') as f:
                acp_data = f.read()
            st.download_button(
                label="📥 Download ACP Training Set (FASTA)",
                data=acp_data,
                file_name="ACP.fasta",
                mime="text/plain",
                key="acp_train"
            )
        else:
            st.warning("ACP.fasta not found")
    
    with col2:
        nonacp_path = PROJECT_ROOT / "data" / "source" / "fasta" / "nonACP.fasta"
        if nonacp_path.exists():
            with open(nonacp_path, 'r') as f:
                nonacp_data = f.read()
            st.download_button(
                label="📥 Download non-ACP Training Set (FASTA)",
                data=nonacp_data,
                file_name="nonACP.fasta",
                mime="text/plain",
                key="nonacp_train"
            )
        else:
            st.warning("nonACP.fasta not found")
    
    st.markdown("---")
    
    # ========== 通用生成数据集 ==========
    st.subheader("🧬 General Generation Dataset")
    st.write("10,000 peptides generated by the broad-spectrum generation model.")
    
    general_path = PROJECT_ROOT / "data" / "output" / "fasta" / "PGD_Diff_10000.fasta"
    if general_path.exists():
        with open(general_path, 'r') as f:
            general_data = f.read()
        st.download_button(
            label="📥 Download General Generated Dataset (FASTA)",
            data=general_data,
            file_name="PGD_Diff_10000.fasta",
            mime="text/plain",
            key="general_gen"
        )
        seq_count = general_data.count('>')
        st.caption(f"📊 Contains {seq_count} sequences")
    else:
        st.warning("PGD_Diff_10000.fasta not found")
    
    st.markdown("---")
    
    # ========== 条件生成数据集（按癌症类型） ==========
    st.subheader("🎯 Cancer-Targeted Generation Datasets")
    st.write("Peptides generated for nine specific cancer types using the conditional generation model.")
    
    cancer_files = [
        ("Blood", "generated_blood_peptides.txt"),
        ("Brain", "generated_brain_peptides.txt"),
        ("Breast", "generated_breast_peptides.txt"),
        ("Cervix", "generated_cervix_peptides.txt"),
        ("Colon", "generated_colon_peptides.txt"),
        ("Liver", "generated_liver_peptides.txt"),
        ("Lung", "generated_lung_peptides.txt"),
        ("Prostate", "generated_prostate_peptides.txt"),
        ("Skin", "generated_skin_peptides.txt")
    ]
    
    cols = st.columns(3)
    
    for idx, (cancer_name, file_name) in enumerate(cancer_files):
        col_idx = idx % 3
        with cols[col_idx]:
            file_path = PROJECT_ROOT / "data" / "output" / "fasta" / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    file_data = f.read()
                seq_count = len([line for line in file_data.split('\n') if line.strip()])
                st.download_button(
                    label=f"📥 {cancer_name} Cancer",
                    data=file_data,
                    file_name=file_name,
                    mime="text/plain",
                    key=f"cond_{cancer_name.lower()}"
                )
                st.caption(f"📊 {seq_count} sequences")
            else:
                st.caption(f"⚠️ {cancer_name} data not found")
    
    st.markdown("---")
    
    # ========== 全部数据集下载（打包） ==========
    st.subheader("📦 Download All Datasets")
    st.write("Download all generated datasets as a single ZIP archive (if available).")
    
    zip_path = PROJECT_ROOT / "data" / "output" / "fasta" / "all_generated_datasets.zip"
    if zip_path.exists():
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        st.download_button(
            label="📦 Download All Datasets (ZIP)",
            data=zip_data,
            file_name="all_generated_datasets.zip",
            mime="application/zip",
            key="all_zip"
        )
    else:
        st.info("💡 Tip: You can download individual datasets from the sections above.")
        st.caption("To create a ZIP package manually, compress all files in the `data/output/fasta/` directory.")

# --- 10.6 Contact Us ---
elif page == "📞 Contact Us":
    st.title("📞 Contact Us")
    
    st.markdown("""
    **Contact the PGD-Diff team** to share suggestions for improving the platform, or to contribute newly reported anticancer peptide sequences for future database curation.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👨‍🏫 Principal Investigator")
        
        liang_pic = PROJECT_ROOT / "images" / "liangPic.png"
        if liang_pic.exists():
            st.image(str(liang_pic), width=150)
        else:
            st.warning("Photo not found")
        
        st.markdown("""
        **Professor Guizhao Liang**  
        College of Bioengineering, Chongqing University  
        
        **E-mail:** gzliang@cqu.edu.cn  
        **Tel:** (86)23-65102507  
        
        **Address:**  
        Room 519, College of Bioengineering, Chongqing University  
        No.174 Shazhengjie, Shapingba, Chongqing, 400044, China
        """)
    
    with col2:
        st.subheader("👨‍💻 Developer")
        
        xiao_pic = PROJECT_ROOT / "images" / "xiaoPic.jpg"
        if xiao_pic.exists():
            st.image(str(xiao_pic), width=150)
        else:
            st.warning("Photo not found")
        
        st.markdown("""
        **Xiao Liang**  
        Ph.D. Candidate, PGD-Diff Development  
        
        **E-mail:**  
        20241901014@stu.cqu.edu.cn  
        1187517465@qq.com  
        
        **Address:**  
        Room 509, College of Bioengineering, Chongqing University  
        No.174 Shazhengjie, Shapingba, Chongqing, 400044, China
        """)
    
    st.markdown("---")
    st.markdown("""
    *We welcome collaboration and feedback from the research community. If you have any questions or suggestions, please feel free to contact us.*
    """)