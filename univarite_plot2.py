import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime

# -------------------------------
# 1. 配置参数
# -------------------------------

# 共同的标识名称
PREFIX = 'Spall'
# 模型路径
MODEL_PATH = f'./checkpoints/informer_{PREFIX}_ftS_sl500_ll50_pl50_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebfixed_dtTrue_mxTrue_Exp_fixed_2/checkpoint.pth'
# 数据路径
DATA_PATH = f'./data/FLEA/{PREFIX}.csv'
# 输出图像保存路径
OUTPUT_PLOT = f'./plots/prediction_{PREFIX}_univariate.png'
# 图像标题
TITLE = f'{PREFIX} Univariate Prediction Result'
SKIP_LINES = 0
if PREFIX == 'Jam':
    SKIP_LINES = 9119

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 模型超参数（必须与训练时一致）
SEQ_LEN = 500      # 输入序列长度
LABEL_LEN = 50     # 解码器引导长度
PRED_LEN = 50      # 预测长度
INPUT_DIM = 1      # 单变量输入

# 确保目录存在
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# -------------------------------
# 2. 加载并预处理数据
# -------------------------------
print("🚀 开始加载数据...")

# 读取数据
df = pd.read_csv(DATA_PATH)

# 确保 'date' 列是时间类型（保留毫秒）
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
# 如果格式不统一，可用：df['date'] = pd.to_datetime(df['date'])

# 检查时间间隔
time_diff = df['date'].diff().dropna().dt.total_seconds().unique()
print(f"时间间隔（秒）: {time_diff[:5]}")

# >>>>>>>>>> Jam 忽略前 9119 行 <<<<<<<<<<
df = df.iloc[SKIP_LINES:].reset_index(drop=True)
print(f"已跳过前 {SKIP_LINES} 行，当前数据长度: {len(df)}")

# 提取目标列
target_col = 'Motor Y Voltage'  # 修改为你的列名
if target_col not in df.columns:
    raise ValueError(f"列 '{target_col}' 不存在！可用列: {list(df.columns)}")

raw_data = df[target_col].values.reshape(-1, 1).astype(np.float32)

# 归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(raw_data)  # (N, 1)

print(f"数据形状: {scaled_data.shape}")
print(f"归一化范围: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")

# -------------------------------
# 3. 构建测试集
# -------------------------------
def create_inference_dataset(data, seq_len, label_len, pred_len, step=None):
    """
    构建滑动窗口测试集
    """
    if step is None:
        step = pred_len  # 默认步长为预测长度

    X, Y = [], []
    for i in range(0, len(data) - seq_len - pred_len + 1, step):
        X.append(data[i:i + seq_len])  # (seq_len, 1)
        Y.append(data[i + seq_len : i + seq_len + pred_len, 0])  # (pred_len,)
    return np.array(X), np.array(Y)

print("🔧 构建测试集...")
X_val, Y_true = create_inference_dataset(
    scaled_data,
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    step=PRED_LEN
)

X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
Y_true = torch.tensor(Y_true, dtype=torch.float32).to(DEVICE)

print(f"X_val shape: {X_val.shape}")  # (B, 500, 1)
print(f"Y_true shape: {Y_true.shape}")  # (B, 50)

# -------------------------------
# 4. 构造解码器输入 x_dec
# -------------------------------
B = X_val.shape[0]

# 解码器输入：最后 LABEL_LEN 个真实值 + PRED_LEN 个 0
dec_inp = torch.zeros(B, PRED_LEN, INPUT_DIM).to(DEVICE)
x_dec = torch.cat([X_val[:, -LABEL_LEN:, :], dec_inp], dim=1)  # (B, LABEL_LEN + PRED_LEN, 1)

print(f"x_dec shape: {x_dec.shape}")

# -------------------------------
# 5. 加载模型（假设模型定义已导入）
# -------------------------------
# 注意：你需要确保 Informer 模型类已定义
# 如果你使用的是官方代码，请确保导入了 model 和 configs

# 由于我们无法直接导入 main_informer 中的模型，
# 这里我们假设你知道模型结构，手动定义或从训练代码中复制

from models.model import Informer  # 根据你的项目结构调整路径

model = Informer(
    enc_in=INPUT_DIM,
    dec_in=INPUT_DIM,
    c_out=1,
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    factor=5,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    dropout=0.05,
    attn='prob',
    embed='fixed',           # ✅ 必须与训练一致
    freq='t',                # 任意值（fixed 时无效）
    activation='gelu'
).to(DEVICE)

# 安全加载权重
print("📥 加载模型权重...")
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print("✅ 模型权重加载成功！")
else:
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

model.eval()

# -------------------------------
# 6. 推理
# -------------------------------
print("🔮 开始推理...")

# ❌ 错误：全部加载
# X_val = torch.tensor(...).to(DEVICE)

# ✅ 正确：分批推理
BATCH_SIZE_INF = 32  # 每次只处理 4 个样本

preds_list = []
trues_list = []

model.eval()
with torch.no_grad():
    for i in range(0, len(X_val), BATCH_SIZE_INF):
        x_enc_batch = X_val[i:i+BATCH_SIZE_INF].to(DEVICE)
        y_true_batch = Y_true[i:i+BATCH_SIZE_INF]

        # 构造 x_dec 和 x_mark（每批都构造）
        B = x_enc_batch.shape[0]
        dec_inp = torch.zeros(B, PRED_LEN, INPUT_DIM).to(DEVICE)
        x_dec = torch.cat([x_enc_batch[:, -LABEL_LEN:, :], dec_inp], dim=1)

        x_mark_enc = torch.zeros(B, SEQ_LEN, 5, dtype=torch.long).to(DEVICE)
        x_mark_dec = torch.zeros(B, LABEL_LEN + PRED_LEN, 5, dtype=torch.long).to(DEVICE)

        pred = model(x_enc_batch, x_mark_enc, x_dec, x_mark_dec)  # (B, 50, 1)

        preds_list.append(pred.cpu())
        trues_list.append(y_true_batch.cpu())

# 拼接结果
preds = torch.cat(preds_list, dim=0)
trues = torch.cat(trues_list, dim=0)

# 移到 CPU 并转为 numpy
preds = preds.cpu().numpy().reshape(-1)        # (B * 50,)
trues = Y_true.cpu().numpy().reshape(-1)       # (B * 50,)

# 反归一化
preds = preds.reshape(-1, 1)
trues = trues.reshape(-1, 1)

pred_original = scaler.inverse_transform(preds).flatten()
true_original = scaler.inverse_transform(trues).flatten()

print(f"预测数据长度: {len(pred_original)}")
print(f"真实数据长度: {len(true_original)}")

# -------------------------------
# 7. 可视化结果（仅显示前 2000 点）
# -------------------------------
print("📊 绘制结果（仅前 2000 个点）...")

N_SHOW = 2000
pred_plot = pred_original[:N_SHOW]
true_plot = true_original[:N_SHOW]

plt.figure(figsize=(8, 6))
plt.plot(true_plot, label='True Value', color='#003f5c', linewidth=2)
plt.plot(pred_plot, label='Predicted', color='#ffa600', linewidth=1.5, alpha=0.9)

plt.title(TITLE, fontsize=16, pad=20)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Motor Y Voltage (V)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存图像
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"✅ 图像已保存至: {OUTPUT_PLOT}")

plt.show()

# -------------------------------
# 8. 保存预测结果到 CSV（仅前 2000 行）
# -------------------------------
result_df = pd.DataFrame({
    'True': true_plot,
    'Predicted': pred_plot
})
result_csv = OUTPUT_PLOT.replace('.png', '.csv')
result_df.to_csv(result_csv, index=False)
print(f"✅ 预测结果已保存至: {result_csv}")