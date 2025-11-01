import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 配置（请根据实际情况调整路径）
# -------------------------------
# 共同的标识名称
PREFIX = 'Spall'
# 模型路径
MODEL_PATH = f'./checkpoints/informer_{PREFIX}_ftMS_sl500_ll50_pl50_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebfixed_dtTrue_mxTrue_Exp_fixed_2/checkpoint.pth'
# 数据路径
DATA_PATH = f'./data/FLEA/{PREFIX}.csv'
# 输出图像保存路径
OUTPUT_PLOT = f'./plots/prediction_{PREFIX}_multivariate.png'
# 图像标题
TITLE = f'{PREFIX} Multivariate Prediction Result'
SKIP_LINES = 0
if PREFIX == 'Jam':
    SKIP_LINES = 9119

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

SEQ_LEN = 500
LABEL_LEN = 50
PRED_LEN = 50
INPUT_DIM = 7   # 7个输入特征
OUTPUT_DIM = 1  # 单输出

# 确保目录存在
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# -------------------------------
# 1. 加载数据
# -------------------------------
print("🚀 加载数据...")
df = pd.read_csv(DATA_PATH)

# 自动识别特征列（排除 'date'）
cols = [col for col in df.columns if col != 'date']
if len(cols) != INPUT_DIM:
    raise ValueError(f"期望 {INPUT_DIM} 列特征，但实际有 {len(cols)} 列: {cols}")

target_col = 'Motor Y Voltage'
if target_col not in cols:
    raise ValueError(f"目标列 '{target_col}' 不在数据中！可用列: {cols}")

print(f"✅ 特征列: {cols}")
print(f"🎯 目标列: {target_col} (应为最后一列)")

# >>>>>>>>>> Jam 忽略前 9119 行 <<<<<<<<<<
df = df.iloc[SKIP_LINES:].reset_index(drop=True)
print(f"已跳过前 {SKIP_LINES} 行，当前数据长度: {len(df)}")

# 提取原始数据
raw_features = df[cols].values.astype(np.float32)          # (N, 7)
raw_target = df[target_col].values.reshape(-1, 1).astype(np.float32)  # (N, 1)

# -------------------------------
# 2. 归一化（关键！）
# -------------------------------
# 对所有输入特征分别归一化（用于模型输入）
feature_scalers = {}
scaled_features = np.zeros_like(raw_features)
for i, col in enumerate(cols):
    scaler = MinMaxScaler()
    scaled_features[:, i:i+1] = scaler.fit_transform(raw_features[:, i:i+1])
    feature_scalers[col] = scaler

# ⚠️ 对目标变量单独归一化（仅用于反变换！）
target_scaler = MinMaxScaler()
target_scaler.fit(raw_target)  # ← 必须用原始值！

print("\n🔍 归一化验证:")
print(f"原始 {target_col} 范围: [{raw_target.min():.2f}, {raw_target.max():.2f}]")
print(f"Scaler 记录范围: [{target_scaler.data_min_[0]:.2f}, {target_scaler.data_max_[0]:.2f}]")
assert np.isclose(target_scaler.data_min_[0], raw_target.min(), atol=1e-3), "Scaler 范围不匹配！"

# -------------------------------
# 3. 构建推理数据集
# -------------------------------
def create_dataset(X, Y, seq_len, label_len, pred_len, step=None):
    if step is None:
        step = pred_len
    Xs, Ys = [], []
    for i in range(0, len(X) - seq_len - pred_len + 1, step):
        Xs.append(X[i:i + seq_len])                          # (seq_len, 7)
        Ys.append(Y[i + seq_len : i + seq_len + pred_len, 0])  # (pred_len,)
    return np.array(Xs), np.array(Ys)

X_val, Y_true_raw = create_dataset(scaled_features, raw_target, SEQ_LEN, LABEL_LEN, PRED_LEN)
X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
Y_true_raw = Y_true_raw  # 保留原始值用于对比（未归一化）

print(f"\n📊 数据集形状: X_val={X_val.shape}, Y_true_raw={Y_true_raw.shape}")

# -------------------------------
# 4. 构造 decoder 输入 x_dec
# -------------------------------
B = X_val.shape[0]
dec_inp = torch.zeros(B, PRED_LEN, INPUT_DIM).to(DEVICE)
x_dec = torch.cat([X_val[:, -LABEL_LEN:, :], dec_inp], dim=1)  # (B, 100, 7)

# 时间特征（占位，若模型使用）
x_mark_enc = torch.zeros(B, SEQ_LEN, 5, dtype=torch.long).to(DEVICE)
x_mark_dec = torch.zeros(B, LABEL_LEN + PRED_LEN, 5, dtype=torch.long).to(DEVICE)

# -------------------------------
# 5. 加载模型
# -------------------------------
from models.model import Informer

model = Informer(
    enc_in=INPUT_DIM,
    dec_in=INPUT_DIM,
    c_out=OUTPUT_DIM,
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
    embed='fixed',
    freq='t',
    activation='gelu'
).to(DEVICE)

print("\n📥 加载模型权重...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 替换原来的权重检查部分
print("\n📥 加载模型权重...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# ✅ 通用参数检查（不再依赖 .linear）
all_params = [p for p in model.parameters() if p.numel() > 0]
if not all_params:
    raise RuntimeError("模型无有效参数！")

first_param = all_params[0]
print(f"✅ 模型加载成功！首个参数: mean={first_param.mean().item():.6f}, std={first_param.std().item():.6f}")

# -------------------------------
# 6. 推理
# -------------------------------
BATCH_SIZE = 32
preds_list = []

with torch.no_grad():
    for i in range(0, len(X_val), BATCH_SIZE):
        x_enc_batch = X_val[i:i+BATCH_SIZE]
        B_batch = x_enc_batch.shape[0]

        dec_inp_batch = torch.zeros(B_batch, PRED_LEN, INPUT_DIM).to(DEVICE)
        x_dec_batch = torch.cat([x_enc_batch[:, -LABEL_LEN:, :], dec_inp_batch], dim=1)

        x_mark_enc_batch = torch.zeros(B_batch, SEQ_LEN, 5, dtype=torch.long).to(DEVICE)
        x_mark_dec_batch = torch.zeros(B_batch, LABEL_LEN + PRED_LEN, 5, dtype=torch.long).to(DEVICE)

        pred = model(x_enc_batch, x_mark_enc_batch, x_dec_batch, x_mark_dec_batch)  # (B, 50, 1)
        preds_list.append(pred.cpu())

preds = torch.cat(preds_list, dim=0)  # (B, 50, 1)
preds = preds.squeeze(-1).numpy()     # (B, 50)

# 展平
pred_flat = preds.reshape(-1, 1)      # (B*50, 1)
true_flat = Y_true_raw.reshape(-1, 1) # (B*50, 1) —— 注意：这是原始值！

# -------------------------------
# 7. 调试输出（关键！）
# -------------------------------
print("\n🔍 推理结果调试:")
print(f"模型输出（归一化）范围: [{pred_flat.min():.6f}, {pred_flat.max():.6f}]")
print(f"模型输出均值: {pred_flat.mean():.6f}")

# 反归一化预测结果
pred_original = target_scaler.inverse_transform(pred_flat)
print(f"反归一化后预测范围: [{pred_original.min():.2f}, {pred_original.max():.2f}]")
print(f"真实值范围: [{true_flat.min():.2f}, {true_flat.max():.2f}]")

# 如果预测范围远小于真实值 → 模型没学好 或 scaler 错误

# -------------------------------
# 8. 绘图
# -------------------------------
N_SHOW = min(2000, len(pred_original))

plt.figure(figsize=(8, 6))
plt.plot(true_flat[:N_SHOW], label='True Value', color='#003f5c', linewidth=1.2)
plt.plot(pred_original[:N_SHOW], label='Predicted', color='#ffa600', linewidth=1.0, alpha=0.9)
plt.title(TITLE, fontsize=14)
plt.xlabel('Time Step')
plt.ylabel('Motor Y Voltage (V)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"\n✅ 图像已保存: {OUTPUT_PLOT}")
plt.show()

# -------------------------------
# 9. 保存 CSV
# -------------------------------
result_df = pd.DataFrame({
    'True': true_flat[:N_SHOW].flatten(),
    'Predicted': pred_original[:N_SHOW].flatten()
})

OUTPUT_CSV = OUTPUT_PLOT.replace('.png', '.csv')
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV 已保存: {OUTPUT_CSV}")

print("\n🎉 推理完成！请检查预测范围是否合理。")