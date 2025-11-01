import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 配置参数
# -------------------------------
PREFIX = 'Normal'
MODEL_PATH = f'./checkpoints/informer_{PREFIX}_ftMS_sl500_ll50_pl50_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebfixed_dtTrue_mxTrue_Exp_fixed_2/checkpoint.pth'
DATA_PATH = f'./data/FLEA/{PREFIX}.csv'
OUTPUT_PLOT = f'./plots/prediction_{PREFIX}_multivariate.png'
TITLE = f'{PREFIX} Prediction Prediction Result'

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

SEQ_LEN = 500
LABEL_LEN = 50
PRED_LEN = 50
INPUT_DIM = 7   # 7个输入特征
OUTPUT_DIM = 1  # 只预测1个目标

os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# -------------------------------
# 1. 加载并预处理数据
# -------------------------------
print("🚀 加载数据...")

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

# 所有数值列（假设共7列，最后一列是目标）
all_cols = [col for col in df.columns if col != 'date']
if len(all_cols) != 7:
    raise ValueError(f"期望7列特征，但实际有 {len(all_cols)} 列。请检查数据！")

feature_cols = all_cols          # 全部7列用于输入
target_col = 'Motor Y Voltage'   # 明确指定目标列（应为最后一列）

if target_col not in feature_cols:
    raise ValueError(f"目标列 '{target_col}' 不在数据中！可用列: {feature_cols}")

print(f"✅ 使用全部 {len(feature_cols)} 列作为输入")
print(f"🎯 目标变量: {target_col}")

raw_data = df[feature_cols].values.astype(np.float32)        # (N, 7)
target_data = df[target_col].values.reshape(-1, 1).astype(np.float32)  # (N, 1)

# 对所有7个输入特征分别归一化（包括目标列也作为输入的一部分）
scalers = {}
scaled_data = np.zeros_like(raw_data)
for i, col in enumerate(feature_cols):
    scaler = MinMaxScaler()
    scaled_data[:, i:i+1] = scaler.fit_transform(raw_data[:, i:i+1])
    scalers[col] = scaler

# 单独对目标变量再做一次归一化（用于反变换预测结果）
target_scaler = MinMaxScaler()
target_scaler.fit(target_data)  # 注意：只拟合原始目标值

print(f"输入数据形状: {scaled_data.shape}")
print(f"目标数据形状: {target_data.shape}")

# -------------------------------
# 2. 构建测试集
# -------------------------------
def create_inference_dataset(data, target, seq_len, pred_len, step=None):
    if step is None:
        step = pred_len
    X, Y = [], []
    for i in range(0, len(data) - seq_len - pred_len + 1, step):
        X.append(data[i:i + seq_len])                     # (seq_len, 7)
        Y.append(target[i + seq_len : i + seq_len + pred_len, 0])  # (pred_len,)
    return np.array(X), np.array(Y)

X_val, Y_true = create_inference_dataset(scaled_data, target_data, SEQ_LEN, PRED_LEN)
X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
Y_true = torch.tensor(Y_true, dtype=torch.float32).to(DEVICE)

print(f"X_val shape: {X_val.shape}")   # (B, 500, 7)
print(f"Y_true shape: {Y_true.shape}") # (B, 50)

# -------------------------------
# 3. 构造解码器输入 x_dec（7维）
# -------------------------------
B = X_val.shape[0]
dec_inp = torch.zeros(B, PRED_LEN, INPUT_DIM).to(DEVICE)
x_dec = torch.cat([X_val[:, -LABEL_LEN:, :], dec_inp], dim=1)  # (B, 100, 7)

# -------------------------------
# 4. 加载模型（c_out=1）
# -------------------------------
from models.model import Informer

model = Informer(
    enc_in=INPUT_DIM,
    dec_in=INPUT_DIM,
    c_out=OUTPUT_DIM,  # ← 关键：输出只有1维
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

print("📥 加载模型权重...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("✅ 模型加载成功！")

# -------------------------------
# 5. 分批推理
# -------------------------------
BATCH_SIZE_INF = 32
preds_list = []

with torch.no_grad():
    for i in range(0, len(X_val), BATCH_SIZE_INF):
        x_enc_batch = X_val[i:i+BATCH_SIZE_INF]
        B_batch = x_enc_batch.shape[0]

        dec_inp_batch = torch.zeros(B_batch, PRED_LEN, INPUT_DIM).to(DEVICE)
        x_dec_batch = torch.cat([x_enc_batch[:, -LABEL_LEN:, :], dec_inp_batch], dim=1)

        x_mark_enc = torch.zeros(B_batch, SEQ_LEN, 5, dtype=torch.long).to(DEVICE)
        x_mark_dec = torch.zeros(B_batch, LABEL_LEN + PRED_LEN, 5, dtype=torch.long).to(DEVICE)

        pred = model(x_enc_batch, x_mark_enc, x_dec_batch, x_mark_dec)  # (B, 50, 1)
        preds_list.append(pred.cpu())

# 合并预测结果
preds = torch.cat(preds_list, dim=0)  # (B, 50, 1)
preds = preds.squeeze(-1).numpy()     # (B, 50)
trues = Y_true.cpu().numpy()          # (B, 50)

# 展平
pred_flat = preds.reshape(-1, 1)      # (B*50, 1)
true_flat = trues.reshape(-1, 1)      # (B*50, 1)

# 反归一化（使用 target_scaler）
pred_original = target_scaler.inverse_transform(pred_flat).flatten()
true_original = target_scaler.inverse_transform(true_flat).flatten()

print(f"预测长度: {len(pred_original)}")

# -------------------------------
# 6. 绘图（仅目标变量）
# -------------------------------
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

plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"✅ 图像已保存至: {OUTPUT_PLOT}")
plt.show()

# -------------------------------
# 7. 保存 CSV
# -------------------------------
result_df = pd.DataFrame({
    'True': true_original[:N_SHOW],
    'Predicted': pred_original[:N_SHOW]
})
result_csv = OUTPUT_PLOT.replace('.png', '.csv')
result_df.to_csv(result_csv, index=False)
print(f"✅ 预测结果已保存至: {result_csv}")