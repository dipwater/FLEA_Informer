import numpy as np
from matplotlib import pyplot as plt

# 假设你保存了 loss 数据
train_losses = np.load('train_losses.npy')
val_losses = np.load('val_losses.npy')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='train_MSE-loss', color='blue')
plt.plot(val_losses, label='test_MSE-loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()