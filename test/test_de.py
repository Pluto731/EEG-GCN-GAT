import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用正确的函数名
from src.feature.de import extract_de_features  # 改为实际存在的函数名

# 模拟一个 segment
np.random.seed(0)
fake_segment = np.random.randn(32, 256)  # 32 channels, 256 samples (2秒 @128Hz)

de_features = extract_de_features(fake_segment)

print("DE features shape:", de_features.shape)
print("DE features (first channel):", de_features[0])