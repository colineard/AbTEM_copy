# 4D-STEM of MoS2 including a ptychographic phase reconstruction.
import matplotlib.pyplot as plt
import numpy as np
import abtem
from ase.build import mx2
from Sample_materials import get_atoms

energy = 200e3  # 加速电压 200 keV
extent = 20      # 波前大小 [Å]
gpts = 512       # 采样点数 (建议 256–1024)

# ========== 手动定义幅度和相位 ==========
# 创建十字形的幅度分布
amplitude = np.zeros((gpts, gpts), dtype=np.float32)

# 十字的宽度可以调节
cross_width = gpts // 20  # 例如 5% 图像宽度的十字

# 中心位置
cx, cy = gpts // 2, gpts // 2

# 水平竖直方向置 1
amplitude[cy - cross_width : cy + cross_width, :] = 1.0
amplitude[:, cx - cross_width : cx + cross_width] = 1.0

# 相位全为 0
phase = np.zeros_like(amplitude)

# 构造复数波函数 ψ = A * exp(iφ)
psi = amplitude * np.exp(1j * phase)



# ========== 创建 Wave 对象 ==========
# Wave 对象代表一个复数二维波函数
wave = abtem.Waves(
    array = psi,
    energy=energy,
    # extent=extent,
    sampling = 2* extent / gpts
)

# ========== 可视化 ==========
# 1. 用 abTEM 自带显示（方便快速预览）
wave.show()
# # 2. 或用 matplotlib 分别查看幅度和相位
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].imshow(np.abs(wave.array), cmap='gray')
# axes[0].set_title("Amplitude (Cross pattern)")
# axes[1].imshow(np.angle(wave.array), cmap='twilight')
# axes[1].set_title("Phase (all zeros)")
plt.show()



# probetest = abtem.Probe(
#         # sampling = 0.05,
#         # # extent = 20,
#         # gpts = 512,
#         energy = 80e3,
#         semiangle_cutoff = 20,
# )
# probetest.show()
# plt.show()
# a = 1


atoms = get_atoms('MoS2')

potential = abtem.Potential(atoms, sampling = 0.02)

potential_array = potential.build().compute()

potential_array.show()

potential_array.to_zarr("MoS2_potential.zarr", overwrite=True)

potential2 = abtem.from_zarr("MoS2_potential.zarr")

potential2.show()
a= 1