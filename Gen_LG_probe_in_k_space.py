import ase
import matplotlib.pyplot as plt
import numpy as np
import abtem
from scipy.special import genlaguerre
from scipy.constants import h, m_e, e, c

def calculate_relativistic_wavelength(energy_ev):
    """计算给定能量下电子的相对论波长 [Å]。"""
    energy_J = energy_ev * e
    term_in_sqrt = 2 * m_e * energy_J * (1 + energy_J / (2 * m_e * c**2))
    wavelength_m = h / np.sqrt(term_in_sqrt)
    return wavelength_m * 1e10
# ------------------- 1. 参数定义 -------------------
# LG模式参数
l = 5  # 拓扑荷数 (决定相位)
p = 4  # 径向模数 (决定振幅的环数)

# 模拟空间和电镜参数
energy = 80e3
extent = 20
sampling = 0.05
gpts = int(round(extent / sampling))
semiangle_cutoff = 15 # mrad

# ------------------- 2. 创建计算网格 (纯NumPy) -------------------
# 创建倒易空间坐标 kx, ky [1/Å]
k_axis = np.fft.fftshift(np.fft.fftfreq(gpts, d=sampling))
kx, ky = np.meshgrid(k_axis, k_axis, indexing='ij')

# 转换为极坐标
rho_k = np.sqrt(kx**2 + ky**2)
phi_k = np.arctan2(ky, kx)

# ------------------- 3. 计算振幅和相位 -------------------

# -- A. 计算振幅 (Amplitude) --
# 首先，定义倒易空间的束腰宽度 w0
wavelength = calculate_relativistic_wavelength(energy)
k_max = (semiangle_cutoff / 1000.) / wavelength
w0 = k_max

# 振幅由三部分相乘得到：
# 1. 高斯包络部分
gaussian_part = np.exp(-rho_k**2 / w0**2)
# 2. 径向 r^|l| 部分
radial_part = (np.sqrt(2) * rho_k / w0)**abs(l)
# 3. 拉盖尔多项式部分
norm_rho_squared = 2 * rho_k**2 / w0**2
laguerre_part = genlaguerre(p, abs(l))(norm_rho_squared)
# # 分别绘制三个部分的幅度图像：
# fig_parts, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
# extent_plot = [-k_max, k_max, -k_max, k_max]
# im1 = ax1.imshow(gaussian_part, cmap='viridis', extent=extent_plot)
# ax1.set_title('高斯包络部分')
# ax1.set_xlabel('$k_x$ [1/Å]')
# ax1.set_ylabel('$k_y$ [1/Å]')
# fig_parts.colorbar(im1, ax=ax1)

# im2 = ax2.imshow(radial_part, cmap='viridis', extent=extent_plot)
# ax2.set_title('径向 $r^{|l|}$ 部分')
# ax2.set_xlabel('$k_x$ [1/Å]')
# ax2.set_ylabel('$k_y$ [1/Å]')
# fig_parts.colorbar(im2, ax=ax2)

# im3 = ax3.imshow(laguerre_part, cmap='viridis', extent=extent_plot)
# ax3.set_title('拉盖尔多项式部分')
# ax3.set_xlabel('$k_x$ [1/Å]')
# ax3.set_ylabel('$k_y$ [1/Å]')
# fig_parts.colorbar(im3, ax=ax3)
# fig_parts.show()



# 组合成总振幅
total_amplitude = radial_part * gaussian_part * laguerre_part


# -- B. 计算相位 (Phase) --
# 相位完全由拓扑荷数 l 和方位角 phi_k 决定
total_phase = np.exp(1j * l * phi_k)


# ------------------- 4. 合并为复数波函数 -------------------

# 将振幅和相位凑成一个复数数组，得到倒易空间的波函数
reciprocal_space_array = total_amplitude * total_phase

# 应用硬孔径，将超出范围的区域振幅设为0
reciprocal_space_array[rho_k > k_max] = 0

# ------------------- 5. 获得最终的实空间探针 -------------------

# 通过逆傅里叶变换得到最终在样品平面上的探针波函数
final_lg_wave = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(reciprocal_space_array)
    )
)

print("已成功生成LG波函数的NumPy复数数组。")
print(f"数组形状: {final_lg_wave.shape}, 数据类型: {final_lg_wave.dtype}")

# ------------------- 6. 可视化验证 -------------------
intensity = np.abs(final_lg_wave)**2
phase = np.angle(final_lg_wave)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
plot_extent = [-extent / 2, extent / 2, -extent / 2, extent / 2]
# 强度图
im1 = ax1.imshow(intensity, extent=plot_extent)
ax1.set_title(f'Probe Intensity (l={l}, p={p})')
ax1.set_xlabel('x [Å]')
ax1.set_ylabel('y [Å]')
fig.colorbar(im1, ax=ax1)

# 相位图
im2 = ax2.imshow(phase, cmap='hsv', extent=plot_extent)
ax2.set_title(f'Probe phase (l={l}, p={p})')
ax2.set_xlabel('x [Å]')
ax2.set_ylabel('y [Å]')
fig.colorbar(im2, ax=ax2)

fig.tight_layout()
plt.show()

# -------------------【补充】可视化倒易空间的波函数 -------------------

# 我们要可视化的变量是 reciprocal_space_array
amplitude_reciprocal = np.abs(reciprocal_space_array)
phase_reciprocal = np.angle(reciprocal_space_array)

# 创建一个新的figure用于显示
fig_reciprocal, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(11, 5))
fig_reciprocal.suptitle('倒易空间 (Reciprocal Space) - "设计图"', fontsize=16)

# --- 绘制倒易空间的振幅 ---
# 定义坐标轴的范围，单位是 1/Å
extent_k = [k_axis.min(), k_axis.max(), k_axis.min(), k_axis.max()]
# extent_k = [-k_max, k_max, -k_max, k_max]
im_r1 = ax_r1.imshow(amplitude_reciprocal, cmap='viridis', extent=extent_k)
ax_r1.set_title(f'振幅 (l={l}, p={p})')
ax_r1.set_xlabel('$k_x$ [1/Å]')
ax_r1.set_ylabel('$k_y$ [1/Å]')
fig_reciprocal.colorbar(im_r1, ax=ax_r1)

# --- 绘制倒易空间的相位 ---
# 创建一个蒙版(mask)，只显示孔径内部有意义的相位
phase_masked = np.copy(phase_reciprocal)
phase_masked[amplitude_reciprocal == 0] = np.nan # 将孔径外部设为nan，imshow会将其显示为白色

im_r2 = ax_r2.imshow(phase_masked, cmap='hsv', extent=extent_k)
ax_r2.set_title(f'相位 (l={l})')
ax_r2.set_xlabel('$k_x$ [1/Å]')
ax_r2.set_ylabel('$k_y$ [1/Å]')
fig_reciprocal.colorbar(im_r2, ax=ax_r2)

fig_reciprocal.tight_layout(rect=[0, 0.03, 1, 0.95])


# ------------------- 显示所有图像 -------------------
# 原有的 plt.show() 会将上面的新图和之前的实空间图一起显示出来
plt.show()
a  = 1  
