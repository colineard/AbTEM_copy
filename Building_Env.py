import ase
import matplotlib.pyplot as plt
import numpy as np
import abtem
from scipy.special import genlaguerre
from ase.build import mx2

abtem.config.set({"local_diagnostics.progress_bar": False})

# 1. 定义物理参数
energy = 100e3  # 100 keV 电子能量
extent = 10.0   # 10 Å x 10 Å 的模拟区域
pixels = 512    # 网格尺寸将是 512x512


## ——————————————————————————创建一个普通的probe——————————————————————

probe = abtem.Probe(
    sampling=0.05,
    extent=20,
    energy=80e3,
    semiangle_cutoff=20,
    C10=50,
    Cs=-50e-6 * 1e10,
)

intensity = probe.build().intensity().compute()
phase = probe.build().phase()

# 绘制普通probe的强度和相位
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

intensity.show(cbar=True, title="probe intensity", ax=ax1)
phase.show(cbar=True, title="probe phase", cmap="hsluv", ax=ax2)

plt.tight_layout()

plt.show()



l = 4        # The quantum number (topological charge)

# ——————————————————————通过aperture创建一个vortex probe——————————————————————
# 1. Create a Vortex aperture using abtem.transfer.Vortex
vortex_aperture = abtem.transfer.Vortex(
    quantum_number=l, 
    semiangle_cutoff= 20
)

# 2. Create a Probe and pass the vortex aperture directly
# The probe's semiangle_cutoff should generally match the aperture's
probe_vortex = abtem.Probe(
    sampling=0.05,
    extent=20,
    energy=80e3,
    semiangle_cutoff=20,
    C10=50,
    Cs=-50e-6 * 1e10,
    aperture=vortex_aperture  # Pass the object here
)


# # 3. 绘制probe和probe_vortex的强度和相位
# waves2 = probe_vortex.build()
# intensity2 = waves2.intensity().compute()
# phase2 = waves2.phase().compute()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# intensity.show(ax=ax1, cbar=True, title=f'Probe Intensity')
# intensity2.show(ax=ax2, cbar=True, title=f'Probe Intensity (l={l}) using abtem.Vortex')
# fig.tight_layout()
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# phase.show(ax=ax1, cbar=True, cmap='hsv', title=f'Phase of standard probe')
# phase2.show(ax=ax2, cbar=True, cmap='hsv', title=f'Phase (l={l}) using abtem.Vortex')
# fig.tight_layout()
# plt.show()

# # 4. 计算并绘制衍射图样（频域）

# fig2,(ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# diffraction_patterns = probe.build().diffraction_patterns(max_angle=60)
# diffraction_patterns2 = probe_vortex.build().diffraction_patterns(max_angle=60)

# diffraction_patterns.show(ax = ax1,
#     cbar=True, units = "mrad", title="reciprocal space probe intensity"
# )
# diffraction_patterns2.show(ax = ax2,
#     cbar=True, units = "mrad", title="reciprocal space probe_vortex intensity"
# )
# fig2.tight_layout()
# plt.show()



## ——————————————————————通过Waves创建一个LG probe——————————————————————

# 拉盖尔-高斯模式参数
l = 5  # 拓扑荷数
p = 4  # 径向模数



##————————————————————————构建经典的二维材料样品——————————————————————————
# 创建一个MoS2
# Create hexagonal supercell of MoS2
structure = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.127,
                size=(1, 1, 1), vacuum=10)
structure.pbc = True

# Orthogonalize cell and rotate and repeat it to match experiment.
atoms = abtem.orthogonalize_cell(structure)
atoms.rotate(-90, 'z', rotate_cell=True)
atoms*=(4,4,1)

# Switching x and y cell axes and centering to get positive coordinates.
atoms.cell=[[22.031686, 0, 0], [0, 12.72, 0], [0.0, 0.0, 12]]
atoms.center()

# Create S vacancy.
del atoms[64]

abtem.show_atoms(atoms)


potential = abtem.Potential(atoms, sampling=0.05)


probe = probe_vortex
## ———————————————————————probe 作用样品———————————————————————————
probe.grid.match(potential)


grid_scan = abtem.GridScan(
    start = (0, 0),
    end = (10, 10),
    gpts = (10, 10),
)
detector = abtem.PixelatedDetector()

measurements = probe.scan(potential, scan=grid_scan, detectors=detector)



measurements.compute()
# a = measurements.array.max()

# filtered_measurements = measurements.gaussian_source_size(.3)

# center_of_mass = filtered_measurements.center_of_mass()

# interpolated_center_of_mass = center_of_mass.interpolate(0.05).tile((3, 2))

# interpolated_center_of_mass.show(cbar=True, vmax=0.04, vmin=0)

# plt.show()


# visualization = measurements.show(
#     explode=True,
#     figsize=(16, 5),
# )

plt.show()

a = 1



