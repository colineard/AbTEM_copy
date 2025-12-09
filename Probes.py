import ase
import matplotlib.pyplot as plt
import numpy as np
import abtem
from scipy.special import genlaguerre
from ase.build import mx2
from scipy.constants import h, m_e, e, c


###  接收来自main_process.py的参数，一般通过一个字符串说明需要的探针类型，如果字符串是“specific",则需要接受更多参数，其他情况未接受的参数设置默认值

def Gaussianprobe(**params):
    """
    Creates an abtem.Probe object using specified keyword arguments.

    This function starts with a set of default parameters and overrides them
    with any parameters provided by the user.

    Args:
        **params: Keyword arguments that will be passed to abtem.Probe.
                  Examples: energy, semiangle_cutoff, C10, Cs, etc.

    Returns:
        abtem.Probe: The configured probe object.
    """
    # 1. Define the default parameters
    default_params = {
        'sampling': 0.05,
        'gpts':512,
        'energy': 80e3,
        'semiangle_cutoff': 20,
        'C10': 50.,              # Defocus in Ångström
        'Cs': -50e-6 * 1e10      # Spherical aberration in Ångström (-50 µm)
    }

    # 2. Create a final dictionary of parameters. The user's provided `params`
    #    will override the defaults.
    final_params = default_params.copy()
    final_params.update(params)

    # 3. Create the Probe object by unpacking the final parameters dictionary
    probe = abtem.Probe(**final_params)
    # print(probe.__class__)
    # print(isinstance(probe, abtem.Probe))
    # print(isinstance(probe, abtem.Waves))
    return probe


def Specificprobe(**params):
    """
    Create an abTEM Waves object representing a Bessel beam probe.

    You can override parameters like:
      - sampling
      - extent
      - energy
      - k_r (radial frequency)
      - l (topological charge)

    Example:
        wave = Specificprobe(energy=200e3, l=1, k_r=0.5)
    """

    # 1. Default parameters
    default_params = {
        'sampling': 0.05,     # in Å
        'extent': 20,         # field size in Å
        'energy': 80e3,       # beam energy in eV
        'l': 1,               # topological charge
        'k_r': 0.8,           # radial wave number (controls ring radius)
    }

    # 2. Merge with user input
    p = default_params.copy()
    p.update(params)

    # 3. Generate coordinate grid
    x = np.arange(-p['extent']/2, p['extent']/2, p['sampling'])
    y = np.arange(-p['extent']/2, p['extent']/2, p['sampling'])
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)

    # 4. Define complex amplitude of Bessel beam
    #    wavefunction ψ(r,θ) = J_l(k_r r) * exp(i * l * θ)
    from scipy.special import jv  # Bessel function of the first kind
    psi = jv(p['l'], p['k_r'] * R) * np.exp(1j * p['l'] * THETA)

    # 5. Normalize (optional)
    psi /= np.sqrt(np.sum(np.abs(psi)**2))

    # 6. Convert to abtem.Waves (二维)
    wave = abtem.Waves(
        array=psi,              # 直接使用二维数组
        energy=p['energy'],
        sampling=p['sampling']
    )

    return wave

def Probe_with_aperture(**params):
    """
    Creates an abtem.Probe object using specified keyword arguments.

    This function starts with a set of default parameters and overrides them
    with any parameters provided by the user.

    Args:
        **params: Keyword arguments that will be passed to abtem.Probe.
                  Examples: energy, semiangle_cutoff, C10, Cs, etc.

    Returns:
        abtem.Probe: The configured probe object.
    """

    # default aperture is a vortex with 4 topological charge
    l = 4        
    vortex_aperture = abtem.transfer.Vortex(
        quantum_number=l,
        semiangle_cutoff=20
    )

    # 1. Define the default parameters
    default_params = {
        'sampling': 0.05,
        'extent': 20,
        'energy': 80e3,
        'semiangle_cutoff': 20,
        'C10': 50.,              # Defocus in Ångström
        'Cs': -50e-6 * 1e10,
        'aperture':  vortex_aperture      # Spherical aberration in Ångström (-50 µm)
    }

    # 2. Create a final dictionary of parameters. The user's provided `params`
    #    will override the defaults.
    final_params = default_params.copy()
    final_params.update(params)

    # 3. Create the Probe object by unpacking the final parameters dictionary
    probe = abtem.Probe(**final_params)

    return probe


def LGprobe(l: int, p: int, **params):
    """
    Creates a Laguerre-Gaussian (LG) vortex beam probe using abtem.Probe.

    This function generates a Laguerre-Gaussian beam characterized by
    azimuthal index `l` and radial index `p`. It starts with a set of
    default parameters and allows the user to override them.

    Args:
        l (int): Azimuthal index of the LG beam (topological charge).
        p (int): Radial index of the LG beam.
        **params: Additional keyword arguments for abtem.Probe.
                  Examples: energy, semiangle_cutoff, w0, Cs, C10, etc.

    Returns:
        abtem.Probe: The configured LG vortex beam probe object.
    """
    
    def calculate_relativistic_wavelength(energy_ev):
        """计算给定能量下电子的相对论波长 [Å]。"""
        energy_J = energy_ev * e
        term_in_sqrt = 2 * m_e * energy_J * (1 + energy_J / (2 * m_e * c**2))
        wavelength_m = h / np.sqrt(term_in_sqrt)
        return wavelength_m * 1e10
    # 1. Define the default parameters for the LG probe
    default_params = {
        'sampling': 0.05,
        'gpts': 512,
        'energy': 80e3,      # Beam waist in mrad
        # 'C10': 50.,              # Defocus in Ångström
        # 'Cs': -50e-6 * 1e10      # Spherical aberration in Ångström (-50 µm)
    }

    # 2. Create a final dictionary of parameters. The user's provided `params`
    #    will override the defaults.
    final_params = default_params.copy()
    final_params.update(params)

    # 1. 获取原始参数 (不要直接取 [0]，先获取整体)
    raw_gpts = final_params['gpts']
    raw_sampling = final_params['sampling']

    # 2. 标准化 gpts -> (ny, nx)
    # 判断是否为标量 (int, np.integer)
    if isinstance(raw_gpts, (int, np.integer)):
        ny = nx = int(raw_gpts)
    # 判断是否为序列 (tuple, list, np.ndarray)
    elif isinstance(raw_gpts, (tuple, list, np.ndarray)) and len(raw_gpts) == 2:
        ny, nx = int(raw_gpts[0]), int(raw_gpts[1])
    else:
        raise ValueError(f"Invalid gpts format: {raw_gpts}")

    # 3. 标准化 sampling -> (dy, dx)
    # 同样处理标量和序列
    if isinstance(raw_sampling, (int, float, np.floating, np.integer)):
        dy = dx = float(raw_sampling)
    elif isinstance(raw_sampling, (tuple, list, np.ndarray)):
        if len(raw_sampling) == 2:
            dy, dx = float(raw_sampling[0]), float(raw_sampling[1])
        elif len(raw_sampling) == 1: # 防止传入 [0.05] 这种单元素列表
            dy = dx = float(raw_sampling[0])
        else:
            raise ValueError(f"Invalid sampling format length: {raw_sampling}")
    else:
        raise ValueError(f"Invalid sampling format: {type(raw_sampling)}")

    # 4. 分别计算两个方向的坐标轴
    # 注意：np.fft.fftfreq 的第一个参数是点数，第二个是步长
    ky_axis = np.fft.fftshift(np.fft.fftfreq(ny, dy))
    kx_axis = np.fft.fftshift(np.fft.fftfreq(nx, dx))

    # 5. 生成网格
    # 注意：使用 indexing='ij' 时，返回顺序对应 (Axis 0, Axis 1)，通常对应 (ky, kx)
    ky, kx = np.meshgrid(ky_axis, kx_axis, indexing='ij')

    # 转换为极坐标
    rho_k = np.sqrt(kx**2 + ky**2)
    phi_k = np.arctan2(ky, kx)

    # ------------------- 3. 计算振幅和相位 -------------------
    semiangle_cutoff = final_params['semiangle_cutoff']  # mrad
    # -- A. 计算振幅 (Amplitude) --
    # 首先，定义倒易空间的束腰宽度 w0
    wavelength = calculate_relativistic_wavelength(final_params['energy'])
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



    # 组合成总振幅
    total_amplitude = radial_part * gaussian_part * laguerre_part


    # -- B. 计算相位 (Phase) --
    # 相位完全由拓扑荷数 l 和方位角 phi_k 决定
    total_phase = np.exp(1j * l * phi_k)

    # 将振幅和相位凑成一个复数数组，得到倒易空间的波函数
    reciprocal_space_array = total_amplitude * total_phase

    # 应用硬孔径，将超出范围的区域振幅设为0
    reciprocal_space_array[rho_k > k_max] = 0
    # 通过逆傅里叶变换得到最终在样品平面上的探针波函数
    final_lg_wave = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(reciprocal_space_array)
        )
    )
    # 4. 使用 abtem.Waves 创建波函数对象

    params_toinit = {
        'array': final_lg_wave,
        'energy': final_params['energy'],
        'sampling': final_params['sampling']
    }
    # final_params.pop('compare', None)  # 移除 compare 参数
    wave = abtem.Waves(**params_toinit)
    return wave



# .----------------------------------------------------------------------------.
# |                          MAIN PROBE FACTORY                                |
# '----------------------------------------------------------------------------'

def create_probe(probe_type: str, **kwargs):
    """
    Factory function to create and return an abtem.Probe object.

    This function receives parameters, typically from a main process, to generate
    a specified type of electron probe.

    Parameters
    ----------
    probe_type : str
        The type of probe to generate. Supported values (case-insensitive):
        - 'gaussian': A standard probe with spherical aberration and defocus.
        - 'lg'      : A Laguerre-Gaussian vortex beam. Requires 'l' and 'p' in kwargs.
        - 'specific': A probe defined entirely by the provided keyword arguments passed
                      directly to the abtem.Probe constructor.

    **kwargs : dict
        Additional parameters for the probe.
        - For 'lg': Must include 'l' (int) and 'p' (int). Can also override defaults
          like 'energy', 'semiangle_cutoff', 'w0', 'Cs', 'C10', etc.
        - For 'gaussian' and 'specific': Can include any valid keyword
          argument for the abtem.Probe constructor, e.g., 'energy', 'Cs',
          'C30' (2-fold astigmatism), 'C12' (defocus-astigmatism), etc.

    Returns
    -------
    abtem.Probe
        The generated electron probe object.
    
    Raises
    ------
    ValueError
        If the probe_type is unknown or if required parameters are missing.
    """
    # Make the selection case-insensitive
    probe_type = probe_type.lower()
    print(f"Received request for probe_type: '{probe_type}' with parameters: {kwargs}")


    if probe_type == 'gaussian':
        # Define default parameters for a standard probe
        params = {
            'sampling': 0.05,
            'gpts': 512,
            'energy': 80e3,
            'semiangle_cutoff': 20,
            'C10': 50,                # Defocus in Angstrom
            'Cs': -50e-6 * 1e10       # Spherical aberration in Angstrom
        }
        # Allow user to override any default with provided kwargs
        params.update(kwargs)
        
        print(f"Creating Gaussian probe with final parameters: {params}")
        return Gaussianprobe(**params)

    elif probe_type == 'lg':
        # Check for required parameters 'l' and 'p' for LG probes
        if 'l' not in kwargs or 'p' not in kwargs:
            print(f"Creating Laguerre-Gaussian probe with default l=5 , p=4")
        # Call the specialized LGprobe function with its specific arguments
        return LGprobe(l = 5, p = 4, **kwargs)

    elif probe_type == 'specific':
        if 'array' not in kwargs:
            print(f"Creating specific probe with default array")
        return Specificprobe(**kwargs)

    elif probe_type == 'aperture':
        if  'aperture' not in kwargs:
            print(f"Creating probe with default aperture")
        return Probe_with_aperture(**kwargs)
    
    else:
        # Handle unknown probe types
        supported_types = "'gaussian', 'lg', 'specific', 'aperture'"
        raise ValueError(f"Unknown probe_type: '{probe_type}'. Supported types are: {supported_types}.")


# .----------------------------------------------------------------------------.
# |                          EXAMPLE USAGE                                     |
# '----------------------------------------------------------------------------'

# This block demonstrates how `main_process.py` would call the `create_probe` function.
if __name__ == '__main__':
    

    # --- Example 1: Request a probe with aperture ---
    # wave = create_probe('aperture')
    
    # # --- Example 2: Request a Gaussian probe---
    # wave = create_probe('lg')

    # # --- Example 3: Request a Laguerre-Gaussian probe ---
    # wave = create_probe('lg', l=4, p=4)

    # --- Example 4: Request a 'specific' probe with custom aberrations ---
    # wave = create_probe('specific')


    # --- Example 5: Request a standard Gaussian probe ---
    # params = {
    #     'sampling': 0.05,
    #     'gpts': 512,
    #     'energy': 80e3,
    #     'semiangle_cutoff': 20,
    #     'C10': 0,                # Defocus in Angstrom
    #     'Cs': 0       # Spherical aberration in Angstrom
    # }
    # wave = create_probe('gaussian', **params)

    Uniform_params = {
        'sampling': 0.02,  # 每个像素的大小，单位Å
        'energy': 80e3,   # 统一的probe加速电压，单位eV
        'gpts': 1024,      # 格点数
        'semiangle_cutoff': 30, # 探针半角截止，单位mrad
    }
    wave = create_probe('lg', **Uniform_params)
    if wave.__class__.__name__ == 'Probe':
        wave = wave.build()
    # Visualize one of the created probes


    print("\nVisualizing the  probe...")
    probe_2d = wave.array
    # 3. 计算强度和相位
    intensity = np.abs(probe_2d)**2
    phase = np.angle(probe_2d)

    # --- 4. 智能相位掩膜 (Masking) ---
    # 相位在强度为0的地方全是噪声，把强度小于最大值 5% 的地方挖掉，不显示相位
    mask = intensity < (intensity.max() * 0.000005)
    phase_masked = np.ma.masked_where(mask, phase)

    # 5. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图：强度 (Intensity) ---
    im1 = axes[0].imshow(intensity, cmap='inferno', origin='lower')
    axes[0].set_title("Probe Intensity")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # --- 右图：相位 (Phase) ---
    # 使用 'twilight' 色图，因为 -pi 和 pi 是连续的
    im2 = axes[1].imshow(phase_masked, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Probe Phase (Masked)")
    cbar = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    # 设置背景色以便看清被 Mask 掉的区域
    axes[1].set_facecolor('#333333')

    plt.tight_layout()
    plt.show()
