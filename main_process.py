import ase
import matplotlib.pyplot as plt
import numpy as np
import abtem
from scipy.special import genlaguerre
from ase.build import mx2
import zarr
import json
from datetime import datetime
import os
from Probes import create_probe  # 假设你在 Probes.py 中定义了一个根据字符串返回 probe 的函数
from Sample_materials import get_atoms
from abtem.reconstruct import RegularizedPtychographicOperator
from matplotlib.patches import Circle
import dask.array as da
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from abtem.core.energy import energy2wavelength
# import cupy as cp

def main_process(materials_list, probes_list, grid_scan_params, uniform_params, compare=False):
    """
    主处理流程：根据样品和探针，进行扫描并计算衍射图样。

    参数：
        materials_list (list): 样品的名称列表
        probes_list (list): 探针的名称列表
        grid_scan_params (dict): 网格扫描参数
        uniform_params (dict): 统一的参数

    返回：
        measurements_list (list): 所有计算的衍射测量结果
        metadata_list (list): 每个实验对应的参数元数据
    """

    measurements_list = []
    measurements_compare_list = []
    metadata_list = []
    # 统一参数
    Sampling = uniform_params['sampling']
    Energy = uniform_params['energy']
    Gpts = uniform_params['gpts']
    semiangle_cutoff = uniform_params['semiangle_cutoff']
    for material_name in materials_list:
        print(f"\n🔬 Processing material: {material_name}")
        atoms = get_atoms(material_name, **uniform_params)
        abtem.show_atoms(atoms)

        # 生成电势（Potential）
        potential = abtem.Potential(
            atoms = atoms,
            sampling = Sampling,
            box = (Sampling * Gpts , Sampling * Gpts, 12),
            periodic= False
            ) # 和probe的grid保持一致
        # visualization = (potential.build() * 0.1).show()
        for probe_name in probes_list:
            print(f"  🌀 Using probe: {probe_name}")
            probe = create_probe(probe_name, **uniform_params)
            # if probe.__class__.__name__ == 'Probe':
            #     probe.grid.match(potential)
            # test_params = {
            #     'sampling': 0.02,
            #     'gpts':512,
            #     'energy': 80e3,
            #     'semiangle_cutoff': 30,
            #     'C10': 50.,              # Defocus in Ångström
            #     'Cs': -50e-6 * 1e10      # Spherical aberration in Ångström (-50 µm)
            # }
            # probe  = create_probe(probe_name, **test_params)
            # 定义扫描参数
            grid_scan = abtem.GridScan(
                start=grid_scan_params['start'],
                end=grid_scan_params['end'],
                gpts=grid_scan_params['gpts']
            )
            fig, ax = abtem.show_atoms(atoms)
            grid_scan.add_to_plot(ax)
            # 定义像素化探测器
            detector = abtem.PixelatedDetector(
                max_angle=5*20,  # 仅示例，真实情况可根据实验设置调整
                resample=False
            )
            if compare:
                detector2 = abtem.FlexibleAnnularDetector()

            # 进行扫描模拟
            if probe.__class__.__name__ == 'Probe':
                measurements = probe.scan(potential=potential, scan=grid_scan , detectors=detector)
            else:
                # probe.is_lazy = True  # 需要分块计算，启用懒加载
                measurements = probe.scan(potential=potential, scan=grid_scan , detectors=detector, max_batch = 50)
            measurements_compare = None
            if compare:
                measurements_compare = probe.scan(potential=potential, scan=grid_scan , detectors=detector2)
            # 不强制 compute()，保留懒加载以节约内存
            # measurements.compute()

            measurements_list.append(measurements)
            measurements_compare_list.append(measurements_compare)
            # 构造 metadata
            metadata = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material": material_name,
                "probe": probe_name,
                "grid_scan_params": grid_scan_params,
                "potential_sampling": Sampling,
                "probe_energy": Energy,
                "probe_type": probe.__class__.__name__,
                "gpts": Gpts,
                "semiangle_cutoff": semiangle_cutoff
            }

            metadata_list.append(metadata)

            print(f"✅ Measurement for {material_name}-{probe_name} done.\n")

    return measurements_list, metadata_list, measurements_compare_list


def save_measurements_to_zarr(measurements, measurements_compare, materials_list, probes_list, 
                              grid_scan_params, uniform_params, compare=False,
                              metadata_list=None,
                              save_dir="results_zarr"):
    """
    保存 abTEM 计算的 measurement 对象到 zarr 文件，
    并同时写入实验参数 metadata。

    参数：
        measurements (list): abTEM 生成的测量对象列表
        materials_list (list): 材料名称列表
        probes_list (list): 探针名称列表
        grid_scan_params (dict): 扫描参数
        metadata_list (list, 可选): 每个测量的详细元数据列表
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    all_metadata = []  # 汇总所有 metadata

    for i, measurement in enumerate(measurements):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        zarr_path = os.path.join(save_dir, f"measurement_{i+1}.zarr")
        # measurement.build().compute()  # 确保数据已计算
        print(f"[Saving] Writing measurement {i+1} to {zarr_path}")
        measurement.to_zarr(zarr_path, overwrite=True)
        if compare:
            zarr_path_compare = os.path.join(save_dir, f"measurement_compare_{i+1}.zarr")
            print(f"[Saving] Writing measurement_compare {i+1} to {zarr_path_compare}")
            measurements_compare[i].to_zarr(zarr_path_compare, overwrite=True)
        # 若单独未提供 metadata_list，则使用循环自动推断
        if metadata_list is not None and i < len(metadata_list):
            metadata = metadata_list[i]
        else:
            metadata = {
                "timestamp": timestamp,
                "material": materials_list[i % len(materials_list)],
                "probe": probes_list[i % len(probes_list)],
                "grid_scan_params": grid_scan_params,
                "uniform_params": uniform_params,
            }

        # 写入单个 JSON 文件
        json_path = os.path.join(save_dir, f"metadata_{i+1}.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        all_metadata.append(metadata)
        print(f"✅ Measurement {i+1} saved with metadata.\n")

    # 保存整个批次的 metadata 汇总
    summary_path = os.path.join(save_dir, "all_metadata_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=4)
    print(f"📦 All metadata saved to {summary_path}\n")

def load_measurement_from_zarr(zarr_path):
    """
    从 zarr 文件读取 measurement 对象。

    参数：
        zarr_path (str): .zarr 文件路径
    返回：
        measurement (abtem Array 或 Measurement 对象)
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"❌ File not found: {zarr_path}")
    
    print(f"[Loading] Reading measurement from {zarr_path} ...")

    try:
        measurement = abtem.from_zarr(zarr_path)
        print(f"✅ Successfully loaded measurement object.")
        return measurement
    except Exception as e:
        print(f"⚠️ Failed to load measurement: {e}")
        return None
    
       
def load_measurement_with_metadata(base_dir, index=1):
    """
    读取 measurement_zarr 和对应的 metadata json。

    参数：
        base_dir (str): 存放 zarr 文件和 json 文件的文件夹
        index (int): 要加载的测量编号（从 1 开始）
    返回：
        measurement, metadata
    """
    zarr_path = os.path.join(base_dir, f"measurement_{index}.zarr")
    zarr_compare_path = os.path.join(base_dir, f"measurement_compare_{index}.zarr")
    json_path = os.path.join(base_dir, f"metadata_{index}.json")

    measurement = load_measurement_from_zarr(zarr_path)
    measurement_compare = None
    if os.path.exists(zarr_compare_path):
        measurement_compare = load_measurement_from_zarr(zarr_compare_path)
    metadata = None
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        print(f"📖 Loaded metadata from {json_path}")
    else:
        print(f"⚠️ Metadata file not found: {json_path}")

    return measurement, metadata, measurement_compare

def process_for_plot(data):
    # 1. 取绝对值 (如果是复数)
    if np.iscomplexobj(data):
        data = np.abs(data)
    
    # 2. 加上一个小量防止 log(0)
    data_log = np.log10(data + 1e-1) 
    
    # 3. 归一化到 0-1 之间 (Min-Max Scaling)
    # 这样 pytcho(0~100) 和 adf(0~10000) 就会变成一样的 0~1 范围
    d_min = np.min(data_log)
    d_max = np.max(data_log)
    data_norm = (data_log - d_min) / (d_max - d_min)
    
    return data_norm

def process_phase_for_plot(phase_img, intensity_img, threshold=0.1):
    """
    1. 去除由于 rPIE 迭代导致的整体相位漂移 (Piston term)
    2. 对背景噪声进行 Mask 处理
    """
    # 归一化强度，用于判断哪里是“背景”
    norm_int = (intensity_img - intensity_img.min()) / (intensity_img.max() - intensity_img.min())
    
    # 创建 Mask：强度低于阈值 (比如最大值的 10%) 的地方视为背景
    mask = norm_int < threshold
    
    # 1. 对齐相位 (Remove Piston): 
    # 计算有效区域(非背景)的平均相位，并将整体减去这个均值
    # 这样可以保证每次迭代的相位颜色是可比的
    valid_phase = phase_img[~mask]
    if len(valid_phase) > 0:
        mean_phase = np.mean(valid_phase)
        # 将相位中心对齐到 0
        aligned_phase = phase_img - mean_phase
        # 重新 Wrap 到 [-pi, pi]
        aligned_phase = (aligned_phase + np.pi) % (2 * np.pi) - np.pi
    else:
        aligned_phase = phase_img

    # 2. 应用 Mask (将背景设为 NaN，matplotlib 会自动留白)
    masked_phase = aligned_phase.copy()
    masked_phase[mask] = np.nan 
    
    return masked_phase




if __name__ == "__main__":

    # region ================== 实验初始化  ========================
    # 定义待测材料和探针列表
    materials_list = ['bp']  # 这里使用你实现的材料名称
    probes_list = ['Gaussian','lg']  # 这里使用你实现的探针名称


    # 使用三级区域：样品势场>扫描区域>感兴趣区域
            # 测试：20>2-18>5-15
            # 实验：40>6-34>10-30

    Area_interest = (10,10)
    Area_offset = (0,0)

    # Area_interest = (20,20)
    # Area_offset = (10,10)

    # 定义测试扫描参数
    # grid_scan_params = {
    #     'start': (2, 2),
    #     'end': (18, 18),
    #     'gpts': (64, 64)
    # }

    grid_scan_params = {
        'start': (0, 0),
        'end': (15, 15),
        'gpts': (30, 30)
    }

    # ##实验扫描参数
    # grid_scan_params = {
    #     'start': (6, 6),
    #     'end': (34, 34),
    #     'gpts': (42, 42)
    # }

    # 定义统一的参数,sampling*gpts决定了探针和样品势场的计算区域大小


    # #测试用
    # Uniform_params = {
    #     'sampling': 0.02,  # 每个像素的大小，单位Å
    #     'energy': 80e3,   # 统一的probe加速电压，单位eV
    #     'gpts': 1024,      # 格点数
    #     'semiangle_cutoff': 30, # 探针半角截止，单位mrad
    # }


    Uniform_params = {
        'sampling': 0.04,  # 每个像素的大小，单位Å
        'energy': 80e3,   # 统一的probe加速电压，单位eV
        'gpts': 384,      # 格点数
        'semiangle_cutoff': 30, # 探针半角截止，单位mrad
    }

    # #实验用
    # Uniform_params = {
    #     'sampling': 0.02,  # 每个像素的大小，单位Å
    #     'energy': 80e3,   # 统一的probe加速电压，单位eV
    #     'gpts': 2048,      # 格点数
    #     'semiangle_cutoff': 30, # 探针半角截止，单位mrad
    # }
    compare = False
    # endregion



    # region ================== 调用主流程获取4D-STEM数据  ========================

    # measurements, metadatas, measurements_compare = main_process(materials_list, probes_list, grid_scan_params, Uniform_params,compare)


    # # 存储measurements为zarr文件,同时存储所有实验参数包括扫描设置探针种类，材料种类，采样率，
    # # 保存所有结果
    # save_measurements_to_zarr(
    #     measurements,
    #     measurements_compare,
    #     materials_list,
    #     probes_list,
    #     grid_scan_params,
    #     metadata_list=metadatas,
    #     uniform_params=Uniform_params,
    #     compare=compare,
    #     save_dir="results_zarr12"
    # ) 

    # endregion

    

    # region ================== 重建过程  ========================
    # 获取存储的4D-STEM数据和对应探针
    base_dir = "results_zarr12"
    index = 2  
    measurement, metadata, measurement_compare = load_measurement_with_metadata(base_dir, index=index)

    # test_params = {
    #     'sampling': 0.02,
    #     'gpts':512,
    #     'energy': 80e3,
    #     'semiangle_cutoff': 30,
    #     'C10': 50.,              # Defocus in Ångström
    #     'Cs': -50e-6 * 1e10      # Spherical aberration in Ångström (-50 µm)
    # }
    # probe  = create_probe(metadata['probe'], **test_params)

    # 如果保存有 compare 数据，则进行对比分析
    if compare:    # 设置三个角度积分探测器
        flexible_measurement = measurement_compare.poisson_noise(1e5, seed=100)
        flexible_measurement.compute()
        bf_measurement = flexible_measurement.integrate_radial(0, probe.semiangle_cutoff)
        maadf_measurement = flexible_measurement.integrate_radial(50, 150)
        haadf_measurement = flexible_measurement.integrate_radial(90, 200)
        measurements = abtem.stack(
            [bf_measurement, maadf_measurement, haadf_measurement], ("BF", "MAADF", "HAADF")
        )
        measurements.show(
            explode=True,
            figsize=(14, 5),
            cbar=True,
        )
        interpolated_measurements = measurements.interpolate(0.05)

        filtered_measurements = interpolated_measurements.gaussian_filter(0.3)

        filtered_measurements.show(
            explode=True,
            figsize=(14, 5),
            cbar=True,
        )

        noisy_measurements = filtered_measurements.poisson_noise(dose_per_area=1e5)

        noisy_measurements.show(
            explode=True,
            figsize=(14, 5),
            cbar=True,
        )

    # 添加噪声和裁剪范围   
    noisy_ptycho = measurement.poisson_noise(1e5)
    cropped_measurements = noisy_ptycho.crop(max_angle=100)

    # 积分模仿ADF图像
    Adf = noisy_ptycho.integrate_radial(50, 100)
    adf_diff = Adf.diffractograms()
    # endregion
   
   
   
    # region ================== 可视化衍射图样  ========================
    # # check衍射图样是否硬截断
    # raw_data = measurement.array

    # # --- 步骤 0: 确保数据已计算 ---
    # try:
        
    #     if isinstance(raw_data, da.Array):
    #         raw_data = raw_data.compute()
    # except ImportError:
    #     pass

    # # --- 步骤 A: 提取坐标轴信息 ---
    # # abtem 的 angular_coordinates 通常包含 [y_coords, x_coords] (对应 axis 0 和 axis 1)
    # # 注意：这里我们提取出来，准备赋值给绘图坐标
    # coords_axis_0 = measurement.angular_coordinates[0]  # 原本的纵轴坐标 (ky)
    # coords_axis_1 = measurement.angular_coordinates[1]  # 原本的横轴坐标 (kx)

    # # --- 步骤 B: 提取单帧并处理 ---
    # single_pattern = raw_data[0, 0, :, :] 
    # log_pattern = np.log10(single_pattern + 1e-5)

    # # --- 步骤 C: 处理“横纵反转”请求 ---
    # # 如果你觉得反了，我们需要做两件事：
    # # 1. 转置图像矩阵 (.T)
    # # 2. 交换用于 extent 的坐标轴数据

    # # 这里执行转置 (Swap X and Y)
    # plot_data_linear = single_pattern.T
    # plot_data_log = log_pattern.T

    # # 因为图像转置了，现在的“横轴”对应原来的 axis_0，“纵轴”对应原来的 axis_1
    # x_coords = coords_axis_0
    # y_coords = coords_axis_1

    # # 计算 extent范围 [x_min, x_max, y_min, y_max]
    # # 为了更精确，最好取首尾坐标作为边界
    # extent_val = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    # # --- 步骤 D: 可视化 ---
    # plt.figure(figsize=(12, 5))

    # # 左图：线性显示
    # plt.subplot(1, 2, 1)
    # # origin='lower' 非常重要！因为倒空间坐标通常是从负到正，原点在中心。
    # # 如果不加 lower，y轴坐标可能是倒着的。
    # plt.imshow(plot_data_linear, cmap='inferno', extent=extent_val, origin='lower')
    # plt.title("Linear Scale (Transposed)")
    # plt.xlabel("Angle axis 0 (mrad)") # 之前的行坐标现在变成了X轴
    # plt.ylabel("Angle axis 1 (mrad)") # 之前的列坐标现在变成了Y轴
    # plt.colorbar()

    # # 右图：对数显示
    # plt.subplot(1, 2, 2)
    # plt.imshow(plot_data_log, cmap='inferno', extent=extent_val, origin='lower')
    # plt.title("Log Scale (Transposed)")
    # plt.xlabel("Angle axis 0 (mrad)")
    # plt.ylabel("Angle axis 1 (mrad)")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()
    # endregion#####################################################



    # region ================== 可视化ADF成像  ========================
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,5))
    # cropped_measurements.show(ax=ax1, title="diffraction pattern", units="mrad")

    # Adf.show(
    # ax=ax2, title="ADF intensity"
    # )
    # noisy_ptycho.integrate_radial(50, 100).interpolate(.05).show(
    # ax=ax3, title="Interpolated ADF intensity"
    # )
    # plt.tight_layout()   
    # adf_diff = Adf.diffractograms()
    # endregion#####################################################




    ####设置保存和读取重建结果的路径
    reconstruction_dir = os.path.join(base_dir, f"reconstruction_results_{index}")
    rpie_zarr_path = os.path.join(reconstruction_dir, "rpie_reconstruction.zarr")
    rpie_probes_zarr_path = os.path.join(reconstruction_dir, "rpie_probes.zarr")
    rpie_positions_path = os.path.join(reconstruction_dir, "rpie_positions.json")
    rpie_sse_path = os.path.join(reconstruction_dir, "rpie_sse.json")
    ####设置重建参数

    max_iterations = 20
    #控制该参数来限制探针更新与否
    probe_fix_iterations = 0
    reconstruction_params = {
        "alpha": 1.0,
        "beta": 1.0,
        "object_step_size": 1.0,
        "probe_step_size": 1.0,
        "position_step_size": 0,#不更新位置
        "step_size_damping_rate": 0.995,
        "pre_position_correction_update_steps": None,
        "pre_probe_correction_update_steps": probe_fix_iterations * grid_scan_params['gpts'][0]*grid_scan_params['gpts'][1],  
    }



    # region ================== RPIE重建  ========================
    #重建得到样品,探针,position,sse
    ###############################################################
    cropped_measurements.compute()
    ## 这里要创建一个和重建像素匹配的探针
    Recon_gpts = cropped_measurements.shape[-2:]
    Angular_sampling = cropped_measurements.angular_sampling
    Recon_sampling =tuple(
            energy2wavelength(metadata['probe_energy']) * 1e3 / dk / n
            for dk, n in zip(Angular_sampling, Recon_gpts)
        )
    Recon_probe_params = {
        'sampling': Recon_sampling,  # 每个像素的大小，单位Å
        'energy': metadata['probe_energy'],   # 统一的probe加速电压，单位eV
        'gpts': Recon_gpts,      #
        'semiangle_cutoff': metadata['semiangle_cutoff'], # 探针半角截止，单位mrad
    }
    probe = create_probe(metadata['probe'], **Recon_probe_params)
    if probe.__class__.__name__ != 'Probe':
        probe = probe.array
    # 计算扫描的样品区域大小
    sample_size = abs(metadata['grid_scan_params']['end'][0]- metadata['grid_scan_params']['start'][0]) # Å
    step_size = sample_size / (metadata['grid_scan_params']['gpts'][0]-1)  # Å
    ptycho_operator = RegularizedPtychographicOperator(
        cropped_measurements,
        energy=metadata['probe_energy'],
        semiangle_cutoff=metadata['semiangle_cutoff'],
        scan_step_sizes = step_size,
        parameters={"object_px_padding": (0,0)},
        probes = probe,
    )

    ptycho_operator.preprocess()
    rpie_objects, rpie_probes, rpie_positions, rpie_sse = ptycho_operator.reconstruct(
        max_iterations=max_iterations, return_iterations=True, random_seed=1, verbose=True,
        **reconstruction_params
    )
    # endregion #####################################################


    # region ================= 保存重建结果  ========================
    ##添加重建结果保存部分
    # 保存重建结果到 zarr 文件

    os.makedirs(reconstruction_dir, exist_ok=True)

    print(f"[Saving] Writing RPIE reconstruction to {rpie_zarr_path}")
    rpie_objects.to_zarr(rpie_zarr_path, overwrite=True)
    
    print(f"[Saving] Writing RPIE probes to {rpie_probes_zarr_path}")
    rpie_probes.to_zarr(rpie_probes_zarr_path, overwrite=True)
    #positions和sse保存为json
    rpie_positions_serializable = [x.tolist() for x in rpie_positions]

    with open(rpie_positions_path, "w") as f:
        json.dump(rpie_positions_serializable, f, indent=4)   
    print(f"[Saving] Writing RPIE positions to {rpie_positions_path}")
    
    with open(rpie_sse_path, "w") as f:
        json.dump(rpie_sse, f, indent=4)
    print(f"[Saving] Writing RPIE SSE to {rpie_sse_path}")
    
    
    #endregion #####################################################


    # region ================== 读取重建结果 ========================
    # 从 zarr 文件读取重建结果
    print(f"[Loading] Reading RPIE reconstruction from {rpie_zarr_path}")
    rpie_objects = abtem.from_zarr(rpie_zarr_path)
    print(f"[Loading] Reading RPIE probes from {rpie_probes_zarr_path}")
    rpie_probes = abtem.from_zarr(rpie_probes_zarr_path)
    with open(rpie_positions_path, "r") as f:
        rpie_positions = np.array(json.load(f))
    print(f"[Loading] Reading RPIE positions from {rpie_positions_path}")
    with open(rpie_sse_path, "r") as f:
        rpie_sse = json.load(f)
    print(f"[Loading] Reading RPIE SSE from {rpie_sse_path}")
    # endregion #####################################################


    #region ================== 可视化重建结果并保存 ========================





    # %%############################################# 绘制RPIE的样品重建相位
    rpie_objects.phase().show(
        explode=True, figsize=(14, 5), cbar=True, common_color_scale=True,
        # vmin=-0.4, vmax=0.5
    )
    #绘制最后一张样品相位
    rpie_objects.phase()[-1].show(
        figsize=(7, 5), cbar=True, vmin=-0.4, vmax=0.5
    )

    c_rpie_objects = rpie_objects.crop(extent=Area_interest, offset=Area_offset)
    c_rpie_objects.phase().show(
        explode=True, figsize=(14, 5), cbar=True, common_color_scale=True,
        # vmin=-0.4, vmax=0.5
    )
    #绘制最后一张样品相位
    c_rpie_objects.phase()[-1].show(
        figsize=(7, 5), cbar=True, vmin=-0.4, vmax=0.5
    )
    save_path = os.path.join(reconstruction_dir, 'recon_phase.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')


    # %%#######################################################绘制RPIE的SSE曲线
    plt.figure(figsize=(8, 4)) # 创建一个新的图形窗口
    plt.plot(rpie_sse, marker='o', linestyle='-', color='b', markersize=3, label='SSE per Iteration')
    plt.title('RPIE Sum of Squared Errors (SSE) Curve', fontsize=14)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线
    plt.legend()
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
    plt.show() # 显示图形
    save_path = os.path.join(reconstruction_dir, 'recon_sse.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
   # %%#######################################################绘制探针强度分布
    raw_stack = rpie_probes.array
    if hasattr(raw_stack, 'compute'):
        raw_stack = raw_stack.compute()
    
    # 2. 转换为强度 (Intensity = |Psi|^2)
    intensity_stack = np.abs(raw_stack)**2

    extent_max = rpie_probes.extent[0]
    extent = [0, extent_max, 0, extent_max]
    total_iter = intensity_stack.shape[0]
    indices = np.linspace(0, total_iter - 1, 5, dtype=int)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), constrained_layout=True)
    for i, ax in zip(indices, axes):
        # --- 关键：独立归一化 (Individual Normalization) ---
        img_data = intensity_stack[i]
        # 归一化到 0-1，消除 rPIE 数值漂移的影响
        norm_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # 绘图
        im = ax.imshow(norm_data, cmap='inferno', origin='lower',
                    extent=extent) # 使用实空间坐标
        
        ax.set_title(f'Iter {i+1}')
        ax.set_xlabel('x (Å)')
        if i == 0:
            ax.set_ylabel('y (Å)')
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Norm. Intensity')
    plt.suptitle(f"Probe Reconstruction (Real Space): 0 - {total_iter} iterations", fontsize=14)
    plt.show()
    save_path = os.path.join(reconstruction_dir, 'recon_probe_intensity.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')





    # 绘制最终强度结果 (Final Result) 
    plt.figure(figsize=(6, 5))
    # 获取最后一张
    final_probe = intensity_stack[-1]
    # 归一化
    final_norm = (final_probe - final_probe.min()) / (final_probe.max() - final_probe.min())

    plt.imshow(final_norm, cmap='inferno', origin='lower', extent=extent)
    plt.title(f"Final Reconstructed Probe (Iter {total_iter})")
    plt.xlabel('x position (Å)')
    plt.ylabel('y position (Å)')
    plt.colorbar(label='Normalized Intensity')
    plt.tight_layout()
    plt.show()
    save_path = os.path.join(reconstruction_dir, 'recon_probe_final_intensity.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')


    #%% #######################################################绘制探针相位分布
    raw_stack = rpie_probes.array
    if hasattr(raw_stack, 'compute'):
        raw_stack = raw_stack.compute()
    # 2. 计算强度和相位
    # 强度用于生成掩膜 (Mask)
    intensity_stack = np.abs(raw_stack)**2
    # 相位范围是 [-pi, pi]
    phase_stack = np.angle(raw_stack)
    # 3. 计算实空间坐标范围 (同上一步，防止上下文丢失再次计算)
    try:
        dy, dx = rpie_probes.sampling
        ny, nx = rpie_probes.shape[-2:]
    except:
        sampling_val = 0.05 
        dy = dx = sampling_val
        ny, nx = intensity_stack.shape[-2:]

    Lx = nx * dx
    Ly = ny * dy
    extent_real = [0, Lx, 0, Ly]
    # 绘制相位演变
    total_iter = phase_stack.shape[0]
    indices = np.linspace(0, total_iter - 1, 5, dtype=int)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), constrained_layout=True)
    for i, ax in zip(indices, axes):
        # 提取单帧
        p_img = phase_stack[i]
        i_img = intensity_stack[i]
        # 处理相位 (去噪、对齐、Mask)
        plot_data = process_phase_for_plot(p_img, i_img, threshold=0.00001)
        # 绘图
        # cmap='twilight': 这种色图首尾相接，非常适合显示相位 (-pi 和 pi 颜色一样)
        im = ax.imshow(plot_data, cmap='twilight', origin='lower',
                    extent=extent_real, vmin=-np.pi, vmax=np.pi)
        ax.set_title(f'Iter {i+1}')
        ax.set_xlabel('x (Å)')
        if i == 0:
            ax.set_ylabel('y (Å)') 
        # 设置背景颜色为深灰色，以便区分 NaN 区域
        ax.set_facecolor('#333333') 
    # 添加 Colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Phase (rad)')
    # 设置 colorbar 刻度为 pi 格式
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    plt.suptitle(f"Probe Phase Evolution (Masked by Intensity)", fontsize=14)
    plt.show()
    save_path = os.path.join(reconstruction_dir, 'recon_probe_phase_evolution.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight') 



    # 绘制最终相位 (Final Result)
    plt.figure(figsize=(6, 5))
    final_phase = phase_stack[-1]
    final_int = intensity_stack[-1]
    final_plot_data = process_phase_for_plot(final_phase, final_int, threshold=0.01)
    # 绘图
    plt.imshow(final_plot_data, cmap='twilight', origin='lower', 
            extent=extent_real, vmin=-np.pi, vmax=np.pi)
    plt.gca().set_facecolor('#333333') # 背景色
    plt.title(f"Final Probe Phase (Iter {total_iter})")
    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    cbar = plt.colorbar(label='Phase (radians)')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.tight_layout()
    plt.show()
    save_path = os.path.join(reconstruction_dir, 'recon_probe_final_phase.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')



    #%%########################################################## 绘制衍射图
    pytcho = c_rpie_objects.phase()[-1]
    # 向pytcho添加metadata中的energy信息
    pytcho.metadata['energy'] = metadata['probe_energy']
    pytcho_diff = pytcho.diffractograms()
    alpha = metadata['semiangle_cutoff']  # mrad
    # 创建图像和坐标轴
    display_ratio = 3.0 
    limit_val = alpha * display_ratio
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'hspace': 0.16, 'wspace': 0.3})
    plt.rcParams.update({'font.size': 13})

    # 左图：Ptychography
    data_pytcho = pytcho_diff.array.compute() # dask array to numpy array
    plot_data_pytcho = process_for_plot(data_pytcho)
    extent_min = pytcho_diff.angular_limits[0][0]
    extent_max = pytcho_diff.angular_limits[0][1]
    extent = [extent_min, extent_max, extent_min, extent_max]
    im1 = axes[0].imshow(plot_data_pytcho, cmap='inferno', # 推荐 inferno 或 magma 看衍射更清晰
                        extent=extent, origin='lower',
                        vmin=0, vmax=1) # 因为归一化了，所以固定 0-1
    axes[0].set_title('(a) Ptychography Diff (Log & Norm)')
    axes[0].add_patch(Circle((0, 0), alpha, edgecolor='white', linestyle='--', facecolor='none'))
    axes[0].text(alpha*1.1, alpha*1.1, r'$\alpha$', fontsize=15, color='white')


    # 右图：ADF Diffraction
    data_adf = adf_diff.array.compute()
    plot_data_adf = process_for_plot(data_adf)
    extent_min = adf_diff.angular_limits[0][0]
    extent_max = adf_diff.angular_limits[0][1]
    extent = [extent_min, extent_max, extent_min, extent_max]
    im2 = axes[1].imshow(plot_data_adf, cmap='inferno',
                        extent=extent, origin='lower',
                        vmin=0, vmax=1) # 同样固定 0-1
    axes[1].set_title('(b) ADF Diff (Log & Norm)')
    axes[1].add_patch(Circle((0, 0), alpha, edgecolor='white', linestyle='--', facecolor='none'))
    axes[1].text(alpha*1.1, alpha*1.1, r'$\alpha$', fontsize=15, color='white')

    # 添加 Colorbar (共用或分别添加)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Relative Log Intensity')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Relative Log Intensity')

    plt.show()
    # 设置对数色标
    # im2.set_norm(LogNorm(vmin=15, vmax=22))  # 对数色标
    save_path = os.path.join(reconstruction_dir, 'diff_images.pdf')
    # 保存图像到 PDF 文件
    plt.savefig(save_path, format='pdf', bbox_inches='tight')



    # endregion#####################################################

    # 显示绘制的图形
    a = 1
    # 重建效果分析




