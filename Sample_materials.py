import ase
import matplotlib.pyplot as plt
import numpy as np
import abtem
from ase.build import mx2
from ase.build import bulk

from ase.lattice.hexagonal import Graphene


def get_atoms(material_name: str, **uniform_params):
    """
    根据指定的名称创建并返回一个原子结构。

    Args:
        material_name (str): 预定义的材料名称。
                             例如: 'MoS2_S_vacancy'。
        **kwargs: 额外的参数，可以用于修改结构，例如 supercell_size。

    Returns:
        ase.Atoms: 构建好的原子结构对象 (abtem 使用 ASE 的 Atoms 对象)。
    
    Raises:
        ValueError: 如果请求的 material_name 未定义。
    """
    boxsize = uniform_params.get('sampling', 0.02) * uniform_params.get('gpts', 512)

    if material_name == 'MoS2':
        

        # 1. 创建 MoS₂ 初始结构
        structure = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.127,
                        size=(1, 1, 1), vacuum=10)
        structure.pbc = True

        # 2. 晶胞处理与超胞构建
        atoms = abtem.orthogonalize_cell(structure)
        atoms.rotate(-90, 'z', rotate_cell=True)
        atoms *= (16,16,1) # 使用变量 size 创建超胞

        # 3. 手动设置超胞(box)的坐标与居中
        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()

            
        return atoms


    elif material_name == 'WSe2':
       # 1. 创建 WSe2初始结构
        structure = mx2(formula='WSe2', kind='2H', a=3.32, thickness=3.324,
                        size=(1, 1, 1), vacuum=10)
        structure.pbc = True

        # 2. 晶胞处理与超胞构建
        atoms = abtem.orthogonalize_cell(structure)
        atoms.rotate(-90, 'z', rotate_cell=True)
        atoms *= (16, 16, 1) # 使用变量 size 创建超胞 


        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()
        return atoms


    elif material_name == 'WS2':
       # 1. 创建 WSe2初始结构
        structure = mx2(formula='WS2', kind='2H', a=3.18, thickness=3.119,
                        size=(1, 1, 1), vacuum=10)
        structure.pbc = True

        # 2. 晶胞处理与超胞构建
        atoms = abtem.orthogonalize_cell(structure)
        atoms.rotate(-90, 'z', rotate_cell=True)
        atoms *= (16, 16, 1) # 使用变量 size 创建超胞

        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()
        return atoms
    

    
    elif material_name == 'graphene':
        lc = 2.47
        vac = 12.0
        structure = Graphene(symbol='C', latticeconstant={'a':lc, 'c':vac},
                            size=(1,1,1))
        structure.pbc = True
         # Orthogonalize cell
        atoms = abtem.orthogonalize_cell(structure)
        atoms.rotate(90, 'z', rotate_cell=True)
        # atoms.cell=[4.3284, 2.499, 12]
        atoms *= (16, 16, 1) # 使用变量 size 创建超胞
        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()
        return atoms


    elif material_name == 'h-BN':
        lc = 2.5
        vac = 12.0

        structure = Graphene(symbol='C', latticeconstant={'a':lc, 'c':vac},
                            size=(1,1,1))
        structure.pbc = True
        structure[0].symbol='B'
        structure[1].symbol='N'

        # Orthogonalize cell
        atoms = abtem.orthogonalize_cell(structure)
        atoms.rotate(90, 'z', rotate_cell=True)
        atoms *= (16,16,1)
        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()
        return atoms
        # abtem.show_atoms(atoms)

    elif material_name == 'FeSe':

        # 1. 晶格常数
        a = 3.933
        c = 12.0 

    # 2. 定义晶胞
        cell = [
            [a, 0, 0],
            [0, a, 0],
            [0, 0, c]
        ]

        # 3. 输入真实坐标 (单位 Å)
        cartesian_positions = np.array([
            [5.9, 0.0, 7.366],
            [3.933, 1.967, 4.348],
            [3.933, 0, 5.857],
            [5.9, 1.967, 5.857],
        ])
            # Se	5.9	0	7.366
            # Se	3.933	1.967	4.348
            # Fe	3.933	0	5.857
            # Fe 	5.9	1.967	5.857


        symbols = ['Se', 'Se', 'Fe', 'Fe']

        # 4. 计算分数坐标 (scaled_positions)
        scaled_positions = np.linalg.solve(np.array(cell).T, cartesian_positions.T).T

        # 5. 创建 ASE 原子对象
        atoms = ase.Atoms(symbols=symbols,
                    scaled_positions=scaled_positions,
                    cell=cell
                    )
        atoms.pbc = True
        atoms *= (16, 16, 1) # 使用变量 size 创建超胞
        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()

        return atoms

    elif material_name == 'bp':
        a = 3.295   # x 方向
        b = 4.54    # y 方向
        c = 12.0    # 留出真空层（z方向）

        cell = [
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c]
        ]

        # === 2. 输入真实空间坐标 (单位 Å) ===
        cartesian_positions = np.array([
            [0.000, -0.404, 6.513],
            [0.000, 0.404, 4.417],
            [-1.648, 1.868, 4.417],
            [-1.648, 2.675, 6.513],
        ])

        symbols = ['P', 'P', 'P', 'P']

        # === 3. 计算分数坐标 ===
        scaled_positions = np.linalg.solve(np.array(cell).T, cartesian_positions.T).T

        # === 4. 构建 Atoms 对象 ===
        atoms = ase.Atoms(
            symbols=symbols,
            scaled_positions=scaled_positions,
            cell=cell 
        )
        atoms.pbc = True
        atoms *= (16, 16, 1)  # 使用变量 size 创建超胞
        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()
        return atoms


    elif material_name == 'SeS2':
        symbols = ['Re']*4 + ['S']*8
        cartesian_positions = np.array([
            [-4.455, -3.527, 9.676],
            [-8.044, -3.383, 9.641],
            [-6.425, -5.667, 9.580],
            [-9.237, -5.811, 9.616],
            [-6.282, -4.006, 11.185],
            [-6.198, -2.295,  8.444],
            [-4.598, -5.187,  8.070],
            [-7.891, -4.941,  7.891],
            [-3.137, -2.132,  8.278],
            [-9.389, -4.253, 11.365],
            [-7.744, -7.062, 10.978],
            [-11.082, -6.899, 10.813],
        ])

        a_vec = np.array([-6.400,  0.000, 0.000])
        b_vec = np.array([-3.143, -5.694, 0.000])  #查阅资料得到a,b矢量
        c_vec = np.array([0.0, 0.0, 12.0])
        cell = np.vstack([a_vec, b_vec, c_vec])
        scaled_positions = np.linalg.solve(cell.T, cartesian_positions.T).T
        atoms = ase.Atoms(symbols=symbols,
              scaled_positions=scaled_positions,
              cell=cell,
              pbc=[True, True, False])
        atoms = abtem.orthogonalize_cell(atoms)
        atoms *= (16, 16, 1) # 使用变量 size 创建超胞
        atoms.cell = [boxsize, boxsize, 12]
        atoms.center()
        return atoms



    else:
        raise ValueError(f"定义的材料 '{material_name}' 不存在。")
    


if __name__ == "__main__":
    
    # ----------------------------------------------------
    # 在这里更改您想测试的样品名称
    MATERIAL_TO_TEST = 'MoS2' 
    # MATERIAL_TO_TEST = 'WSe2'
    # MATERIAL_TO_TEST = 'WS2'
    # MATERIAL_TO_TEST = 'graphene'
    # MATERIAL_TO_TEST = 'h-BN'
    # MATERIAL_TO_TEST = 'FeSe'
    # MATERIAL_TO_TEST = 'bp'
    # MATERIAL_TO_TEST = 'SeS2'

    # ----------------------------------------------------
    atoms = get_atoms(MATERIAL_TO_TEST)
    print(f"--- [调试模式] 正在构建: {MATERIAL_TO_TEST} ---")
    abtem.show_atoms(atoms, plane='xy', legend=True, title=f"{MATERIAL_TO_TEST} (Top View)")
    plt.show()

    abtem.show_atoms(atoms, plane='xz', legend=True, title=f"{MATERIAL_TO_TEST} (Side View1)")
    plt.show()

    abtem.show_atoms(atoms, plane='yz', legend=True, title=f"{MATERIAL_TO_TEST} (Side View2)")
    plt.show()
        