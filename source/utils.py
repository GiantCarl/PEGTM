import torch
import torch.nn as nn
import numpy as np
import gmshparser
from scipy.io import loadmat
import meshio
from shapely.geometry import LineString, Polygon, Point
import random


class DistanceFunction:
    def __init__(self, x_init, y_init, theta, L, d0, order: int = 2):
        self.x_init = x_init
        self.y_init = y_init
        self.theta = theta
        self.L = L
        self.d0 = d0
        self.order = order

    def __call__(self, inp):
        '''
        This function computes distance function given a line with origin at (x_init, y_init),
        oriented at an angle theta from x-axis, and of length L. Value of the function is 1 at
        the line and goes to 0 at a distance of d0 from the line.

        '''

        L = torch.tensor([self.L], device=inp.device)
        d0 = torch.tensor([self.d0], device=inp.device)
        theta = torch.tensor([self.theta], device=inp.device)
        input_c = torch.clone(inp)

        # transform coordinate to shift origin to (x_init, y_init) and rotate axis by theta
        input_c[:, -2:] = input_c[:, -2:] - torch.tensor([self.x_init, self.y_init], device=inp.device)
        Rt = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=inp.device)
        input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
        x = input_c[:, -2]
        y = input_c[:, -1]

        if self.order == 1:
            dist_fn_p1 = nn.ReLU()(x*(L-x))/(abs(x*(L-x))+np.finfo(float).eps)* \
                            nn.ReLU()(d0-abs(y))/(abs(d0-abs(y))+np.finfo(float).eps)* \
                            (1-abs(y)/d0)
            
            dist_fn_p2 = nn.ReLU()(x-L)/(abs(x-L)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-((x-L)**2+y**2))/(abs(d0**2-((x-L)**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt((x-L)**2+y**2)/d0)
            
            dist_fn_p3 = nn.ReLU()(-x)/(abs(x)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-(x**2+y**2))/(abs(d0**2-(x**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt(x**2 + y**2)/d0)
            
            dist_fn = dist_fn_p1 + dist_fn_p2 + dist_fn_p3

            
        if self.order == 2:
            dist_fn_p1 = nn.ReLU()(x*(L-x))/(abs(x*(L-x))+np.finfo(float).eps)* \
                            nn.ReLU()(d0-abs(y))/(abs(d0-abs(y))+np.finfo(float).eps)* \
                            (1-abs(y)/d0)**2
            
            dist_fn_p2 = nn.ReLU()(x-L)/(abs(x-L)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-((x-L)**2+y**2))/(abs(d0**2-((x-L)**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt((x-L)**2+y**2)/d0)**2
            
            dist_fn_p3 = nn.ReLU()(-x)/(abs(x)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-(x**2+y**2))/(abs(d0**2-(x**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt(x**2 + y**2)/d0)**2
            
            dist_fn = dist_fn_p1 + dist_fn_p2 + dist_fn_p3

        return dist_fn
    

    

def hist_alpha_init(inp, matprop, pffmodel, crack_dict,crack_point):
    '''
    This function computes the initial phase field for a sample with a crack.
    See the paper "Phase-field modeling of fracture with physics-informed deep learning" for details.

    '''
    hist_alpha = torch.zeros((inp.shape[0], ), device = inp.device)

    # if len(crack_point) > 0:
    #     l0 = matprop.True_l0
    #     for j, L_crack in enumerate(crack_dict["L_crack"]):
    #         Lc = torch.tensor([L_crack], device=inp.device)
    #         theta = torch.tensor([crack_dict["angle_crack"][j]], device=inp.device)
    #         input_c = torch.clone(inp)

    #         # transform coordinate to shift origin to (x_init, y_init) and rotate axis by theta
    #         input_c[:, -2:] = input_c[:, -2:] - torch.tensor([crack_dict["x_init"][j], crack_dict["y_init"][j]], device=inp.device)
    #         Rt = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=inp.device)
    #         input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
    #         x = input_c[:, -2]
    #         y = input_c[:, -1]

    #         if pffmodel.PFF_model == 'AT1':
    #             hist_alpha_p1 = nn.ReLU()(x*(Lc-x))/(abs(x*(Lc-x))+np.finfo(float).eps)* \
    #                                 nn.ReLU()(2*l0-abs(y))/(abs(2*l0-abs(y))+np.finfo(float).eps)* \
    #                                 (1-abs(y)/l0/2)**2

    #             hist_alpha_p2 = nn.ReLU()(x-Lc+np.finfo(float).eps)/(abs(x-Lc)+np.finfo(float).eps)* \
    #                                 nn.ReLU()(2*l0-torch.sqrt((x-Lc)**2+y**2)+np.finfo(float).eps)/(abs(2*l0-torch.sqrt((x-Lc)**2+y**2))+np.finfo(float).eps)* \
    #                                 (1-torch.sqrt((x-Lc)**2+y**2)/l0/2)**2

    #             hist_alpha_p3 = nn.ReLU()(-x+np.finfo(float).eps)/(abs(x)+np.finfo(float).eps)* \
    #                                 nn.ReLU()(2*l0-torch.sqrt(x**2+y**2)+np.finfo(float).eps)/(abs(2*l0-torch.sqrt(x**2+y**2))+np.finfo(float).eps)* \
    #                                 (1-torch.sqrt(x**2+y**2)/l0/2)**2
                
    #         elif pffmodel.PFF_model == 'AT2':
    #             hist_alpha_p1 = nn.ReLU()(x*(Lc-x))/(abs(x*(Lc-x))+np.finfo(float).eps)* \
    #                                 torch.exp(-abs(y)/l0)

    #             hist_alpha_p2 = nn.ReLU()(x-Lc+np.finfo(float).eps)/(abs(x-Lc)+np.finfo(float).eps)* \
    #                                 torch.exp(-torch.sqrt((x-Lc)**2+y**2)/l0)

    #             hist_alpha_p3 = nn.ReLU()(-x+np.finfo(float).eps)/(abs(x)+np.finfo(float).eps)* \
    #                                 torch.exp(-torch.sqrt(x**2+y**2)/l0)

    #         hist_alpha = hist_alpha + hist_alpha_p1 + hist_alpha_p2 + hist_alpha_p3
    hist_alpha[crack_point] = 1.0

    # import matplotlib.pyplot as plt
    # sc = plt.scatter(inp[:,0].cpu().numpy(),inp[:,1].cpu().numpy(),s = 1,c= hist_alpha.cpu().numpy())
    # plt.gca().set_aspect('equal', adjustable='box') 
    # plt.colorbar(sc) 
    # plt.show()  

    return hist_alpha



def parse_mesh(filename="meshed_geom.msh", gradient_type = 'numerical'):
    '''
    Parses .msh file to obtain nodal coordinates and connectivity assuming triangular elements.
    If numr_dict["gradient_type"] = autodiff, then Gauss points of elements in a one point Gauss 
    quadrature are returned.

    '''
    mesh = loadmat(filename)

    mesh_pos = mesh['mesh_pos']
    X = mesh_pos[:,0]
    Y = mesh_pos[:,1]
    T = mesh['cells']
    edges = mesh['edge']
    assert T.shape[-1] == 3, "Discretization must have only triangle elements"

    area = X[T[:, 0]]*(Y[T[:, 1]]-Y[T[:, 2]]) + X[T[:, 1]]*(Y[T[:, 2]]-Y[T[:, 0]]) + X[T[:, 2]]*(Y[T[:, 0]]-Y[T[:, 1]])
    area = 0.5*area

    if gradient_type == 'autodiff':
        X = (X[T[:, 0]] + X[T[:, 1]] + X[T[:, 2]])/3
        Y = (Y[T[:, 0]] + Y[T[:, 1]] + Y[T[:, 2]])/3

    #loading boundary 
    loading_geo = mesh['loading_geo'][0,0]

    crack_point = mesh['crack_point'].squeeze()

    return X, Y, T, edges, area, loading_geo,crack_point

def parse_mesh_from_gmsh(mesh_info,crack_dict):

    elem_type = mesh_info["elem_type"]
    Pre_Crack_type = mesh_info["Pre_Crack_type"] 

    mesh = meshio.read(mesh_info["mesh_file"])
    points = mesh.points.astype(float)  
    
    elements = {}
    for cell_block in mesh.cells:
        cell_type = cell_block.type
        # print(cell_type)
        cells_node = cell_block.data
        if cell_type in elements:
            elements[cell_type].append(cells_node)
        else:
            elements[cell_type] = []
            elements[cell_type].append(cells_node)
    cells = np.concatenate(elements[elem_type],axis=0)   

    physical_groups_nodes = extract_all_physical_groups_nodes(mesh)

    if Pre_Crack_type == "discrete":
        Crack_point = []
    else:
        crack_element_nodes = []
        
        for x0, y0, L, theta in zip(crack_dict["x_init"], crack_dict["y_init"], crack_dict["L_crack"], crack_dict["angle_crack"]):

            Crack_point1 = [x0,y0]
            x1 = x0 + L * np.cos(theta)
            y1 = y0 + L * np.sin(theta)
            Crack_point2 = [x1,y1]

            Crack = LineString([tuple(Crack_point1), tuple(Crack_point2)])
            for triangle in cells:
                triangle_vertices = [tuple(points[triangle[0]][:2]), tuple(points[triangle[1]][:2]), tuple(points[triangle[2]][:2])]
                triangle_cell = Polygon(triangle_vertices)  
                if Crack.intersects(triangle_cell):
                    crack_element_nodes.append(triangle[0])
                    crack_element_nodes.append(triangle[1])
                    crack_element_nodes.append(triangle[2])        

        crack_element_nodes = np.array(crack_element_nodes,dtype=int)
        Crack_point = np.unique(crack_element_nodes)

    loading_geo = loading_info(mesh_info,physical_groups_nodes,points)

    if elem_type == "triangle" :
        edges = np.concatenate([cells[:,0:2],
                        cells[:,1:3],
                        np.stack([cells[:,2], cells[:,0]],axis=1)], axis=0)  
        reverse_topology = np.concatenate((edges[:,1:2], edges[:,0:1]), axis=1)
        selftoself = np.stack((np.arange(points.shape[0]),np.arange(points.shape[0])),axis=1)
        new_edges = np.concatenate((edges, reverse_topology,selftoself), axis=0)
        new_edges = np.unique(new_edges, axis=0)


    area = points[:,0][cells[:, 0]]*(points[:,1][cells[:, 1]]-points[:,1][cells[:, 2]]) +  points[:,0][cells[:, 1]]*(points[:,1][cells[:, 2]]-points[:,1][cells[:, 0]]) +  points[:,0][cells[:, 2]]*(points[:,1][cells[:, 0]]-points[:,1][cells[:, 1]])
    area = 0.5*area

    return points[:,0:2], cells,new_edges,area,loading_geo,Crack_point 

def get_physical_group_nodes(mesh, group_name):
    """
    获取指定物理组名称的节点编号和坐标。
    
    参数：
        mesh (meshio.Mesh): 读取后的网格对象。
        group_name (str): 物理组的名称。
        
    返回：
        set: 节点编号的集合。
    """
    physical_groups = mesh.field_data
    group_info = physical_groups.get(group_name, None)
    
    if group_info is None:
        raise ValueError(f"未找到名为 '{group_name}' 的物理组")
    
    tag = group_info[0]
    dimension = group_info[1]
    
    node_ids = set()
    
    # 根据维度选择单元类型
    if dimension == 0:
        cell_types = ["vertex"]
    elif dimension == 1:
        cell_types = ["line", "bspline"]
    elif dimension == 2:
        cell_types = ["triangle", "quad", "polygon"]
    elif dimension == 3:
        cell_types = ["tetra", "hexahedron", "wedge", "pyramid"]
    else:
        print(dimension)
        raise ValueError(f"未知的维度: {dimension}")
    
    for cell_type in cell_types:
        if cell_type in mesh.cells_dict and "gmsh:physical" in mesh.cell_data_dict:
            cells = mesh.cells_dict[cell_type]
            physical_tags = mesh.cell_data_dict["gmsh:physical"].get(cell_type, None)
            
            if physical_tags is None:
                continue
            
            for i, tag_value in enumerate(physical_tags):
                if tag_value == tag:
                    node_ids.update(cells[i])
                    
    return node_ids

def extract_all_physical_groups_nodes(mesh):
    """
    提取所有物理组的节点编号和坐标。
    
    参数：
        mesh (meshio.Mesh): 读取后的网格对象。
        
    返回：
        dict: 物理组名称作为键，对应节点编号和坐标。
    """
    physical_groups = mesh.field_data
    result = {}
    
    for name, data in physical_groups.items():
        try:
            node_ids = get_physical_group_nodes(mesh, name)
            node_coords = mesh.points[list(node_ids)]
            result[name] = np.array(list(node_ids),dtype=int).squeeze()               
        except ValueError as e:
            print(e)
    
    return result

def loading_info(mesh_info,physical_groups_nodes,points):

    fix_X_boundary = []
    fix_Y_boundary = []
    disp_X_boundary = []
    disp_Y_boundary = []
    tract_X_boundary = []
    tract_Y_boundary = []

    if mesh_info["fix_X_boundary"] is not None:
        index = mesh_info["fix_X_boundary"].split("+")        
        for i in range (len(index)):
            fix_X_boundary.extend(physical_groups_nodes.get(index[i], None).reshape(-1))
        fix_X_boundary = np.unique(fix_X_boundary)

    if mesh_info["fix_Y_boundary"] is not None:
        index = mesh_info["fix_Y_boundary"].split("+")        
        for i in range (len(index)):
            fix_Y_boundary.extend(physical_groups_nodes.get(index[i], None).reshape(-1))
        fix_Y_boundary = np.unique(fix_Y_boundary)

    if mesh_info["disp_X_boundary"] is not None:
        index = mesh_info["disp_X_boundary"].split("+")        
        for i in range (len(index)):
            disp_X_boundary.extend(physical_groups_nodes.get(index[i], None).reshape(-1))
        disp_X_boundary = np.unique(disp_X_boundary)

    if mesh_info["disp_Y_boundary"] is not None :
        index = mesh_info["disp_Y_boundary"].split("+")        
        for i in range (len(index)):
            disp_Y_boundary.extend(physical_groups_nodes.get(index[i], None).reshape(-1))
        disp_Y_boundary = np.unique(disp_Y_boundary)

    if mesh_info["tract_X_boundary"] is not None :
        index = mesh_info["tract_X_boundary"].split("+")        
        for i in range (len(index)):
            tract_X_boundary.extend(physical_groups_nodes.get(index[i], None).reshape(-1))
        tract_X_boundary = np.unique(tract_X_boundary)

    if mesh_info["tract_Y_boundary"] is not None :
        index = mesh_info["tract_Y_boundary"].split("+")        
        for i in range (len(index)):
            tract_Y_boundary.extend(physical_groups_nodes.get(index[i], None).reshape(-1))
        tract_Y_boundary = np.unique(tract_Y_boundary)

    loading_geo = {
        'fix_X_boundary': np.array(fix_X_boundary),
        'fix_Y_boundary': np.array(fix_Y_boundary),
        'disp_X_boundary': np.array(disp_X_boundary),
        'disp_Y_boundary': np.array(disp_Y_boundary),
        'tract_X_boundary': np.array(tract_X_boundary),
        'tract_Y_boundary': np.array(tract_Y_boundary)
        }
    return loading_geo

def expand_hop_edges(E: torch.LongTensor, num_nodes: int, n_hop: int, contain_self:bool) -> torch.LongTensor:
    """
    在原始有向边集 E 上扩展 n_hop 阶邻域：
    - E: Tensor of shape [M, 2]（dtype=torch.long），每行 [u, v]
    - num_nodes: 图中节点总数 N
    - n_hop: 要扩展的阶数（如 3）
    返回：
    - E_new: Tensor of shape [M + K, 2]，包含原始边与所有中心点到 n_hop 邻域点的新边
    """
    # 1. 构建稀疏邻接矩阵 A (N×N)
    senders, receivers = E[:, 0], E[:, 1]
    values = torch.ones(E.size(0), dtype=torch.float32,device=E.device)
    A = torch.sparse_coo_tensor(
        torch.stack([senders, receivers], dim=0),
        values,
        (num_nodes, num_nodes)
    ).coalesce()   # coalesce 以保证 indices 唯一并排序

    # 2. 迭代计算 A^k（k=1..n_hop），并累积第 n_hop 阶的连接
    A_k = A.clone()
    R = A.clone() 
    for k in range(1, n_hop):
        # 稀疏矩阵乘：相当于从 k-hop 再走一步得到 (k+1)-hop
        A_k = torch.sparse.mm(A_k, A).coalesce()
        mask = A_k.values() > 0
        A_k = torch.sparse_coo_tensor(A_k.indices()[:,mask],
                                     torch.ones(mask.sum(), device=E.device),
                                     (num_nodes, num_nodes))
        R = torch.sparse_coo_tensor(
            torch.cat([R.coalesce().indices(), A_k.coalesce().indices()], dim=1),
            torch.cat([R.coalesce().values(), A_k.coalesce().values()]),
            (num_nodes, num_nodes)
        ).coalesce()
    # 3. 去掉对角线（不算自己到自己的“邻居”），并取并集 (values>0)
    R = R.coalesce()
    idx_all = R.indices()   # [2, K']
    if not contain_self:
        # 过滤掉 idx_all[0]==idx_all[1]
        mask = idx_all[0] != idx_all[1]
        idx_all = idx_all[:, mask]  # [2, K]
    # 4. idx_filt[0] -> idx_filt[1] 就是我们要的新边
    new_E = idx_all.t().contiguous()  # [K,2]
    return new_E 

def set_seed(seed: int = 42):
    # Python 内置的随机模块
    random.seed(seed)
    # NumPy 随机数种子
    np.random.seed(seed)
    # PyTorch CPU 随机数种子
    torch.manual_seed(seed)
    # 如果使用 GPU，则设置所有 GPU 的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 强制 PyTorch 使用确定性算法（可能会损失一点性能）
    torch.backends.cudnn.deterministic = True
    # 关闭 cudnn 的自动找最优算法功能
    torch.backends.cudnn.benchmark = False
