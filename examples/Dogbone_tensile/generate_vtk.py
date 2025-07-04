from config import *
import torch_scatter
import torch
import torch.nn as nn

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE/Path('source')))
import os
import pyvista as pv
from vtk import VTK_TRIANGLE 

from utils import parse_mesh
from compute_energy import gradients,stress,compute_energy_per_elem
from construct_model import construct_model
from input_data_from_mesh import prep_input_graph_data, build_graph
from utils import hist_alpha_init
from field_computation import FieldComputation
npz_file = npz_file = os.listdir(intermediateResults_path)
mesh_file = mesh_info_dict['mesh_file']
vtk_file_path = intermediateResults_vtk_path

device = 'cpu'

pffmodel, matprop, network = construct_model(PFF_model_dict, mat_prop_dict, 
                                             network_dict, domain_extrema, device)

Gc = mat_prop_dict['True_w1']
E = mat_prop_dict['Ture_mat_E']
l0 = mat_prop_dict['True_l0']
Lx = domain_extrema[0,1] - domain_extrema[0,0]
Ly = domain_extrema[1,1] - domain_extrema[1,0]
L = torch.max(Lx,Ly).numpy()    
non_dim = (Gc/E/l0)**(-0.5)
transform_energy = L**2 * Gc / l0  
transform_disp = L / (Gc/E/l0)**(-0.5) 
transform_force = L**2 / (l0/E/Gc)**(0.5)
transform_strain = 1/ (Gc/E/l0)**(-0.5) 
transform_stress = 1/(l0/E/Gc)**(0.5)
k_hop = network_dict["K_hop"]

field_comp = FieldComputation(net = network,
                              domain_extrema = domain_extrema, 
                              lmbda = torch.tensor([0.0], device = device), 
                              lmbda_delta = torch.tensor([0.0], device = device), 
                              theta = loading_angle, 
                              matprop = matprop,
                              alpha_constraint = numr_dict["alpha_constraint"])
assert Path(mesh_file).suffix == '.msh', "Mesh file should be a gmsh file"

graph_data, hist_alpha = prep_input_graph_data(matprop, pffmodel, crack_dict,k_hop, numr_dict, field_comp.domain_extrema, mesh_info=mesh_info_dict, device=device)
cells = np.hstack([np.full((graph_data['T_conn'].shape[0], 1), 3), graph_data['T_conn']]).flatten()

mesh_pos = graph_data['mesh_pos']
# 创建UnstructuredGrid对象，并设置网格的cells和cell_types
grid = pv.UnstructuredGrid(cells, np.full(graph_data['T_conn'].shape[0], VTK_TRIANGLE), np.concatenate([mesh_pos.numpy()*L,np.zeros_like(mesh_pos[:,0:1].numpy())],axis=1))

world_pos = graph_data['mesh_pos']
inp_train = build_graph(graph_data,world_pos,hist_alpha,field_comp)

mesh_edge = torch.cat([graph_data['T_conn'][:,0:2],
                       graph_data['T_conn'][:,1:3],
                       torch.stack([graph_data['T_conn'][:,2],graph_data['T_conn'][:,0]],dim=1)
                       ],dim=0)
mesh_edge_sort,_ = torch.sort(mesh_edge,dim = -1)
mesh_unique = torch.unique(mesh_edge_sort,dim = 0)

disp_x_node = graph_data["disp_X"]
disp_y_node = graph_data["disp_Y"]

energy = np.zeros([1, 3])
force_disp = np.zeros([1, 4])

for i in range(len(npz_file)):
    npz_file_path = intermediateResults_path/Path('intermediate_disp_' +f'{i:04}'  + '.npz')
    savedir = intermediateResults_vtk_path/Path('intermediate_disp_' +f'{i:04}')

    result = np.load(npz_file_path)
    world_pos =  torch.tensor(result['world_pos']).to(torch.float).to(device)
    alpha =  torch.tensor(result['phase_field']).to(torch.float).to(device) 

    grid.point_data["world_pos"] = world_pos.numpy() * L
    grid.point_data["Ux"] =  (world_pos - graph_data['mesh_pos'])[:,0].numpy()* L
    grid.point_data["Uy"] =  (world_pos - graph_data['mesh_pos'])[:,1].numpy() * L
    grid.point_data["phase field"] =  alpha

    u = (world_pos[:,0] -  inp_train.node_features[:,0]) * non_dim
    v = (world_pos[:,1] -  inp_train.node_features[:,1])  * non_dim

    strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y = gradients(inp_train.node_features[:,0:2], u, v, alpha, 
                                                                            inp_train.area_elem, inp_train.T_conn) 

    count = torch_scatter.scatter_add(src=torch.ones_like(strain_11),index= inp_train.T_conn[:, 0],dim=0, dim_size=mesh_pos.shape[0])+ \
            torch_scatter.scatter_add(src=torch.ones_like(strain_11),index= inp_train.T_conn[:, 1],dim=0, dim_size=mesh_pos.shape[0])+ \
            torch_scatter.scatter_add(src=torch.ones_like(strain_11),index= inp_train.T_conn[:, 2],dim=0, dim_size=mesh_pos.shape[0])

    strain_11_node = torch_scatter.scatter_add(src=strain_11,index= inp_train.T_conn[:, 0],dim=0, dim_size=mesh_pos.shape[0])+ \
                     torch_scatter.scatter_add(src=strain_11,index= inp_train.T_conn[:, 1],dim=0, dim_size=mesh_pos.shape[0])+ \
                     torch_scatter.scatter_add(src=strain_11,index= inp_train.T_conn[:, 2],dim=0, dim_size=mesh_pos.shape[0])
    strain_11_node = strain_11_node/count
    grid.point_data["strain_11"] =  (strain_11_node * transform_strain).numpy()

    strain_12_node = torch_scatter.scatter_add(src=strain_12,index= inp_train.T_conn[:, 0],dim=0, dim_size=mesh_pos.shape[0])+ \
                     torch_scatter.scatter_add(src=strain_12,index= inp_train.T_conn[:, 1],dim=0, dim_size=mesh_pos.shape[0])+ \
                     torch_scatter.scatter_add(src=strain_12,index= inp_train.T_conn[:, 2],dim=0, dim_size=mesh_pos.shape[0])
    strain_12_node = strain_12_node/count
    grid.point_data["strain_12"] =  (strain_12_node  * transform_strain).numpy()

    strain_22_node = torch_scatter.scatter_add(src=strain_22,index= inp_train.T_conn[:, 0],dim=0, dim_size=mesh_pos.shape[0])+ \
                     torch_scatter.scatter_add(src=strain_22,index= inp_train.T_conn[:, 1],dim=0, dim_size=mesh_pos.shape[0])+ \
                     torch_scatter.scatter_add(src=strain_22,index= inp_train.T_conn[:, 2],dim=0, dim_size=mesh_pos.shape[0])
    strain_22_node = strain_22_node/count
    grid.point_data["strain_22"] =  (strain_22_node  * transform_strain).numpy()

    #Stress calculation
    stress_11_node, stress_22_node, stress_12_node = stress(strain_11_node, strain_22_node, strain_12_node, alpha, matprop, pffmodel)
    grid.point_data["stress_11"] =  (stress_11_node  * transform_stress).numpy()
    grid.point_data["stress_12"] =  (stress_12_node  * transform_stress).numpy()
    grid.point_data["stress_22"] =  (stress_22_node  * transform_stress).numpy()

    stress_1_node = 0.5*(stress_11_node + stress_22_node) + torch.sqrt((0.5*(stress_11_node - stress_22_node))**2 + stress_12_node**2)
    stress_2_node = 0.5*(stress_11_node + stress_22_node) - torch.sqrt((0.5*(stress_11_node - stress_22_node))**2 + stress_12_node**2)
    grid.point_data["stress_1"] =  (stress_1_node  * transform_stress).numpy()
    grid.point_data["stress_2"] =  (stress_2_node  * transform_stress  ).numpy()

    E_el_elem, E_d_elem = compute_energy_per_elem(inp_train.node_features[:,0:2], u, v, alpha, hist_alpha, matprop, pffmodel,  inp_train.area_elem, T_conn=inp_train.T_conn)
    E_el = torch.sum(E_el_elem)
    E_d = torch.sum(E_d_elem)

    energy = np.append(energy, np.array([[disp[i]*np.cos(field_comp.theta.numpy().squeeze()) * transform_disp, E_el * transform_energy, E_d * transform_energy]]), axis = 0)
    # 将网格写入.vtu文件
    grid.save(f"{savedir}.vtk")   
    print(f'intermediate_disp_' +f'{i:04}'  + '.npz',"\tdisp:","{:.8f}".format(disp[i]),"\tdamage:","{:.8f}".format(alpha.max()),"{:.8f}".format(alpha.min()))

    #load-disp 
    is_in_load_x = torch.isin(mesh_unique,disp_x_node) 
    is_in_load_y = torch.isin(mesh_unique,disp_y_node) 
    rows_match_x = torch.all(is_in_load_x, dim=1) 
    rows_match_y = torch.all(is_in_load_y, dim=1) 
    matching_rows_x = torch.nonzero(rows_match_x).squeeze()
    matching_rows_y = torch.nonzero(rows_match_y).squeeze()
    edge_x = mesh_unique[matching_rows_x]
    edge_y = mesh_unique[matching_rows_y]
    relative_edge_mesh_pos = torch.norm(graph_data['mesh_pos'][mesh_unique[:,0],:] - graph_data['mesh_pos'][mesh_unique[:,1],:],dim=-1)

    force_x = torch.sum(torch.mean(stress_11_node[edge_x],dim=-1) * relative_edge_mesh_pos[matching_rows_x]).numpy() 
    force_y = torch.sum(torch.mean(stress_12_node[edge_y],dim=-1) * relative_edge_mesh_pos[matching_rows_y]).numpy() 
    
    force_disp = np.append(force_disp, np.array([[disp[i] * np.cos(field_comp.theta.numpy().squeeze()) * transform_disp, force_x * transform_force,
                                                  disp[i] * np.sin(field_comp.theta.numpy().squeeze()) * transform_disp, force_y * transform_force]]), axis = 0)

energy_savedir = intermediateResults_vtk_path/Path('energy')
force_disp_savedir = intermediateResults_vtk_path/Path('force_disp')
np.save(energy_savedir,energy)
np.save(force_disp_savedir,force_disp)