import torch
import numpy as np
from pathlib import Path
from utils import parse_mesh, hist_alpha_init,parse_mesh_from_gmsh,expand_hop_edges
from network import EdgeSet,MultiGraph,BCSet

# Prepare input data
def prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file, device):
    '''
    Input data is prepared from the .msh file.
    If gradient_type = numerical:  
        X, Y = nodal coordinates
        T_conn = connectivity
    If gradient_type = autodiff:   
        X, Y = coordinate of the Gauss point in one point Gauss quadrature
        T_conn = None
    area_T: area of elements

    hist_alpha = initial alpha field

    '''
    assert Path(mesh_file).suffix == '.msh', "Mesh file should be a .msh file"
    
    X, Y, T_conn, area_T = parse_mesh(filename = mesh_file, gradient_type=numr_dict["gradient_type"])

    inp = torch.from_numpy(np.column_stack((X, Y))).to(torch.float).to(device)
    T_conn = torch.from_numpy(T_conn).to(torch.long).to(device)
    area_T = torch.from_numpy(area_T).to(torch.float).to(device)
    if numr_dict["gradient_type"] == 'autodiff':
        T_conn = None

    hist_alpha = hist_alpha_init(inp, matprop, pffmodel, crack_dict)
    
    return inp, T_conn, area_T, hist_alpha

# Prepare input Graph data
def prep_input_graph_data(matprop, pffmodel, crack_dict,k_hop, numr_dict,domain_extrema, mesh_info, device):

    assert Path(mesh_info["mesh_file"]).suffix == '.msh', "Mesh file should be a gmsh file"
    inp, T_conn,edges,area_T,loading_geo,crack_point = parse_mesh_from_gmsh(mesh_info,crack_dict)
    
    inp = torch.from_numpy(inp).to(torch.float).to(device)
    hist_alpha = hist_alpha_init(inp, matprop, pffmodel, crack_dict,crack_point)

    Lx = domain_extrema[0,1] - domain_extrema[0,0]
    Ly = domain_extrema[1,1] - domain_extrema[1,0]
    L = torch.max(Lx,Ly)    
    
    inp[:,0] = inp[:,0]/L
    inp[:,1] = inp[:,1]/L    
    T_conn = torch.from_numpy(T_conn).to(torch.long).to(device)
    area_T = torch.from_numpy(area_T).to(torch.float).to(device)/L/L
    edges =  torch.from_numpy(edges).to(torch.long).to(device)
    edges = expand_hop_edges(edges,num_nodes = inp.shape[0],n_hop = k_hop,contain_self = mesh_info["contain_self"])

    node_type = torch.zeros_like(inp).to(torch.long).to(device)
    disp_node = torch.zeros_like(inp).to(torch.float).to(device)
    fix_X_boundary = torch.from_numpy(loading_geo['fix_X_boundary'].squeeze()).to(torch.long).to(device)
    fix_Y_boundary = torch.from_numpy(loading_geo['fix_Y_boundary'].squeeze()).to(torch.long).to(device)
    disp_X_boundary = torch.from_numpy(loading_geo['disp_X_boundary'].squeeze()).to(torch.long).to(device)
    disp_Y_boundary = torch.from_numpy(loading_geo['disp_Y_boundary'].squeeze()).to(torch.long).to(device)
    tract_X_boundary = torch.from_numpy(loading_geo['tract_X_boundary'].squeeze()).to(torch.long).to(device)
    tract_Y_boundary = torch.from_numpy(loading_geo['tract_Y_boundary'].squeeze()).to(torch.long).to(device)
    node_type[fix_X_boundary,0] = 1
    node_type[fix_Y_boundary,1] = 1
    node_type[disp_X_boundary,0] = 1
    node_type[disp_Y_boundary,1] = 1
    disp_node[disp_X_boundary,0] = 1
    disp_node[disp_Y_boundary,1] = 1

    graph_data = {'mesh_pos':inp,
                  'T_conn':T_conn,
                  "edges":edges,
                  'area_T':area_T,
                  'node_type':node_type,
                  'disp_node':disp_node,
                  'fix_X':fix_X_boundary,
                  'fix_Y':fix_Y_boundary,
                  'disp_X':disp_X_boundary,
                  'disp_Y':disp_Y_boundary
    }

    # hist_alpha = hist_alpha_init(graph_data['mesh_pos'], matprop, pffmodel, crack_dict,crack_point)

    return graph_data, hist_alpha

def build_graph(graph_data,world_pos,hist_alpha,field_comp,):
    load_disp_delta = field_comp.lmbda_delta
    node_features = torch.cat([graph_data['mesh_pos'],world_pos,hist_alpha.reshape(-1,1),graph_data['node_type'],load_disp_delta*graph_data['disp_node']],dim=-1).to(torch.float)

    edges = graph_data['edges']
    senders  = edges[:,0]
    receivers = edges[:,1]

    mesh_edge = EdgeSet(
            name = "mesh_edges",            
            receivers = receivers,
            senders = senders)     
    bcSet = BCSet(
        name = "BoundaryCondition",
        fix_X_boundary = graph_data['fix_X'],
        fix_Y_boundary = graph_data['fix_Y'],
        disp_X_boundary = graph_data['disp_X'],
        disp_Y_boundary = graph_data['disp_Y']
    )

    return MultiGraph(node_features = node_features,
                          edge_sets = mesh_edge,
                          T_conn = graph_data['T_conn'],
                          area_elem = graph_data['area_T'],
                          bc_set = bcSet
                          )