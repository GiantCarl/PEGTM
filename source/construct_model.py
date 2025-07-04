import torch
from pff_model import PFFModel
from material_properties import MaterialProperties
from network import init_xavier, GraphNeuralNetwork
from utils import set_seed

def construct_model(PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device):
    # Phase field model
    set_seed(network_dict["seed"])
    pffmodel = PFFModel(PFF_model = PFF_model_dict["PFF_model"], 
                        se_split = PFF_model_dict["se_split"],
                        tol_ir = torch.tensor(PFF_model_dict["tol_ir"], device=device))

    # Material model
    matprop = MaterialProperties(True_mat_E = torch.tensor(mat_prop_dict["Ture_mat_E"], device=device), 
                                mat_E = torch.tensor(mat_prop_dict["mat_E"], device=device), 
                                mat_nu = torch.tensor(mat_prop_dict["mat_nu"], device=device), 
                                True_w1 = torch.tensor(mat_prop_dict["True_w1"], device=device),
                                w1 = torch.tensor(mat_prop_dict["w1"], device=device), 
                                True_l0 = torch.tensor(mat_prop_dict["True_l0"], device=device),
                                l0 = torch.tensor(mat_prop_dict["l0"], device=device))

    # Neural network
    network = GraphNeuralNetwork(input_dimension = domain_extrema.shape[0], 
                                output_dimension = domain_extrema.shape[0]+1,
                                n_hidden_layers = network_dict["hidden_layers"],
                                neurons = network_dict["neurons"],
                                activation = network_dict["activation"],
                                init_coeff = network_dict["init_coeff"],
                                MPNN_layer = network_dict["MPNN_layer"],
                                diffMPS =  network_dict["diffMPS"],
                                phase_field_variable = mat_prop_dict["l0"],
                                attention_head = network_dict["attention_head"],
                                aggregate_fun = network_dict["aggregate_fun"] )
    
    return pffmodel, matprop, network