import numpy as np
import torch
from pathlib import Path
import sys
from tensorboardX import SummaryWriter
'''
## ############################################################################
Refer to the paper 
"Phase-field modeling of fracture with physics-informed deep learning"
for details of the model.
## ############################################################################
'''

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

## ############################################################################
## customized for each problem ################################################
## ############################################################################

'''
network_dict:
parameters to construct an MLP
seed: seed to initialize the network
activation: choose from {SteepTanh, SteepReLU, TrainableTanh, TrainableReLU,ReLU,SiLU,LeakyReLU,Tanh}
init_coeff: initial coefficient in activation function 
setting init_coeff = 1 in SteepTanh/SteepReLU gives standard Tanh/ReLU activation
'''

network_dict = {"model_type": 'GNN',
                "MPNN_layer":   8,
                "hidden_layers": 2,
                "neurons": 128,
                "diffMPS":True,
                "attention_head":8,
                "K_hop":1,
                "aggregate_fun":"attention", #add, mean, attention
                "seed": int(sys.argv[3]) if len(sys.argv) > 3 else 1,
                "activation": str(sys.argv[4]) if len(sys.argv) > 4 else 'SiLU',
                "init_coeff": float(sys.argv[5]) if len(sys.argv) > 5 else 2.0}

'''
optimizer_dict:
weight_decay: weighing of neural network weight regularization
optim_rel_tol_pretrain: relative tolerance of loss in pretraining as an stopping criteria
optim_rel_tol: relative tolerance of loss in main training as an stopping criteria
'''

optimizer_dict = {"weight_decay": 1e-6,
                  "n_epochs_RPROP": 10000,
                  "n_epochs_LBFGS": 0,
                  "optim_rel_tol_pretrain": 1e-7,
                  "optim_rel_tol": 1e-8}

mesh_info_dict = {"mesh_file":"Coalescence.msh",
                  "contain_self":True,
                  "fix_X_boundary":"bottom_edge+top_edge",
                  "fix_Y_boundary":"bottom_edge",
                  "disp_X_boundary":None,
                  "disp_Y_boundary":"top_edge",
                  "tract_X_boundary":None,
                  "tract_Y_boundary":None,
                  "elem_type":"triangle",
                  "Pre_Crack_type":"smeard"  #smeard  ,discrete   
                  }


# save intermediate model during training every "save_model_every_n" steps
training_dict = {"save_model_every_n": 10000}


# Domain definition
'''
domain_extrema: tensor([[x_min, x_max], [y_min, y_max]])
x_init: list of x-coordinates of one end of cracks
y_init: list of y-coordinates of one end of cracks
L_crack: list of crack lengths
angle_crack: list of angles of cracks from the x-axis with the origin shifted to (x_init[i], y_init[i])
'''
domain_extrema = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
Lx = domain_extrema[0,1] - domain_extrema[0,0]
Ly = domain_extrema[1,1] - domain_extrema[1,0]
L = torch.max(Lx,Ly).numpy()  

'''
numr_dict:
"alpha_constraint" in {'nonsmooth', 'smooth','phasefieldexp','None',adaptive_Relu}
"gradient_type" in {'numerical', 'autodiff'}

PFF_model_dict:
PFF_model in {'AT1', 'AT2'} 
se_split in {'volumetric', None}
tol_ir: irreversibility tolerance

mat_prop_dict:
w1: Gc/l0, where Gc is energy release rate.
In the normalized formulation, mat_E=1, w1=1, and only nu and l0 are the properties to be set.
'''
numr_dict = {"alpha_constraint": 'nonsmooth', "gradient_type": 'numerical'}
PFF_model_dict = {"PFF_model" : 'AT1', "se_split" : 'volumetric', "tol_ir" : 1e-3}
mat_prop_dict = {"Ture_mat_E" : 210, "mat_E" : 1.0, "mat_nu" : 0.3,
                 "True_w1" : 0.0027, "w1" : 1.0,
                 "True_l0" : 0.01, "l0" : 0.01/L}

crack_dict = {"x_init" : [-0.5,0.5], "y_init" : [-0.05,0.05], "L_crack" : [0.3,0.3], "angle_crack" : [0,-np.pi]}


# Prescribed incremental displacement
loading_angle = torch.tensor([np.pi/2])
disp = np.concatenate((np.linspace(0.0, 0.1, 26), np.linspace(0.1, 0.2, 101)[1:]), axis=0)
disp_delta  = np.diff(disp)
disp = disp[1:]

## ############################################################################
## ############################################################################

## ############################################################################
## Setting output directory ###################################################
## ############################################################################
PATH_ROOT = Path(__file__).parents[0]
model_path = PATH_ROOT/Path("model_type_" + network_dict["model_type"]+
                            "_MPNN_layer_" + str(network_dict["MPNN_layer"])+
                            '_hl_'+str(network_dict["hidden_layers"])+
                            '_diffMPS_' + str(network_dict["diffMPS"])+
                            '_Neurons_'+str(network_dict["neurons"])+
                            '_activation_'+network_dict["activation"]+
                            '_coeff_'+str(network_dict["init_coeff"])+
                            '_constraint_'+str(numr_dict["alpha_constraint"])+
                            '_Seed_'+str(network_dict["seed"])+
                            '_PFFmodel_'+str(PFF_model_dict["PFF_model"])+
                            '_gradient_'+str(numr_dict["gradient_type"]))

model_path.mkdir(parents=True, exist_ok=True)
trainedModel_path = model_path/Path('best_models/')
trainedModel_path.mkdir(parents=True, exist_ok=True)
intermediateModel_path = model_path/Path('intermediate_models/')
intermediateModel_path.mkdir(parents=True, exist_ok=True)
intermediateResults_path = model_path/Path('intermediate_results/')
intermediateResults_path.mkdir(parents=True, exist_ok=True)
intermediateResults_vtk_path = model_path/Path('intermediate_VTK_results/')
intermediateResults_vtk_path.mkdir(parents=True, exist_ok=True)

with open(model_path/Path('model_settings.txt'), 'w') as file:
    file.write(f'model_type: {network_dict["model_type"]}')
    file.write(f'\nhidden_layers: {network_dict["hidden_layers"]}')
    file.write(f'\nneurons: {network_dict["neurons"]}')
    file.write(f'\nseed: {network_dict["seed"]}')
    file.write(f'\nactivation: {network_dict["activation"]}')
    file.write(f'\ncoeff: {network_dict["init_coeff"]}')
    file.write(f'\nPFF_model: {PFF_model_dict["PFF_model"]}')
    file.write(f'\nse_split: {PFF_model_dict["se_split"]}')
    file.write(f'\nalpha_constraint: {numr_dict["alpha_constraint"]}')
    file.write(f'\ngradient_type: {numr_dict["gradient_type"]}')
    file.write(f'\ndiffMPS: {network_dict["diffMPS"]}')
    file.write(f'\ndevice: {device}')

## #############################################################################
## #############################################################################

# logging loss to tensorboard
writer = SummaryWriter(model_path/Path('TBruns'))
