from config import *
import time
# 获取当前时间戳
start_time = time.time()

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE/Path('source')))

from field_computation import FieldComputation
from construct_model import construct_model
from model_train import train
torch.manual_seed(network_dict["seed"])

## ############################################################################
## Model construction #########################################################
## ############################################################################
pffmodel, matprop, network = construct_model(PFF_model_dict, mat_prop_dict, 
                                             network_dict, domain_extrema, device)
field_comp = FieldComputation(net = network,
                              domain_extrema = domain_extrema, 
                              lmbda = torch.tensor([0.0], device = device), 
                              lmbda_delta = torch.tensor([0.0], device = device), 
                              theta = loading_angle, 
                              matprop = matprop,
                              alpha_constraint = numr_dict["alpha_constraint"])

field_comp.net = field_comp.net.to(device)
field_comp.domain_extrema = field_comp.domain_extrema.to(device)
field_comp.theta = field_comp.theta.to(device)
k_hop = network_dict["K_hop"]

## #############################################################################
## #############################################################################

## #############################################################################
# Training #####################################################################
## #############################################################################
if __name__ == "__main__":
    train(field_comp, disp,disp_delta, pffmodel, matprop, crack_dict,k_hop, numr_dict,
          optimizer_dict, training_dict, mesh_info_dict, 
          device, trainedModel_path, intermediateModel_path, intermediateResults_path, writer)
    
    end_time = time.time()
    # 计算时间差
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
