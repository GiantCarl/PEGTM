import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path

from input_data_from_mesh import prep_input_data, prep_input_graph_data, build_graph
from fit import fit, fit_with_early_stopping
from optim import *
from plotting import plot_field

def train(field_comp, disp,disp_delta, pffmodel, matprop, crack_dict,k_hop, numr_dict, 
        optimizer_dict, training_dict,  mesh_info_dict, 
        device, trainedModel_path, intermediateModel_path,intermediateResults_path, writer):
    '''
    Neural network training: pretraining with a coarser mesh in the first stage before the main training proceeds.
    
    Input is prepared from the .msh file.

    Network training to learn the solution of the BVP in step wise loading.
    Trained network from the previous load step is used for learning the solution
    in the current load step.

    Trained models and loss data are saved in the trainedModel_path directory.
    '''
    
    ## #############################################################################
    # Initial training #############################################################
    # Prepare initial input data
    graph_data, hist_alpha = prep_input_graph_data(matprop, pffmodel, crack_dict,k_hop, numr_dict, field_comp.domain_extrema, mesh_info=mesh_info_dict, device=device)
    field_comp.lmbda = torch.tensor(disp[0]).to(device)
    field_comp.lmbda_delta = torch.tensor(disp_delta[0]).to(device)
    world_pos = graph_data['mesh_pos']

    loss_data = list()
    start = time.time()

    n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "LBFGS")

    inp = build_graph(graph_data,world_pos,hist_alpha,field_comp) 

    loss_data1 = fit(inp,
                    field_comp,
                    world_pos,
                    hist_alpha,
                    matprop, 
                    pffmodel,
                    optimizer_dict["weight_decay"], 
                    num_epochs=n_epochs, 
                    optimizer=optimizer, 
                    intermediateModel_path=None, 
                    writer=writer, 
                    training_dict=training_dict)
    
    loss_data = loss_data + loss_data1

    n_epochs = optimizer_dict["n_epochs_RPROP"]
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "RPROP")

    loss_data2 = fit_with_early_stopping(inp,
                                        field_comp, 
                                        world_pos,
                                        hist_alpha,                                        
                                        matprop, 
                                        pffmodel,
                                        optimizer_dict["weight_decay"], 
                                        num_epochs=n_epochs, 
                                        optimizer=optimizer, 
                                        min_delta=optimizer_dict["optim_rel_tol_pretrain"], 
                                        intermediateModel_path=None, 
                                        writer=writer, 
                                        training_dict=training_dict,
                                        device = device)    
    loss_data = loss_data + loss_data2

    end = time.time()
    print(f"Execution time: {(end-start)/60:.03f}minutes")

    torch.save(field_comp.net.state_dict(), trainedModel_path/Path('trained_1NN_initTraining.pt'))
    with open(trainedModel_path/Path('trainLoss_1NN_initTraining.npy'), 'wb') as file:
        np.save(file, np.asarray(loss_data))

    ## #############################################################################


    ## #############################################################################
    # Main training ################################################################

    # Prepare input data
    graph_data, hist_alpha = prep_input_graph_data(matprop, pffmodel, crack_dict,k_hop, numr_dict, field_comp.domain_extrema, mesh_info=mesh_info_dict, device=device)
    world_pos = graph_data['mesh_pos']

    # solve BVP by step wise loading.
    for j, disp_i in enumerate(disp):
        field_comp.lmbda = torch.tensor(disp_i).to(device)
        field_comp.lmbda_delta = torch.tensor(disp_delta[j]).to(device)
        inp = build_graph(graph_data,world_pos,hist_alpha,field_comp) 
        print(f'idx: {j}; displacement: {field_comp.lmbda}; \t delt—displacement: {field_comp.lmbda_delta}')
        loss_data = list()

        start = time.time()
        
        if j == 0 or optimizer_dict["n_epochs_LBFGS"] > 0:
            n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
            NNparams = field_comp.net.parameters()
            optimizer = get_optimizer(NNparams, "LBFGS")
            loss_data1 = fit(inp,
                            field_comp, 
                            world_pos,
                            hist_alpha, 
                            matprop, 
                            pffmodel,
                            optimizer_dict["weight_decay"], 
                            num_epochs=n_epochs, 
                            optimizer=optimizer,
                            intermediateModel_path=None, 
                            writer=writer, 
                            training_dict=training_dict)
            loss_data = loss_data + loss_data1

        if optimizer_dict["n_epochs_RPROP"] > 0:
            n_epochs = optimizer_dict["n_epochs_RPROP"]
            NNparams = field_comp.net.parameters()
            optimizer = get_optimizer(NNparams, "RPROP")
            loss_data2 = fit_with_early_stopping(inp,
                                                field_comp, 
                                                world_pos,
                                                hist_alpha,  
                                                matprop, 
                                                pffmodel,
                                                optimizer_dict["weight_decay"], 
                                                num_epochs=n_epochs, 
                                                optimizer=optimizer, 
                                                min_delta=optimizer_dict["optim_rel_tol"],
                                                intermediateModel_path=intermediateModel_path, 
                                                writer=writer, 
                                                training_dict=training_dict,
                                                device = device)
            loss_data = loss_data + loss_data2

        end = time.time()
        print(f"Execution time: {(end-start)/60:.03f}minutes")
 
        world_pos,hist_alpha = field_comp.update_hist_alpha(inp,world_pos,hist_alpha)
        intermediateResults_file = intermediateResults_path/Path('intermediate_disp_' +f'{j:04}')
        np.savez(intermediateResults_file, world_pos = world_pos.cpu().numpy(), phase_field = hist_alpha.cpu().numpy())

        torch.save(field_comp.net.state_dict(), trainedModel_path/Path('trained_1NN_' + str(j) + '.pt'))
        with open(trainedModel_path/Path('trainLoss_1NN_' + str(j) + '.npy'), 'wb') as file:
            np.save(file, np.asarray(loss_data))
