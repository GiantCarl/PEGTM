import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from input_data_from_mesh import build_graph
from compute_energy import compute_energy
from torch.nn.utils import parameters_to_vector


class EarlyStopping:
    '''
    If the relative decrease in the loss is < min_delta for # of consecutive steps = tolerance,
    then the training is stopped.
    '''
    def __init__(self, tol_steps=10, min_delta=1e-3, device='cpu'):
        self.tol_steps = torch.tensor([tol_steps], dtype=torch.int, device=device)
        self.min_delta = torch.tensor([min_delta], dtype=torch.float, device=device)
        self.counter = torch.tensor([0], dtype=torch.int, device=device)
        self.early_stop = False        
        
    def __call__(self, train_loss, train_loss_prev):
        delta = torch.abs(train_loss - train_loss_prev)/(torch.abs(train_loss_prev)+np.finfo(float).eps)
        if delta > self.min_delta:
            self.counter = self.counter * 0
        else:
            self.counter += 1
            if self.counter >= self.tol_steps:  
                self.early_stop = True

def fit(
        inp_train,
        field_comp, 
        world_pos,
        hist_alpha,
        matprop, 
        pffmodel, 
        weight_decay, 
        num_epochs, 
        optimizer, 
        intermediateModel_path=None, 
        writer=None, 
        training_dict={}):
    loss_data = list()
    
    # Loop over epochs
    for epoch in range(num_epochs):    
            
        def closure():
            optimizer.zero_grad()
            if inp_train.T_conn == None:
                inp_train.requires_grad = True
            u, v, alpha,_,_ = field_comp.fieldCalculation(inp_train,world_pos,hist_alpha)
            loss_E_el, loss_E_d = compute_energy(inp_train, u, v, alpha, hist_alpha,matprop, pffmodel)            
            loss_var = torch.log10(loss_E_el + loss_E_d )

            loss_reg = 0.0
            if weight_decay != 0:
                for name, param in field_comp.net.named_parameters():
                    if 'weight' in name:
                        loss_reg += torch.sum(param**2)  
            loss = loss_var + weight_decay*loss_reg

            if writer is not None:
                writer.add_scalars('U_p_'+str(field_comp.lmbda.item()), {'loss':loss.item(), "loss_E":loss_var.item()}, epoch)

            print(f"U_p: {field_comp.lmbda.item():.5f}, \tU_p_delta: {field_comp.lmbda_delta.item():.5f}, \tEpoch [{epoch}/{num_epochs}], \tloss={loss.item():.15f}, \tloss_E={loss_var.item():.15f}")   
            
            loss_data.append(loss.item())
            if intermediateModel_path is not None:
                idx = len(loss_data)
                steps = training_dict["save_model_every_n"]
                if steps > 0 and idx >= steps and idx % steps == 0:
                    intermModel_path = intermediateModel_path/Path('intermediate_1NN_' + str(int(field_comp.lmbda*1000000)) + 'by1000000_' + str(idx) + '.pt')
                    torch.save(field_comp.net.state_dict(), intermModel_path)
                    
            loss.backward()
            return loss
        
        optimizer.step(closure=closure)

    return loss_data



def fit_with_early_stopping(inp_train,
                            field_comp, 
                            world_pos,
                            hist_alpha,
                            matprop, 
                            pffmodel, 
                            weight_decay, 
                            num_epochs, 
                            optimizer, 
                            min_delta, 
                            intermediateModel_path=None, 
                            writer=None, 
                            training_dict={},
                            device  = None):
    loss_data = list()
    early_stopping = EarlyStopping(tol_steps=20, min_delta=min_delta, device=device)
    loss_prev = torch.tensor([0.0], device=device)
    
    # Loop over epochs
    for epoch in range(num_epochs):   
            
        optimizer.zero_grad()
        if inp_train.T_conn == None:
            inp_train.requires_grad = True
        u, v, alpha,_,_ = field_comp.fieldCalculation(inp_train,world_pos,hist_alpha)
        loss_E_el, loss_E_d = compute_energy(inp_train, u, v, alpha, hist_alpha,matprop, pffmodel)            
        loss_var = torch.log10(loss_E_el + loss_E_d )

        loss_reg = 0.0
        if weight_decay != 0:
            for name, param in field_comp.net.named_parameters():
                if 'weight' in name:
                    loss_reg += torch.sum(param**2)  
        loss = loss_var + weight_decay*loss_reg

        if writer is not None:
                writer.add_scalars('U_p_'+str(field_comp.lmbda.item()), {'loss':loss.item(), "loss_E":loss_var.item()}, epoch)

        print(f"U_p: {field_comp.lmbda.item():.5f}, \tU_p_delta: {field_comp.lmbda_delta.item():.5f}, \tEpoch [{epoch}/{num_epochs}], \tloss={loss.item():.15f}, \tloss_E={loss_var.item():.15f}")

        loss_data.append(loss.item())
        if intermediateModel_path is not None:
            idx = len(loss_data)
            steps = training_dict["save_model_every_n"]
            if steps > 0 and idx >= steps and idx % steps == 0:
                intermModel_path = intermediateModel_path/Path('intermediate_1NN_' + str(int(field_comp.lmbda*1000000)) + 'by1000000_' + str(idx) + '.pt')
                torch.save(field_comp.net.state_dict(), intermModel_path)

        loss.backward()
        optimizer.step()
            
        early_stopping(loss, loss_prev)
        if early_stopping.early_stop:
            break
        loss_prev = loss

    return loss_data
