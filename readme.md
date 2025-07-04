# **The physics-enhanced graph neural network for phase-field fracture modelling**
***

This is an open source object oriented Python project developed for the simulation of fracture in brittle materials via deep learning with phase field approch. 

If you have any questions, feel free to contact: fengbo19940401@126.com.

## Structure of the subdirectories
* ./source
  - all source files are included in this folder
* ./examples 
  - benchmark problems 

## Instruction
To run any of the benchmark problems, place the related "examples/Main.py" in the root directory. 

## Explanation and description
* ./example/Dogbone/cofig.py
  - The config.py file is used to set material parameters, network parameters, and mesh information.
  ### Network seeting 
  ```bibtex
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
   ```
  ### Material parameter setting
  ```bibtex
    numr_dict = {"alpha_constraint": 'nonsmooth', "gradient_type": 'numerical'}
    PFF_model_dict = {"PFF_model" : 'AT1', "se_split" : 'volumetric', "tol_ir" : 1e-3}
    mat_prop_dict = {"Ture_mat_E" : 210, "mat_E" : 1.0, "mat_nu" : 0.3,
                    "True_w1" : 0.0027, "w1" : 1.0,
                    "True_l0" : 0.01, "l0" : 0.01/L}
   ```
  ### mesh information 
    ```bibtex
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
     ```
     ```bibtex
     # L： characteristic length of the physical system mesh
    domain_extrema = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
    Lx = domain_extrema[0,1] - domain_extrema[0,0]
    Ly = domain_extrema[1,1] - domain_extrema[1,0]
    L = torch.max(Lx,Ly).numpy()  
     ```
    ### loading 
    ```bibtex
    loading_angle = torch.tensor([np.pi/2])
    disp = np.concatenate((np.linspace(0.0, 0.1, 26), np.linspace(0.1, 0.2, 101)[1:]), axis=0)
    disp_delta  = np.diff(disp)
    disp = disp[1:]
     ```
* ./example/Dogbone/field_computation.py
  - field_computation.py is used to set boundary and loading conditions.
  ```bibtex
  u = ((inp.node_features[:, 1]-self.y0)*(self.yL-inp.node_features[:, 1])*out_disp[:, 0] + (inp.node_features[:, 1]-self.y0)/(self.yL-self.y0)*torch.cos(self.theta))*self.lmbda_delta + (world_pos[:,0] - inp.node_features[:,0]) * self.non_dim 
  v = ((inp.node_features[:, 1]-self.y0)*(self.yL-inp.node_features[:, 1])*out_disp[:, 1] + (inp.node_features[:, 1]-self.y0)/(self.yL-self.y0)*torch.sin(self.theta))*self.lmbda_delta + (world_pos[:,1] - inp.node_features[:,1]) * self.non_dim 
   ```
## Citation 
If you find our work and/or our code useful, please cite us via:
```bibtex
@article{FENG2024117410,
    title = {The novel graph transformer-based surrogate model for learning physical systems},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {432},
    pages = {117410},
    year = {2024},
    issn = {0045-7825},
    doi = {https://doi.org/10.1016/j.cma.2024.117410},
    url = {https://www.sciencedirect.com/science/article/pii/S0045782524006650},
    author = {Bo Feng and Xiao-Ping Zhou}
}
```