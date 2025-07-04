import torch
import torch.nn as nn
import numpy as np
import warnings
from torch_geometric.nn import MessagePassing
import torch_scatter
from torch import Tensor
import math
import collections
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'senders','receivers'])
BCSet = collections.namedtuple('BC', ['name', 'fix_X_boundary', 'fix_Y_boundary','disp_X_boundary','disp_Y_boundary'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets','T_conn','area_elem','bc_set'])

class SteepTanh(nn.Module):
    def __init__(self, coeff):
        super(SteepTanh, self).__init__()
        self.coeff = coeff
        
    def forward(self, x):
        activation = nn.Tanh()
        return activation(self.coeff*x)
    
class SteepReLU(nn.Module):
    def __init__(self, coeff):
        super(SteepReLU, self).__init__()
        self.coeff = coeff
        
    def forward(self, x):
        activation = nn.ReLU()
        return activation(self.coeff*x)
    
class TrainableTanh(nn.Module):
    def __init__(self, init_coeff):
        super(TrainableTanh, self).__init__()
        self.coeff = nn.Parameter(torch.tensor(init_coeff))
    def forward(self, x):
        activation = nn.Tanh()
        return activation(self.coeff*x)
    
class TrainableReLU(nn.Module):
    def __init__(self, init_coeff):
        super(TrainableReLU, self).__init__()
        self.coeff = nn.Parameter(torch.tensor(init_coeff))
    def forward(self, x):
        activation = nn.ReLU()
        return activation(self.coeff*x)

          
def init_xavier(model,activation,init_coeff):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if activation == 'TrainableReLU' or activation == 'SteepReLU':
                g = nn.init.calculate_gain('leaky_relu', np.sqrt(init_coeff**2-1.0))
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                # m.bias.data.fill_(0)
                nn.init.zeros_(m.bias)
            elif activation == 'TrainableTanh' or activation == 'SteepTanh':
                g = nn.init.calculate_gain('Tanh')/init_coeff
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)
                nn.init.zeros_(m.bias)
            elif activation == 'ReLU' or activation == 'PReLU':
                g = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight,gain = g)
                nn.init.zeros_(m.bias)
            elif activation == 'SiLU':
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
                # nn.init.kaiming_uniform_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            elif activation == 'Tanh':
                g = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight,gain = g)
                nn.init.zeros_(m.bias)
            elif activation == 'LeakyReLU':
                g = nn.init.calculate_gain('leaky_relu')
                nn.init.xavier_uniform_(m.weight,gain = g)
                nn.init.zeros_(m.bias)
            elif activation == None:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            else:
                warnings.warn('Prescribed activation does not match the available choices. The default initialization mode (xavier_iniform) is in use.')

    model.apply(init_weights)

class MLP(nn.Module):
    def __init__(self, 
                input_size: int,
                hidden_layer_sizes:  int = None,
                output_size: int = None,
                num_hidden_layers: int = None,
                outermost_linear:bool = None,    
                activation: nn.Module = nn.ReLU,
                layer_norm: bool = None,
                init_coeff:float = None
                ) -> None:
        super().__init__()
        """Build a MultiLayer Perceptron.

        Args:
            input_size: Size of input layer.
            layer_sizes: An array of input size for each hidden layer.
            output_size: Size of the output layer.
            output_activation: Activation function for the output layer.
            activation: Activation function for the hidden layers.

        Returns:
            mlp: An MLP sequential container.
        """

        # Size of each layer
        layer_sizes = [input_size] + [hidden_layer_sizes] * num_hidden_layers
        if output_size:
            layer_sizes.append(output_size)

         # Number of layers
        nlayers = len(layer_sizes) - 1

        if activation == 'SteepTanh':
            act = [SteepTanh(init_coeff) for _ in range(nlayers)]
            self.trainable_activation = False
        elif activation == 'SteepReLU':
            act = [SteepReLU(init_coeff) for _ in range(nlayers)]
            self.trainable_activation = False
        elif activation == 'TrainableTanh':
            act = [TrainableTanh(init_coeff) for _ in range(nlayers)]
            self.trainable_activation = True
        elif activation == 'TrainableReLU':
            act = [TrainableReLU(init_coeff) for _ in range(nlayers)]
            self.trainable_activation = True
        elif activation == 'ReLU':
            act = [nn.ReLU() for _ in range(nlayers)]   
            self.trainable_activation = False
        elif activation == 'SiLU' :      
            act = [nn.SiLU() for _ in range(nlayers)]  
            self.trainable_activation = False
        elif activation == 'LeakyReLU' :      
            act = [nn.LeakyReLU(negative_slope=0.2) for _ in range(nlayers)]  
            self.trainable_activation = False
        elif activation == 'Tanh' :      
            act = [nn.Tanh() for _ in range(nlayers)]  
            self.trainable_activation = False
        elif activation == 'PReLU' :      
            act = [nn.PReLU() for _ in range(nlayers)]  
            self.trainable_activation = False
        else:
            raise('Prescribed activation does not match the available choices.')   
        
        if outermost_linear:
            act[-1] = None

        # Create a torch sequential container
        self.mlp = nn.Sequential()

        for i in range(nlayers):
            self.mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                                    layer_sizes[i + 1]))
            if act[i] :
                self.mlp.add_module("Act-" + str(i), act[i])
                
        if layer_norm:
            self.mlp.add_modules("Layer_Norm",nn.LayerNorm([layer_sizes[-1]]))

        init_xavier(self.mlp,activation=activation,init_coeff=init_coeff)

    def forward(self,x):

        return self.mlp(x) 
    
class processor_mean(nn.Module):
    def __init__(
        self, 
        latent_size,
        attention_head ,
        n_mlp_layers ,
        act_name ,
        layer_norm ,
        latent_size_mlp ,
        init_coeff ,
        name = 'processor_mean'
    ):
        super().__init__()  
        #Training setting
        self.latent_size = latent_size
        self.attention_head = attention_head

        #MLP setting
        self.n_mlp_layers = n_mlp_layers
        self.act_name = act_name
        self.layer_norm = layer_norm
        self.latent_size_mlp = latent_size_mlp    
        self.init_coeff = init_coeff 
        self.name = name

        self.node_processor = MLP(input_size = self.latent_size_mlp,
                                 hidden_layer_sizes = self.latent_size_mlp,
                                 output_size = self.latent_size_mlp,
                                 num_hidden_layers = self.n_mlp_layers,
                                 outermost_linear = False,
                                 activation = self.act_name,
                                 layer_norm = self.layer_norm,
                                 init_coeff = self.init_coeff
                                 )
        
    def forward(self, graph):

        senders = graph.edge_sets.senders
        receivers = graph.edge_sets.receivers
        
        #aggregate
        node_latent = torch_scatter.scatter_mean(graph.node_features[senders,:],receivers.to(torch.int64),dim = 0)

        #update
        node_latent = self.node_processor(node_latent) + graph.node_features

        return   MultiGraph(node_features = node_latent,
                          edge_sets = graph.edge_sets,
                          T_conn = graph.T_conn,
                          area_elem = graph.area_elem,
                          bc_set = graph.bc_set
                          )
    
class processor_add(nn.Module):
    def __init__(
        self, 
        latent_size,
        attention_head ,
        n_mlp_layers ,
        act_name ,
        layer_norm ,
        latent_size_mlp ,
        init_coeff ,
        name = 'processor_add'
    ):
        super().__init__()  
        #Training setting
        self.latent_size = latent_size
        self.attention_head = attention_head

        #MLP setting
        self.n_mlp_layers = n_mlp_layers
        self.act_name = act_name
        self.layer_norm = layer_norm
        self.latent_size_mlp = latent_size_mlp    
        self.init_coeff = init_coeff 
        self.name = name

        self.node_processor = MLP(input_size = self.latent_size_mlp,
                                 hidden_layer_sizes = self.latent_size_mlp,
                                 output_size = self.latent_size_mlp,
                                 num_hidden_layers = self.n_mlp_layers,
                                 outermost_linear = False,
                                 activation = self.act_name,
                                 layer_norm = self.layer_norm,
                                 init_coeff = self.init_coeff
                                 )
        
    def forward(self, graph):

        senders = graph.edge_sets.senders
        receivers = graph.edge_sets.receivers
        
        #aggregate
        node_latent = torch_scatter.scatter_add(graph.node_features[senders,:],receivers.to(torch.int64),dim = 0)

        #update
        node_latent = self.node_processor(node_latent) + graph.node_features

        return   MultiGraph(node_features = node_latent,
                          edge_sets = graph.edge_sets,
                          T_conn = graph.T_conn,
                          area_elem = graph.area_elem,
                          bc_set = graph.bc_set
                          )

class processor_attention(nn.Module):
    def __init__(
        self, 
        latent_size,
        attention_head ,
        n_mlp_layers ,
        act_name ,
        layer_norm ,
        latent_size_mlp ,
        init_coeff, 
        name = 'processor_GT'
    ):
        super().__init__()  
        #Training setting
        self.latent_size = latent_size
        self.attention_head = attention_head

        #MLP setting
        self.n_mlp_layers = n_mlp_layers
        self.act_name = act_name
        self.layer_norm = layer_norm
        self.latent_size_mlp = latent_size_mlp    
        self.init_coeff = init_coeff 
        self.mid_channels = self.latent_size_mlp
        self.name = name

        self.node_query_layer = nn.Linear(self.latent_size, self.latent_size_mlp,bias=False)
        nn.init.kaiming_normal_(self.node_query_layer.weight,mode='fan_in',nonlinearity='leaky_relu')
        # nn.init.zeros_(self.node_query_layer.bias)
        self.node_key_layer = nn.Linear(self.latent_size, self.latent_size_mlp,bias=False)
        nn.init.kaiming_normal_(self.node_key_layer.weight,mode='fan_in',nonlinearity='leaky_relu')
        # nn.init.zeros_(self.node_key_layer.bias)
        self.node_value_layer = nn.Linear(self.latent_size, self.latent_size_mlp,bias=False)
        nn.init.kaiming_normal_(self.node_value_layer.weight,mode='fan_in',nonlinearity='leaky_relu')
        # nn.init.zeros_(self.node_value_layer.bias)

        self.node_processor = MLP(input_size = self.latent_size_mlp,
                                 hidden_layer_sizes = self.latent_size_mlp,
                                 output_size = self.latent_size_mlp,
                                 num_hidden_layers = self.n_mlp_layers,
                                 outermost_linear = False,
                                 activation = self.act_name,
                                 layer_norm = self.layer_norm,
                                 init_coeff = self.init_coeff
                                 )
        
    def forward(self, graph):

        senders = graph.edge_sets.senders
        receivers = graph.edge_sets.receivers

        # linear transform node_features
        node_features_query = self.node_query_layer(graph.node_features)       
        node_features_key = self.node_key_layer(graph.node_features)
        node_features_value = self.node_value_layer(graph.node_features)
        node_features_query_head = node_features_query.view(-1, self.attention_head, int(self.mid_channels / self.attention_head)).permute(1, 0, 2)
        node_features_key_head = node_features_key.view(-1, self.attention_head, int(self.mid_channels / self.attention_head)).permute(1, 0, 2) 
        node_features_value_head = node_features_value.view(-1, self.attention_head, int(self.mid_channels / self.attention_head)).permute(1, 0, 2) 

        alpha = torch.transpose(torch.sum(node_features_query_head[:,receivers,:] * node_features_key_head[:,senders,:], dim=-1) / torch.sqrt(
                torch.tensor(self.mid_channels / self.attention_head, dtype=torch.float32)), 0, 1) 
        alpha_max,_ = torch_scatter.scatter_max(alpha,receivers.to(torch.long), dim = 0)
        alpha = torch.exp(alpha - alpha_max[receivers])
        Attention =  torch.transpose(alpha / torch_scatter.scatter_add(alpha,receivers.to(torch.int64),dim=0)[receivers],dim0=0,dim1=1).unsqueeze(-1) 
        node_latent = (torch_scatter.scatter_add((Attention * node_features_value_head[:,senders,:]).permute(1,2,0),receivers.to(torch.int64),dim = 0)).permute(0,2,1).reshape(-1, self.mid_channels) 
        node_latent = self.node_processor(node_latent) + graph.node_features

        return   MultiGraph(node_features = node_latent,
                          edge_sets = graph.edge_sets,
                          T_conn = graph.T_conn,
                          area_elem = graph.area_elem,
                          bc_set = graph.bc_set
                          )
    
    
class GraphNeuralNetwork(nn.Module):
    """Graph neural network
    Attributes::

    """
    def __init__(self, 
                input_dimension = 3,
                output_dimension = 2,
                n_hidden_layers = 2, 
                neurons = 128,
                activation = None,
                init_coeff = 1.0,
                MPNN_layer = 4,  
                layer_norm = False,
                diffMPS = False, 
                phase_field_variable = 1.0,
                attention_head = 8,
                aggregate_fun = None
                 ) -> None:
        
        super().__init__()

        #Training setting
        self.model_type = "GNN"
        self.message_passing_steps = MPNN_layer  
        self.input_dimension = input_dimension*4+1
        self.output_dimension = output_dimension
        self.edge_input_dimension = input_dimension*3 
        self.neurons = neurons
        self.diffMPS = diffMPS
        self.attention_head = attention_head
        self.init_coeff = init_coeff

        self.phase_field_para = nn.Parameter(data=torch.tensor(phase_field_variable))

        #MLP setting
        self.n_mlp_layers = n_hidden_layers
        self.latent_size_mlp = neurons
        self.layer_norm = layer_norm
        self.name_activation = activation
        
        # Encoder        
        self.encoder_nodes = MLP(input_size = self.input_dimension,
                                 hidden_layer_sizes = self.latent_size_mlp,
                                 output_size = self.latent_size_mlp,
                                 num_hidden_layers = self.n_mlp_layers,
                                 outermost_linear = True,
                                 activation = self.name_activation,
                                 layer_norm = self.layer_norm,
                                 init_coeff = self.init_coeff
                                )      

        # Processor
        if diffMPS:
            # message passing with different MLP for each steps
            self.processors = []
            for _ in range(self.message_passing_steps):
                if aggregate_fun == 'attention':
                    self.processors.append((processor_attention(latent_size = self.latent_size_mlp,
                                                        attention_head = self.attention_head,
                                                        n_mlp_layers = self.n_mlp_layers,
                                                        act_name = self.name_activation,
                                                        layer_norm = self.layer_norm,
                                                        latent_size_mlp = self.latent_size_mlp,
                                                        init_coeff = self.init_coeff
                                                    )))
                elif aggregate_fun == 'add':
                    self.processors.append((processor_add(latent_size = self.latent_size_mlp,
                                                        attention_head = self.attention_head,
                                                        n_mlp_layers = self.n_mlp_layers,
                                                        act_name = self.name_activation,
                                                        layer_norm = self.layer_norm,
                                                        latent_size_mlp = self.latent_size_mlp,
                                                        init_coeff = self.init_coeff
                                                    )))
                elif aggregate_fun == 'mean':
                                self.processors.append((processor_mean(latent_size = self.latent_size_mlp,
                                                        attention_head = self.attention_head,
                                                        n_mlp_layers = self.n_mlp_layers,
                                                        act_name = self.name_activation,
                                                        layer_norm = self.layer_norm,
                                                        latent_size_mlp = self.latent_size_mlp,
                                                        init_coeff = self.init_coeff
                                                    )))
            
            self.processors = torch.nn.Sequential(*self.processors)
        else:
            if aggregate_fun == 'attention':
                self.processors = ((processor_attention(latent_size = self.latent_size_mlp,
                                                        attention_head = self.attention_head,
                                                        n_mlp_layers = self.n_mlp_layers,
                                                        act_name = self.name_activation,
                                                        layer_norm = self.layer_norm,
                                                        latent_size_mlp = self.latent_size_mlp,
                                                        init_coeff = self.init_coeff
                                                        )))
            elif aggregate_fun == 'add':
                self.processors = ((processor_add(latent_size = self.latent_size_mlp,
                                                        attention_head = self.attention_head,
                                                        n_mlp_layers = self.n_mlp_layers,
                                                        act_name = self.name_activation,
                                                        layer_norm = self.layer_norm,
                                                        latent_size_mlp = self.latent_size_mlp,
                                                        init_coeff = self.init_coeff
                                                        )))
            elif aggregate_fun == 'mean':
                self.processors = ((processor_mean(latent_size = self.latent_size_mlp,
                                                        attention_head = self.attention_head,
                                                        n_mlp_layers = self.n_mlp_layers,
                                                        act_name = self.name_activation,
                                                        layer_norm = self.layer_norm,
                                                        latent_size_mlp = self.latent_size_mlp,
                                                        init_coeff = self.init_coeff
                                                        )))
        # Decoder
        self.decoder_node = MLP(input_size = self.latent_size_mlp,
                                 hidden_layer_sizes = self.latent_size_mlp,
                                 output_size = self.output_dimension,
                                 num_hidden_layers = self.n_mlp_layers,
                                 outermost_linear = True,
                                 activation = self.name_activation,
                                 layer_norm = self.layer_norm,
                                 init_coeff = self.init_coeff
                                )
        
    def encoder_block (self,graph):

        # node encoder    
        node_latent = self.encoder_nodes(graph.node_features)

        # update data Dict
        node_features = node_latent

        return MultiGraph(node_features = node_features,
                          edge_sets = graph.edge_sets,
                          T_conn = graph.T_conn,
                          area_elem = graph.area_elem,
                          bc_set = graph.bc_set
                         )
    
    def processor_block(self,graph):
        
        #Aggration
        if self.diffMPS:
            for i in range(self.message_passing_steps):
                graph = self.processors[i](graph)

        else:
            for _ in range(self.message_passing_steps):
                graph = self.processors(graph)

        return  graph
    
    def decoder_block(self,graph):

        node_latent = self.decoder_node(graph.node_features) 
        
        return node_latent
        
    def forward(self, data):

        
        #Encoder block
        graph_latent = self.encoder_block(data)

        #Processor block
        graph_latent = self.processor_block(graph_latent)

        #Decoder block
        prediction = self.decoder_block(graph_latent)

        return prediction         

    

