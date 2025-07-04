import torch
import torch.nn as nn
from utils import DistanceFunction

class FieldComputation:
    '''
    This class constructs the displacement and phase fields from the NN outputs by baking in the
    Dirichlet boundary conditions (BCs) and other constraints.

    net: neural network
    domain_extrema: tensor([[x_min, x_max], [y_min, y_max]])
    lmbda: prescribed displacement
    theta: Angle of the direction of loading from the x-axis (not used in all problems)
    alpha_ansatz: type of function to constrain alpha in {'smooth', 'nonsmooth'}

    fieldCalculation: applies BCs amd constraint on alpha (needs to be customized for each problem)

    update_hist_alpha: alpha_field for use in the next loading step to enforce irreversibility

    '''
    def __init__(self, net, domain_extrema, lmbda, lmbda_delta, theta, matprop,alpha_constraint = 'nonsmooth'):
        self.net = net
        self.domain_extrema = domain_extrema
        self.theta = theta
        self.lmbda = lmbda
        self.lmbda_delta = lmbda_delta
        self.matprop = matprop
        self.non_dim = (matprop.True_w1 / matprop.True_mat_E / matprop.True_l0)**(-0.5)
        if alpha_constraint == 'smooth':
            self.alpha_constraint = torch.sigmoid
        elif alpha_constraint =='nonsmooth':
            self.alpha_constraint = NonsmoothSigmoid(support=2.0, coeff=1e-3)
        else:
            raise('Transformer of phase-field variable is not be define!')
        
        Lx = domain_extrema[0,1] - domain_extrema[0,0]
        Ly = domain_extrema[1,1] - domain_extrema[1,0]
        L = torch.max(Lx,Ly)  
        self.L = L

        self.x0 = self.domain_extrema[0, 0] / L
        self.xL = self.domain_extrema[0, 1] / L
        self.y0 = self.domain_extrema[1, 0] / L
        self.yL = self.domain_extrema[1, 1] / L
        self.ReLU1 = ReLU1Funcrion() 

    def fieldCalculation(self, inp,world_pos,hist_alpha ):

        out = self.net(inp)
        out_disp = out[:, 0:2]
        
        alpha = self.ReLU1(self.alpha_constraint(out[:, 2]) +  hist_alpha)

        u = ((inp.node_features[:, 1]-self.y0)*(self.yL-inp.node_features[:, 1])*out_disp[:, 0] + \
             (inp.node_features[:, 1]-self.y0)/(self.yL-self.y0)*torch.cos(self.theta))*self.lmbda_delta + (world_pos[:,0] - inp.node_features[:,0]) * self.non_dim 
        v = ((inp.node_features[:, 1]-self.y0)*(self.yL-inp.node_features[:, 1])*out_disp[:, 1] + \
             (inp.node_features[:, 1]-self.y0)/(self.yL-self.y0)*torch.sin(self.theta))*self.lmbda_delta + (world_pos[:,1] - inp.node_features[:,1]) * self.non_dim 
        
        world_x_pos = u / self.non_dim + inp.node_features[:,0]
        world_y_pos = v / self.non_dim + inp.node_features[:,1]

        return u, v, alpha, world_x_pos,world_y_pos
    
    def update_hist_alpha(self, inp,world_pos,hist_alpha):
        _, _, pred_alpha,pred_world_x_pos,pred_world_y_pos = self.fieldCalculation(inp,world_pos,hist_alpha)   
        return torch.stack([pred_world_x_pos,pred_world_y_pos],dim=1).detach(), pred_alpha.detach(), 

class NonsmoothSigmoid(nn.Module):
    '''
    Constructs a continuous piecewise linear increasing function with the
    central part valid in (-support, support) and its value going from 0 to 1. 
    Outside this region, the slope equals coeff.

    '''
    def __init__(self, support=2.0, coeff=1e-3):
        super(NonsmoothSigmoid, self).__init__()
        self.support = support
        self.coeff =  coeff
    def forward(self, x):
        a = x>self.support
        b = x<-self.support
        c = torch.logical_not(torch.logical_or(a, b))
        out = a*(self.coeff*(x-self.support)+1.0)+ \
                b*(self.coeff*(x+self.support))+ \
                c*(x/2.0/self.support+0.5)
        return out
    
class ReLU1Funcrion(nn.Module):
    '''
    Constructs a continuous piecewise linear increasing function with the
    central part valid in (-support, support) and its value going from 0 to 1. 
    Outside this region, the slope equals coeff.

    '''
    def __init__(self, support=1.0, coeff=0):
        super(ReLU1Funcrion, self).__init__()

        self.support = support
        self.coeff =  coeff

    def forward(self, x):

        a = x>self.support
        b = x<0.0
        c = torch.logical_not(torch.logical_or(a, b))

        out = a*(self.coeff*(x-self.support)+1.0)+ b*self.coeff*x+ c*x

        return out
   