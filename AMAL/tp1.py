#%%
# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        res = (yhat - y)
        return torch.sum(res**2)/len(yhat)

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées

        yhat, y = ctx.saved_tensors

        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        return grad_output * 2*(yhat - y), grad_output * 2*(y - yhat)

mse = MSE.apply

# Inherit from Function
class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, torch.transpose(weight, 0, 1)) #On mutliplie les tenseurs x et W
        output += bias.unsqueeze(0).expand_as(output)
        #On y ajoute b, en lui donnant les bonnes profondeur et dimension
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #Ils avaient été sauvegardés pour le backward
        return torch.matmul(grad_output, weight), torch.matmul(grad_output, input), grad_output

linear = LinearFunction.apply
