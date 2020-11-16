import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, LinearFunction, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3, requires_grad=True)

# Les paramètres du modèle à optimiser
w = torch.randn(3, 13, requires_grad=True)
b = torch.randn(3, requires_grad=True)

epsilon = 0.05

ctx1 = Context()
ctx2 = Context()
writer = []
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    loss = MSE.forward(ctx1, LinearFunction.forward(ctx2, x, w, bias=b), y).mean()
    print(loss)
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.append(['Loss/train', loss, n_iter])


    loss.backward()

    with torch.no_grad():
        w -= epsilon * w.grad
        b -= epsilon * b.grad
        w.grad.zero_()
        b.grad.zero_()
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
print(writer)
    ##  TODO:  Mise à jour des paramètres du modèle
