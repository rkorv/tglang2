import numpy as np


def convert_Conv1D(torch_layer, tf_layer):
    tf_layer.weights[0].assign(
        np.transpose(torch_layer.weight.detach().numpy(), (2, 1, 0))
    )
    tf_layer.bias.assign(torch_layer.bias.detach().numpy())


def convert_Linear(torch_layer, tf_layer):
    tf_layer.weights[0].assign(torch_layer.weight.detach().numpy().T)
    if torch_layer.bias is not None:
        tf_layer.weights[1].assign(torch_layer.bias.detach().numpy())


def convert_LayerNorm(torch_ln, tf_ln):
    tf_ln.gamma.assign(torch_ln.weight.detach().numpy())
    tf_ln.beta.assign(torch_ln.bias.detach().numpy())


def convert_BatchNorm(torch_bn, tf_bn):
    tf_bn.gamma.assign(torch_bn.weight.detach().numpy())
    tf_bn.beta.assign(torch_bn.bias.detach().numpy())
    tf_bn.moving_mean.assign(torch_bn.running_mean.detach().numpy())
    tf_bn.moving_variance.assign(torch_bn.running_var.detach().numpy())


def convert_Embedding(torch_embs, tf_embs):
    torch_weights = torch_embs.weight.data.cpu().numpy()
    tf_embs.assign(torch_weights)
