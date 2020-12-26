"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, l_net, g_net, done, state, buffer_s, buffer_a, buffer_r, gamma):
    if done:
        v_s_ = torch.tensor(0.)               # terminal
    else:
        v_s_ = l_net.forward(state)[-1]

    buffer_v_target = [v_s_]
    for r in buffer_r[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = l_net.loss_func(
        list2tensor(buffer_s),
        list2tensor(buffer_a),
        list2tensor(buffer_v_target))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(l_net.parameters(), g_net.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    l_net.load_state_dict(g_net.state_dict())


def list2tensor(inputs):
    for i in range(len(inputs)):
        inputs[i] = inputs[i].unsqueeze(0)
    inputs = torch.cat(inputs, dim=0)
    return inputs


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep: %4d | Ep_r: %.2f" % (global_ep.value, global_ep_r.value)
    )
