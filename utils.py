import os
import torch
import pickle
from torch.autograd import Variable
import numpy as np

def to_var(x, *args, **kwargs):

    if torch.cuda.is_available():
        x = Variable(x, *args, **kwargs).cuda()
    else:
        x = Variable(x, *args, **kwargs)

    return x

def to_tensor(x, *args, **kwargs):

    if torch.cuda.is_available():
        x = torch.from_numpy(x).cuda()
    else:
        x = torch.from_numpy(x)
    return x

########################## Pickle helper ###############################


def read_pickle(path, model, solver):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/model_" + recent_iter + ".pkl", "rb") as f:
            model.load_state_dict(torch.load(f))
        with open(path + "/solver_" + recent_iter + ".pkl", "rb") as f:
            solver.load_state_dict(torch.load(f))

    except Exception as e:

        print("fail read pickle", e)



def save_new_pickle(path, iteration, model, solver):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/model_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(model.state_dict(), f)
    with open(path + "/solver_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(solver.state_dict(), f)


########################## path helper ###############################

def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + "_" + str(i) + "=" + str(hyparam_dict[i])
    return str_result