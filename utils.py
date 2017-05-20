import os
import torch
import pickle

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver, state):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))

        with open(path + "/state_" + recent_iter + ".pkl", "rb") as f:
            load_state = pickle.load(f)
            state['k'] = load_state['k']
        print(state, "in uitls")

    except Exception as e:

        print("fail try", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, state):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)

    with open(path + "/state_" + str(iteration) + ".pkl", "wb") as f:
        pickle.dump(state, f)

########################## path helper ###############################

def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + "_" + str(i) + "=" + str(hyparam_dict[i])
    return str_result