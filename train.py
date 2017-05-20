import argparse
import torch.nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from collections import OrderedDict
import os
import torch


# my module
from transforms import Grayscale, Normalize
from utils import make_hyparam_string, read_pickle, save_new_pickle
from lr_scheduler import ReduceLROnPlateau


def train(args):
    print(args)
    ########################## tensorboard ###############################
    state = {'k': args.k_init}

    hyparam_list = [("h_dim", args.h_dim),
                    ("lr", args.lr),
                    ("lam", args.lam),
                    ("k", args.k_init),
                    ("gamma", args.gamma),
                    ("nm", args.normalization),
                    ("lrs", args.lr_scheduler)]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    log_param = args.model_name + "_" + log_param
    print(log_param)
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
            if hasattr(value, '__len__'):
                pass
                # for idx, img in enumerate(value):
                #     summary = tf.Summary()
                #     sio = StringIO.StringIO()
                #     scipy.misc.toimage(img).save(sio, format="png")
                #     image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
                #     summary.value.add(tag="{}/{}".format(tag, idx), image=image_summary)
                #     summary_writer.add_summary(summary, global_step=step)
            else:
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    ########################## data loading ###############################

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(args.scale),
            transforms.ToTensor(),
            Grayscale(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], args.normalization),

        ])
    }

    dsets_path = args.input_dir + args.data_dir
    dsets = {x: datasets.ImageFolder(os.path.join(dsets_path, x), data_transforms[x])
             for x in ['train']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.mb_size,
                                                   shuffle=True, num_workers=4)
                    for x in ['train']}
    dset_sizes = {x: int(len(dsets[x])) for x in ['train']}
    dset_classes = dsets['train'].classes

    if args.model_name.upper().count("V2") == 1:
        from modelV2 import _D, _G

        G = _G(args)
        G.cuda()
        D_ = _D(args)
        D_.cuda()

    else:
        from model import _D, _G

        G = _G(args)
        G.cuda()
        D_ = _D(args)
        D_.cuda()

    # D is an autoencoder, approximating Gaussian
    def D(X):
        X = X.view([-1, 1, 64, 64])
        X_recon = D_(X)
        # Use Laplace MLE as in the paper
        return torch.mean(torch.sum(torch.abs(X - X_recon), 1))

    def reset_grad():
        G.zero_grad()
        D_.zero_grad()

    G_solver = optim.Adam(G.parameters(), lr=args.lr)
    D_solver = optim.Adam(D_.parameters(), lr=args.lr)

    if args.lr_scheduler:
        G_scheduler = ReduceLROnPlateau(G_solver, 'min', patience=20)
        D_scheduler = ReduceLROnPlateau(D_solver, 'min', patience=30)

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D_, D_solver, state)
    k = state['k']

    print("load complete k : ", k, state)
    # Training
    for epoch in range(0, args.epoch_iter):
        for i, (X, _) in enumerate(dset_loaders['train']):
            # Sample data
            X = X.view(X.size(0), -1)
            X = Variable(X).cuda()

            # Dicriminator
            z_D = Variable(torch.randn(args.mb_size, args.z_dim)).cuda()

            D_loss_R = D(X)
            D_loss = D_loss_R - k * D(G(z_D))

            D_loss.backward()
            D_solver.step()
            reset_grad()

            # Generator
            z_G = Variable(torch.randn(args.mb_size, args.z_dim)).cuda()

            G_loss = D(G(z_G))

            G_loss.backward()
            G_solver.step()
            reset_grad()

            # Update k, the equlibrium
            k = k + args.lam * (args.gamma * D(X) - D(G(z_G)))
            k = k.data[0]  # k is variable, so unvariable it so that no gradient prop.

            if k < 0:
                k = 0.0
            elif k > 1:
                k = 1.0


        # global measure of convergence
        measure = D(X) + torch.abs(args.gamma * D(X) - D(G(z_G)))
        iteration = G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step']
        #measure.data[0] = 0
        if args.lr_scheduler:
            print(
                'Iter-{}; Convergence measure: {:.4} , k : {:.4}, D_loss_R : {:.4}, D_loss : {:.4}, G_loss : {:.4}, D_lr : {:.4}, G_lr : {:.4}'
                .format(str(iteration), measure.data[0], k, D_loss_R.data[0], D_loss.data[0], G_loss.data[0],
                        D_solver.state_dict()['param_groups'][0]["lr"], G_solver.state_dict()['param_groups'][0]["lr"]))
        else:
            print('Iter-{}; Convergence measure: {:.4} , k : {:.4}, D_loss_R : {:.4}, D_loss : {:.4}, G_loss : {:.4}'
                  .format(str(iteration), measure.data[0], k, D_loss_R.data[0], D_loss.data[0], G_loss.data[0]))

        if args.lr_scheduler:
            try:

                D_scheduler.step(D_loss_R.data[0])
                G_scheduler.step(measure.data[0])

            except Exception as e:

                print("fail lr scheduling", e)

        if (epoch + 1) % args.image_save_step == 0:

            samples = G(z_G).cpu().data[:16]

            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            torchvision.utils.save_image(samples,
                                         image_path + '/{}.png'.format(str(iteration).zfill(3)),
                                         normalize=True)

        if (epoch + 1) % args.pickle_step == 0:
            state['k'] = k
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D_, D_solver, state)

        if args.use_tensorboard and (epoch + 1) % args.log_step == 0:

            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)


            info = {
                'loss/loss_D_R': D_loss_R.data[0],
                'loss/loss_D': D_loss.data[0],
                'loss/loss_G': G_loss.data[0],
                'misc/measure': measure.data[0],
                'misc/k_t': k,
                'lr/D_lr' : D_solver.state_dict()['param_groups'][0]["lr"],
                'lr/D_lr': G_solver.state_dict()['param_groups'][0]["lr"],
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, iteration)

            summary_writer.flush()

