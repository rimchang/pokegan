import argparse
import torch.nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from collections import OrderedDict
import os
import torch

from model.vae_deconv import Vae_deconv
# my module
from transforms import Grayscale, Normalize
from utils import make_hyparam_string, read_pickle, save_new_pickle, to_var, to_tensor



def train(args):
    print(args)

    ########################## tensorboard ##############################

    hyparam_list = [("z_dim", args.z_dim),
                    ("lr", args.lr),
                    ("nm", args.normalization)]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    log_param = args.model_name + "_" + log_param
    print(log_param)

    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    ########################## data loading ###############################

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(args.image_scale),
            transforms.ToTensor(),
            Grayscale(),
            #Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], args.normalization),

        ])
    }

    dsets_path = args.input_dir + args.data_dir
    dsets = {x: datasets.ImageFolder(os.path.join(dsets_path, x), data_transforms[x])
             for x in ['train']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.mb_size,
                                                   shuffle=True, num_workers=4, drop_last=True)
                    for x in ['train']}

    model = Vae_deconv(args)

    if torch.cuda.is_available():
        model.cuda()

    solver = optim.Adam(model.parameters(), lr=args.lr)

    solver.param_groups[0]['epoch'] = 0
    solver.param_groups[0]['iter'] = 0

    #criterion = torch.nn.BCELoss(size_average=False)
    criterion = torch.nn.MSELoss(size_average=False)

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, model, solver)

    # Training
    for epoch in range(0, args.epoch_iter):

        solver.param_groups[0]['epoch'] += 1
        epoch = solver.param_groups[0]['epoch']

        for img, label in dset_loaders['train']:
            # Sample data
            solver.param_groups[0]['iter'] += 1
            iteration = solver.param_groups[0]['iter']


            img = to_var(img)

            recon, mu, log_var = model(img)

            recon_loss = criterion(recon, img) #/ img.size(0)
            kld_loss = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

            ELBO = recon_loss + kld_loss
            solver.zero_grad()
            ELBO.backward()
            solver.step()

            print('Iter-{}; recon_loss: {:.4} , kld_loss : {:.4}, ELBO  : {:.4}'
                  .format(str(iteration), recon_loss.data[0], kld_loss.data[0], ELBO.data[0]))

            if (iteration) % args.image_save_step == 0:

                samples = recon.cpu().data[:16]

                image_path = args.output_dir + args.image_dir + log_param
                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                torchvision.utils.save_image(samples,
                                             image_path + '/{}.png'.format(str(iteration).zfill(3)),
                                             normalize=True)

            if args.use_tensorboard and (iteration) % args.log_step == 0:

                log_save_path = args.output_dir + args.log_dir + log_param
                if not os.path.exists(log_save_path):
                    os.makedirs(log_save_path)


                info = {
                    'loss/loss_D_R': recon_loss.data[0],
                    'loss/loss_D': kld_loss.data[0],
                    'loss/loss_G': ELBO.data[0],
                }

                for tag, value in info.items():
                    inject_summary(summary_writer, tag, value, iteration)

                summary_writer.flush()


        if (epoch) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, iteration, model, solver)

