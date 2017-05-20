import argparse
from train import train

def main(args):
    train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--k_init', type=float, default=0.001,
                        help='k_init value')
    parser.add_argument('--mb_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='latent variable size')
    parser.add_argument('--scale', type=int, default=64,
                        help='input image scale')
    parser.add_argument('--h_dim', type=int, default=64,
                        help='first hidden channel size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learing rate')
    parser.add_argument('--lam', type=float, default=1e-3,
                        help='lambda for k update')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='gamma')
    parser.add_argument('--epoch_iter', type=int, default=500,
                        help='max epoch')
    parser.add_argument('--normalization', type=bool, default=True,
                        help='using input images nomoralization')
    parser.add_argument('--lr_scheduler', type=bool, default=True,
                        help='using lr_scheduler')

    # other parameters
    parser.add_argument('--model_name', type=str, default="V2_2",
                        help='this model name for save pickle, logs, output image path and if model_name contain V2 modelV2 excute')
    parser.add_argument('--use_tensorboard', type=bool, default=True,
                        help='using tensorboard logging')

    # dir parameters
    parser.add_argument('--output_dir', type=str, default="../output",
                        help='output path')
    parser.add_argument('--input_dir', type=str, default='../input',
                        help='input path')
    parser.add_argument('--pickle_dir', type=str, default='/pickle/',
                        help='input path')
    parser.add_argument('--log_dir', type=str, default='/log/',
                        help='for tensorboard log path save in output_dir + log_dir')
    parser.add_argument('--image_dir', type=str, default='/image/',
                        help='for output image path save in output_dir + image_dir')
    parser.add_argument('--data_dir', type=str, default='/new_data2/',
                        help='dataset load path')

    # step parameter
    parser.add_argument('--pickle_step', type=int, default=50,
                        help='pickle save at pickle_step epoch')
    parser.add_argument('--log_step', type=int, default=1,
                        help='tensorboard log save at log_step epoch')
    parser.add_argument('--image_save_step', type=int, default=1,
                        help='output image save at image_save_step epoch')

    args = parser.parse_args()
    print(args)
    main(args)
