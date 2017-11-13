import argparse
from train import train

def main(args):
    train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters

    parser.add_argument('--mb_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='latent variable size')
    parser.add_argument('--image_scale', type=int, default=64,
                        help='input image scale')
    parser.add_argument('--image_channel', type=int, default=1,
                        help='input image channel')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learing rate')

    parser.add_argument('--epoch_iter', type=int, default=1000,
                        help='max epoch')
    parser.add_argument('--normalization', type=bool, default=False,
                        help='using input images nomoralization')

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
    parser.add_argument('--data_dir', type=str, default='/new_data/',
                        help='dataset load path')

    # step parameter
    parser.add_argument('--pickle_step', type=int, default=50,
                        help='pickle save at pickle_step epoch')
    parser.add_argument('--log_step', type=int, default=10,
                        help='tensorboard log save at log_step epoch')
    parser.add_argument('--image_save_step', type=int, default=100,
                        help='output image save at image_save_step epoch')

    args = parser.parse_args()
    print(args)
    main(args)
