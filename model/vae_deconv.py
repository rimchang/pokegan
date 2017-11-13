import torch
from torch import nn
import math
# Keras autoencoder_deconv example
# https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py#L93


class Vae_deconv(nn.Module):
    def __init__(self, args):
        super(Vae_deconv, self).__init__()
        self.args = args
        self.encoder = self._make_encoder()
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * self.args.image_scale//2 * self.args.image_scale//2, self.args.z_dim * 2),
            nn.ReLU()
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.args.z_dim, 64 * self.args.image_scale//2 * self.args.image_scale//2),
            nn.ReLU()
        )
        self.decoder = self._make_decoder()

        """
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
          elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        """

    def _make_encoder(self):
        encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.args.image_channel, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        encoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        encoder_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        encoder = nn.Sequential(encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4)

        return encoder

    def _make_decoder(self):
        decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        decoder_mean_squash = nn.Sequential(
            nn.Conv2d(64, self.args.image_channel, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.Sigmoid()
        )

        decoder = nn.Sequential(decoder_deconv1, decoder_deconv2, decoder_deconv3, decoder_mean_squash)

        return decoder

    def sample_z(self, mu, log_var):
        eps = torch.autograd.Variable(torch.randn(self.args.mb_size, self.args.z_dim))
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, image):
        out = self.encoder(image)  # 32, 32
        #print(out.size())
        out = out.view(-1, 64 * self.args.image_scale//2 * self.args.image_scale//2)
        out = self.encoder_fc(out)

        mu, log_var = torch.chunk(out, 2, dim=1)
        #print(mu, log_var)
        out = self.sample_z(mu, log_var)
        #print(out.size())
        out = self.decoder_fc(out)
        out = out.view(-1, 64, self.args.image_scale//2, self.args.image_scale//2)
        out = self.decoder(out)  # 64, 64

        #print(out.size())
        return out, mu, log_var



