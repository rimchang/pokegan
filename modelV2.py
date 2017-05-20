import torch

########################## model ###############################

class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.linear = torch.nn.Linear(self.args.z_dim, self.args.h_dim*8*8)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.Conv2d(self.args.h_dim, 1, 3, 1, 1),
            torch.nn.ELU())

    def forward(self, x):
        out = self.linear(x)
        #print(out.size())
        out = out.view([-1, self.args.h_dim, 8, 8])
        #print(out.size())
        out = self.layer1(out)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = self.layer3(out)
        #print(out.size())
        out = self.layer4(out)
        #print(out.size())
        return out


class _D(torch.nn.Module):

    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        
        # Encoder
        self.encoderlayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU())

        self.encoderlayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU())

        self.encoderlayer3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 2, 1),
            torch.nn.ELU())

        self.encoderlayer4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 2, 1),
            torch.nn.ELU())

        self.encoderlayer5 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 2, 1),
            torch.nn.ELU())

        self.encoderlinear = torch.nn.Linear(self.args.h_dim * 8 * 8, self.args.z_dim)

        # Decoder
        self.decoderlinear = torch.nn.Linear(self.args.z_dim, self.args.h_dim * 8 * 8)
        self.decoderlayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2))

        self.decoderlayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2))

        self.decoderlayer3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2))

        self.decoderlayer4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.h_dim, self.args.h_dim, 3, 1, 1),
            torch.nn.ELU(),
            torch.nn.Conv2d(self.args.h_dim, 1, 3, 1, 1))

    def forward(self, x):
        # encoder
        out = self.encoderlayer1(x)
        #print(out.size())
        out = self.encoderlayer2(out)
        #print(out.size())
        out = self.encoderlayer3(out)
        #print(out.size())
        out = self.encoderlayer4(out)
        #print(out.size())
        out = self.encoderlayer5(out)
        #print(out.size())
        out = out.view([-1, self.args.h_dim * 8 * 8])
        #print(out.size())
        out = self.encoderlinear(out)
        #print(out.size())

        # dercoder
        out = self.decoderlinear(out)
        #print(out.size())
        out = out.view([-1, self.args.h_dim, 8, 8])
        #print(out.size())
        out = self.decoderlayer1(out)
        #print(out.size())
        out = self.decoderlayer2(out)
        #print(out.size())
        out = self.decoderlayer3(out)
        #print(out.size())
        out = self.decoderlayer4(out)
        #print(out.size())
        return out