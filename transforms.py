

class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()

        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        # gs[1].copy_(gs[0])
        # gs[2].copy_(gs[0])

        return gs[0].resize_(1, gs[0].size()[0], gs[0].size()[1])

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std, input_nomalization):
        self.mean = mean
        self.std = std
        self.input_nomalization = input_nomalization

    def __call__(self, tensor):
        # TODO: make efficient
        if self.input_nomalization == True:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor

        else:
            return tensor