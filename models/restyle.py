import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import restyle_e4e_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        self.n_styles = 18
        # Define architecture
        self.encoder = self.set_encoder().eval()
        # self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        encoder = restyle_e4e_encoders.ProgressiveBackboneEncoder(50, 'ir_se', self.n_styles, self.opts)
        return encoder

    def load_weights(self):
        print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.pretrained_e4e_path))
        ckpt = torch.load(self.opts.pretrained_e4e_path, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        # self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        self.__load_latent_avg(ckpt)
            
    def forward(self, x, latent=None, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        return codes

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
