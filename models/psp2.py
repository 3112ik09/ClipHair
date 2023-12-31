import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import psp_encoders_psp_features2
from models.stylegan2.model import Generator
# from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		# self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder().eval()
		# self.decoder = Generator(self.opts.output_size, 512, 8)
		# self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		encoder = psp_encoders_psp_features2.GradualStyleEncoder(50, 'ir_se', self.opts)
		return encoder

	def load_weights(self):
		print('Loading pSp over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
		ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
		self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
		# self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
		self.__load_latent_avg(ckpt)
        

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes, features = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if codes.ndim == 2:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
				else:
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

		return codes, features
# 		input_is_latent = not input_code
# 		images, result_latent = self.decoder([codes],
# 		                                     input_is_latent=input_is_latent,
# 		                                     randomize_noise=randomize_noise,
# 		                                     return_latents=return_latents)

# 		if resize:
# 			images = self.face_pool(images)

# 		if return_latents:
# 			return images, result_latent
# 		else:
# 			return images

	def set_opts(self, opts):
		self.opts = opts
        
	def forward_features(self, features):
		return self.encoder.forward_features(features)
        
	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
