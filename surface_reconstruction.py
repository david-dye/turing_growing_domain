import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
from tqdm import tqdm
import h5py
import os
import imageio.v2 as imageio
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable

# def list_datasets(h5file):
#     dataset_names = []
#     def visitor(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             dataset_names.append(name)
#     h5file.visititems(visitor)
#     return dataset_names

# with h5py.File("C:\\Users\\dwdjr\\Downloads\\rd_data_full_new.h5", 'r') as f:
# 	lam_data = f['lam'][::1000]
# 	c1_data = f['c1'][::1000]
# 	c2_data = f['c2'][::1000]
# 	np.save('lam_data.npy', lam_data)
# 	np.save('c1_data.npy', c1_data)
# 	np.save('c2_data.npy', c2_data)

lam_data = torch.from_numpy(np.load('lam_data.npy'))
c1_data = torch.from_numpy(np.load('c1_data.npy'))
c2_data = torch.from_numpy(np.load('c2_data.npy'))
lam_train = lam_data.reshape(len(lam_data), -1)
c_dif_data = (c1_data - c2_data).reshape(len(lam_data), -1)

# os.makedirs('frames_cdif', exist_ok=True)

# # Generate and save frames
# for i, frame in enumerate(c1_data - c2_data):
#     plt.imshow(frame, cmap='viridis', interpolation='nearest')
#     plt.axis('off')  # Hide axes
#     plt.tight_layout()
#     plt.savefig(f'frames_cdif/frame_{i:03d}.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Create GIF
# with imageio.get_writer('cdif_animation.gif', mode='I', duration=0.05) as writer:
#     for i in range(len(lam_data)):
#         image = imageio.imread(f'frames_cdif/frame_{i:03d}.png')
#         writer.append_data(image)

# print("GIF saved as lam_animation.gif")

# raise Exception

# Physical Parameters
a, b = 0.2, 1.3
d = 0.1
mu = 0.5
k1, k2 = 0, 0
gmin, gmax = 0.1, 10.0
Lx, Ly = 10.0, 10.0

# Saving Parameters
Tfinal  = 50.0
Nx, Ny  = 128, 128
dx, dy  = Lx/Nx, Ly/Ny
dt      = 1e-5 * (128/Nx)**2
n_steps = int(Tfinal/dt) + 1
save_interval = 10
plot_interval = 10000

alpha_min = 0
alpha_max = 10
beta_min = 0
beta_max = 10
alphas = torch.linspace(alpha_min, alpha_max, Nx)
betas = torch.linspace(beta_min, beta_max, Ny)
alpha_coords, beta_coords = torch.meshgrid(alphas, betas, indexing='ij')

all_coords = torch.stack((alpha_coords, beta_coords), dim=-1)  # shape: (Ny+1, Nx+1, 2)
all_coords = all_coords.reshape(-1, 2)  # shape: ((Nx)*(Ny), 2)

model_vis_inputs = torch.stack([alpha_coords.flatten(), beta_coords.flatten()], dim=-1)
P = torch.eye(2)
g_min = gmin
g_max = gmax
lambda_0 = (g_min + g_max)/2



def get_lam(t, idcs):
	'''
	Returns lambda(alpha, beta, t) for each (alpha, beta) pair in locs.
	'''
	# return 1 + 0.2 * torch.sin(locs[:, 0] * locs[:, 1] * torch.sin(t))
	return lam_train[t, idcs]



def compute_jacobian(u, x):
    '''
    Compute the Jacobian du/dx for a batch of samples.
    Inputs:
        u: Tensor of shape [N, 3]
        x: Tensor of shape [N, 2], requires_grad=True
    Returns:
        J: Tensor of shape [N, 3, 2]
    '''
    N, out_dim = u.shape
    J = []

    for i in range(out_dim):
        grads = torch.autograd.grad(
            outputs=u[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(u[:, i]),
            retain_graph=True,
            create_graph=True
        )[0]  # shape: [N, 2]
        J.append(grads)

    return torch.stack(J, dim=1)  # shape: [N, 3, 2]


class SurfaceMLP(nn.Module):
	def __init__(self, t:float, prev_model:"SurfaceMLP" = None):
		super(SurfaceMLP, self).__init__()
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 3)
		self.act1 = nn.SiLU()
		self.act2 = nn.SiLU()
		self.act3 = nn.SiLU()
		self.t = t
		self.prev_model = prev_model
		self.prev_model.prev_model = None

		if prev_model:
			with torch.no_grad():
				self.fc1.weight.copy_(prev_model.fc1.weight)
				self.fc1.bias.copy_(prev_model.fc1.bias)

				self.fc2.weight.copy_(prev_model.fc2.weight)
				self.fc2.bias.copy_(prev_model.fc2.bias)

				self.fc3.weight.copy_(prev_model.fc3.weight)
				self.fc3.bias.copy_(prev_model.fc3.bias)

				self.fc4.weight.copy_(prev_model.fc4.weight)
				self.fc4.bias.copy_(prev_model.fc4.bias)

			self.prev_model.eval()
			for param in self.prev_model.parameters():
				param.requires_grad = False

		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		# self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
		# self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)

	def forward(self, x:torch.Tensor):
		x = self.act1(self.fc1(x))
		x = self.act2(self.fc2(x))
		x = self.act3(self.fc3(x))
		output = self.fc4(x)  # No activation on the output layer
		return output

	def loss(self, x:torch.Tensor, J:torch.Tensor, u:torch.Tensor, idcs, gamma:float=1):
		'''
		Compute the loss evaluated at each x with corresponding Jacobian J and self model output u.

		shape(x) = N x 2, (alpha, beta)
		shape(J) = N x 3 x 2, (dx_i/dalpha, dx_i/dbeta)
		shape(u) = N x 3
		'''
		G_hat = torch.matmul(J.transpose(1, 2), J)
		
		if self.t == 0:
			#the original surface
			with torch.no_grad():
				#construct a plane stretched out by a factor of sqrt(lambda_0) to match the initial metric.
				u_ref = torch.cat((torch.sqrt(torch.tensor(lambda_0)) * x, torch.zeros(len(x), 1)), 1)

			G_term = torch.sum((G_hat[:, 0, 0] - lambda_0)**2 + (G_hat[:, 1, 1] - lambda_0)**2 + 2 * (G_hat[:, 0, 1])**2)
			psi_term = torch.sum((u - u_ref)**2)

			return G_term + gamma * psi_term

		
		#otherwise, use prev_model
		with torch.no_grad():
			lams = get_lam(self.t, idcs)
			u_ref = self.prev_model(x)

		G_term = torch.sum((G_hat[:, 0, 0] - lams)**2 + (G_hat[:, 1, 1] - lams)**2 + 2 * (G_hat[:, 0, 1])**2)
		psi_term = torch.sum((u - u_ref)**2)

		return G_term + gamma * psi_term


	def train_model(self, epochs:int, batch_size:int, gamma:float=1, device:str='cpu', progress_bar=False):
		self.to(device)
		losses = []
		iterator = tqdm(range(epochs)) if progress_bar else range(epochs)
		
		all_x = all_coords.to(device)

		for epoch in iterator:
			self.train()

			# Sample input batch in the given domain
			idcs = sample_batch(batch_size, device)
			# idcs = np.arange(0, lam_train.shape[1])
			x = all_x[idcs]
			x.requires_grad_(True)

			# Forward pass
			u = self(x)
			J = compute_jacobian(u, x)

			loss = self.loss(x, J, u, idcs, gamma)

			# Backpropagation
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			# self.scheduler.step(loss)
			losses.append(loss.item())
		
		#clean up
		self.eval()
		return losses


def main():
	epochs_t0 = 5000
	epochs = 500
	batch_size = 256
	gamma = 0.1
	device = 'cpu'

	models = []

	# model = SurfaceMLP(0, None)
	# losses = model.train_model(epochs_t0, batch_size, gamma, device, progress_bar=True)
	# torch.save(model, 'sgd_model_0.pth')
	model = torch.load('sgd_model_0.pth')
	models.append(model)
	# plt.plot(losses)
	# plt.yscale('log')
	# plt.show()

	# fig = plt.figure(figsize=(8, 6))
	# ax = fig.add_subplot(111, projection='3d')
	# ax.set_xlabel('$u_1$')
	# ax.set_ylabel('$u_2$')
	# ax.set_zlabel('$u_3$')
	# ax.set_xlim([10 * (alpha_min - (alpha_min + alpha_max)/2), 10 * (alpha_max - (alpha_min + alpha_max)/2)])
	# ax.set_ylim([10 * (beta_min - (beta_min + beta_max)/2), 10 * (beta_max - (beta_min + beta_max)/2)])
	# ax.set_zlim([-10, 10])
	# cmap = plt.get_cmap('plasma')

	# with torch.no_grad():
	# 	outputs = model(model_vis_inputs)

	# print(outputs.shape)

	# u = outputs.cpu().numpy()
	# x, y, z = u[:, 0], u[:, 1], u[:, 2]
	# ax.plot_trisurf(x, y, z, cmap=cmap, linewidth=0)
	# ax.set_title(f"Reconstructed Surface, $t = {model.t}$")
	# ax.set_axis_off()
	# fig.tight_layout()
	# plt.show()

	# raise Exception

	
	#===================
	# Training iterative models
	#===================
	
	# ts = torch.linspace(0, 2 * torch.pi, 100) #must start at 0
	ts = torch.arange(0, len(lam_train))
	for t in tqdm(ts[1:]):
		# model = SurfaceMLP(t, models[-1])
		# losses = model.train_model(epochs, batch_size, gamma, device)	
		# torch.save(model, f'sgd_model_{t}.pth')
		model = torch.load(f'sgd_model_{t}.pth')
		models.append(model)
		
		# plt.plot(losses)
		# plt.yscale('log')
		# plt.show()

	#===================
	# Visualization
	#===================

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('$u_1$')
	ax.set_ylabel('$u_2$')
	ax.set_zlabel('$u_3$')
	ax.set_xlim([alpha_min, alpha_max])
	ax.set_ylim([beta_min, beta_max])
	ax.set_zlim([-1.5, 1.5])
	cmap = plt.get_cmap('plasma')

	min_idx = 120
	max_idx = 210
	c_dif = c_dif_data
	models = models[min_idx:max_idx]
	
	# min_cdif = c_dif.min().item()
	# max_cdif = c_dif.max().item()
	# abs_max = np.max([np.abs(min_cdif), np.abs(max_cdif)])
	# norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
	norm = plt.Normalize(vmin=0, vmax=1)
	sm = ScalarMappable(norm=norm, cmap=cmap)
	cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0)
	cbar.ax.set_xlabel('$c_1 - c_2$', rotation=0, loc='center')
	cbar.ax.xaxis.set_label_position('top')

	def update(model):
		ax.clear()
		with torch.no_grad():
			outputs = model(model_vis_inputs)

		u = outputs.cpu().numpy()
		x, y = u[:, 0], u[:, 1]

		c_dif_frame = c_dif[model.t].numpy()

		eps = 1
		tri = mtri.Triangulation(x, y)
		c_avgs = np.mean(c_dif_frame[tri.triangles], axis=1)
		
		norm.vmin = c_avgs.min().item()
		norm.vmax = c_avgs.max().item()
		cbar.update_normal(sm)

		colors = cmap(norm(c_avgs))
		tris = u[tri.triangles]

		# ax.plot_trisurf(x, y, z, linewidth=0, triangles=tri.triangles, facecolors=colors, antialiased=False)
		poly = Poly3DCollection(tris, facecolors=colors, linewidths=0, edgecolors=None, zsort='min', shade=True)
		poly.set_antialiased(False)
		ax.add_collection3d(poly)
		ax.set_title(f"Reconstructed Surface, $t$ = {(model.t * 1000 * dt):0.3f}")
		ax.set_xlim([alpha_min - eps, alpha_max * 2.5 + eps])
		ax.set_ylim([beta_min - eps, beta_max * 2.5 + eps])
		ax.set_zlim([-4, 4])
		ax.set_xlabel('$u_1$')
		ax.set_ylabel('$u_2$')
		ax.set_zlabel('$u_3$')
		ax.set_axis_off()
		fig.tight_layout()

	ani = FuncAnimation(fig, update, frames=tqdm(models), interval=100)

	# Save as GIF
	ani.save('animated_surface.gif', writer='pillow', fps=10)
	plt.close()



def sample_batch(N, device='cpu'):
	'''
	Sample N values in the domain [alpha_min, alpha_max] \\times [beta_min, beta_max]
	'''
	idcs = torch.randint(0, lam_train.shape[1], (N,), device=device)
	return idcs




if __name__ == '__main__':
	main()