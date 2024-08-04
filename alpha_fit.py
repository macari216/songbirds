import numpy as np
import pynapple as nap
import torch
import torch.optim as optim
import scipy.io as sio
from itertools import product
from time import perf_counter
import matplotlib.pyplot as plt
import nemos as nmo


class Alpha:
    def __init__(self, x, target, lr=1e-3, cond=0.0075, max_iteration=200000):

        self.x = x
        self.target = target
        self.w = self.init_w()
        self.tau = torch.tensor([0.001], requires_grad=True)
        self.d = torch.tensor([0.001], requires_grad=True)
        self.optimizer = optim.Adam([self.w, self.tau, self.d], lr=lr)
        self.losses = []
        self.cond = cond
        self.max_iter = max_iteration

    def alpha_function(self, params=None):
        if params==None:
            return self.w * ((self.x - self.d)/self.tau) * torch.exp(-(self.x - self.d)/self.tau)
        else:
            w = params[0]
            tau = params[1]
            d = params[2]

            return w * ((x - d) / tau) * torch.exp(-(x - d) / tau)

    def init_w(self):
        peak = self.target.max()
        w = torch.tensor([peak], requires_grad=True)
        return w

    def loss_mse(self, alpha):
        alpha[alpha < 0] = 0
        alpha = alpha.float()
        target = self.target.float()
        return torch.mean(torch.square(alpha - target))

    def nan_to_zero(self):
        if self.w == 'nan':
            self.w = torch.tensor([0.0])
        if self.tau == 'nan':
            self.tau = torch.tensor([0.0])
        if self.d == 'nan':
            self.d = torch.tensor([0.0])

    def run(self):
        loss = 1
        iteration = 0

        while loss > self.cond:
            self.optimizer.zero_grad()

            alpha = self.alpha_function()

            loss = self.loss_mse(alpha)

            if iteration % 100 == 0:
                self.losses.append(loss.detach().numpy())

            loss.backward()
            self.optimizer.step()

            iteration += 1

            if iteration > self.max_iter:
                break

        self.nan_to_zero()

        return [self.w, self.tau, self.d], self.losses


def compute_filters(coef, n_neurons=195, n_fun=9, binsize=0.0001, hist_window_sec=0.004):
    hist_window_size = int(hist_window_sec / binsize)
    basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
    time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
    time *= hist_window_sec
    weights = coef.reshape(n_neurons, n_fun, n_neurons)
    filters = np.einsum("jki,tk->ijt", weights, basis_kernels)

    return filters, time


spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds_data/c57EI.mat')['c57EI']

results_dict = np.load("/Users/macari216/Desktop/glm_songbirds/songbird_output/mp_results_all.npy", allow_pickle=True).item()
coef = results_dict["params"][0]

filters, time = compute_filters(coef)

# ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
# spike_times_quiet = nap.TsGroup(ts_dict_quiet)
#
# spike_times_quiet["EI"] = ei_labels
#
# spike_times_sorted = nap.TsGroup.merge_group(spike_times_quiet.getby_category("EI")[1],
#                                              spike_times_quiet.getby_category("EI")[-1],
#                                              reset_index=True)

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(10)}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

spike_times_quiet["EI"] = ei_labels[:10]

spike_times_sorted = nap.TsGroup.merge_group(spike_times_quiet.getby_category("EI")[1],
                                             spike_times_quiet.getby_category("EI")[-1],
                                             reset_index=True)

# non zero filters
# number of connections

# n1=124
# n2=125

res_filt = []
res_ccg = []

pairs = product(list(spike_times_sorted.keys()),list(spike_times_sorted.keys()))

iter = 0
for n1, n2 in pairs:
    if n1==n2:
        continue
    t0 = perf_counter()
    n1_spikes = {n1: spike_times_sorted[n1]}
    n1_spikes = nap.TsGroup(n1_spikes)
    n2_spikes = {n2: spike_times_sorted[n2]}
    n2_spikes = nap.TsGroup(n2_spikes)

    ccg = nap.compute_crosscorrelogram((n1_spikes,n2_spikes), 0.0001, 0.004, norm=True)
    ccg_cut = ccg.iloc[39:]
    # ccg_norm = (ccg_cut[(n1,n2)] - ccg_cut[(n1,n2)].min()) / (ccg_cut[(n1,n2)].max() - ccg_cut[(n1,n2)].min())
    # ccg_norm = torch.from_numpy(ccg_norm.values)

    gaussian = np.exp(-(np.linspace(0,1,40)/0.3)**2/2)

    ccg_conv = torch.from_numpy(np.convolve(ccg_cut[(n1,n2)], gaussian, mode="full"))
    # ccg_conv_norm = (ccg_conv - ccg_conv.min()) / (ccg_conv.max() - ccg_conv.min())

    filter = filters[n2,n1]
    filter_conv = torch.from_numpy(np.convolve(filter, gaussian, mode="full"))
    # filter_conv_norm = (filter_conv - filter_conv.min()) / (filter_conv.max() - filter_conv.min())

    x = torch.linspace(0,1,79)
    #x2 = torch.linspace(0,0.02,40)

    alpha_ccg = Alpha(x, ccg_conv, cond=0.0075)
    alpha_filter = Alpha(x, filter_conv, cond=0)

    params_ccg, loss_ccg = alpha_ccg.run()
    params_filter, loss_filter = alpha_filter.run()

    res_ccg.append(params_ccg)
    res_filt.append(params_filter)
    iter +=1
    print(f"ready: {n1, n2}, %: {iter/37830},time: {np.round(perf_counter() - t0, 5)}")



# print("CCG:", params_ccg)
# print("Filter:", params_filter)

# res_ccg = alpha_ccg.alpha_function(params_ccg)
# res_ccg[res_ccg < 0] = 0
#
# res_filter = alpha_filter.alpha_function(params_filter)
# res_filter[res_filter < 0] = 0

# fig, axs = plt.subplots(1,3, figsize=(10,4))
# axs[0].plot(ccg_conv_norm, label="ccg conv")
# axs[0].plot(filter_conv_norm, label="filter")
# axs[0].set_title("Target")
# axs[0].legend()
# axs[1].plot(res_ccg.detach().numpy(), label="ccg")
# axs[1].plot(res_filter.detach().numpy(), label="filter")
# axs[1].set_title("Alpha function")
# axs[1].legend()
# axs[2].plot(loss_ccg, label="ccg")
# axs[2].plot(loss_filter, label="filter")
# axs[2].set_title("Loss")
# axs[2].legend()
#
# plt.show()