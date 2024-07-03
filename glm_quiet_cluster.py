import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
import pandas as pd
import argparse

rows = [101,174]

nap.nap_config.suppress_conversion_warnings = True

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57EI.mat')['c57EI']

#convert times to Interval Sets and spikes to TsGroups
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in rows}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add E/I labels
subset_labels = []
for idx in range(195):
    if idx in rows:
        subset_labels.append(ei_labels[idx])

spike_times_quiet["EI"] = subset_labels

# spike count
training_end = off_time * 0.7
time_quiet_train = nap.IntervalSet(0, training_end).set_diff(audio_segm)
time_quiet_test = nap.IntervalSet(training_end, off_time).set_diff(audio_segm)

binsize = 0.001   # in seconds
count_train = spike_times_quiet.count(binsize, ep=time_quiet_train)

count_test = spike_times_quiet.count(binsize, ep=time_quiet_test)

n_neurons = count_test.shape[1]

#choose spike history window
hist_window_sec = 0.01
hist_window_size = int(hist_window_sec * count_test.rate)

# define  basis
n_fun = 9
basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

X_test = basis.compute_features(count_test)


# implement minibatching
n_bat = 1000
batch_size = time_quiet_train.tot_length() / n_bat

def batcher(start):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)
    start = end
    counts = count_train.restrict(ep)
    X = basis.compute_features(counts)
    return X, counts, start

# # mask for group lasso
# n_groups = n_neurons
# n_features = n_neurons * n_fun
#
# mask = np.zeros((n_groups, n_features))
# for i in range(n_groups):
#     mask[i, i*n_fun:i*n_fun+n_fun] = np.ones(n_fun)

# define and initialize model
model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized(
    solver_name="GradientDescent", solver_kwargs={"stepsize": 0.1, "acceleration": False}))
start = time_quiet_train.start[0]
params, state = model.initialize_solver(*batcher(start))

# train model
n_ep = 60
score_train = np.zeros(n_ep)
score_test = np.zeros(n_ep)

for ep in range(n_ep):
    start = time_quiet_train.start[0]
    for i in range(n_bat):
        # Get a batch of data
        X, Y, start = batcher(start)

        # Do one step of gradient descent.
        params, state = model.update(params, state, X, Y)

    # Score the model along the time axis
    score_train[ep] = model.score(X, Y, score_type="log-likelihood")
    score_test[ep] = model.score(X_test, count_test.squeeze(), score_type="log-likelihood")

    if ep%5 ==0:
        print(f"Epoch: {ep}, train ll:{score_train[ep]}, test ll:{score_test[ep]}")

# model output
weights = model.coef_.reshape(n_neurons, basis.n_basis_funcs, n_neurons)
filters = np.einsum("jki,tk->ijt", weights, basis_kernels)
spike_pred = model.predict(X_test)
intercept = model.intercept_
coef = model.coef_
results_dict = {"weights": weights, "filters": filters, "intercept": intercept, "coef": coef,
                "spike_pred": spike_pred, "time": time, "train_ll": score_train, "test_ll": score_test}

np.save("/Users/macari216/Desktop/glm_songbirds/songbirds/results_101_174.npy", results_dict)

plt.figure()
plt.plot(score_train, color='b', label="train")
plt.plot(score_test, color='r', label="test")
plt.xlabel("Epoch")
plt.ylabel("log-likelihood")
plt.legend()
plt.show()




