import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from sklearn.utils import gen_batches

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

#convert times to Interval Sets and spikes to TsGroups
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

audio_on = audio_segm[:149]
training_end = off_time * 0.7
time_on_train = nap.IntervalSet(0, training_end)
time_on_test = nap.IntervalSet(training_end, off_time)
time_quiet_train = time_on_train.set_diff(audio_on)
time_quiet_train = time_quiet_train.drop_short_intervals(1,'s')
time_quiet_test = time_on_test.set_diff(audio_on)
time_quiet_test = time_quiet_test.drop_short_intervals(1,'s')

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add neuron subset marker
spike_times_quiet["neuron_subset"] = [0] * 195 + [1] * 0

# spike count
binsize = 0.005   # in seconds
count_train = spike_times_quiet.count(binsize, ep=time_quiet_train)
count_test = spike_times_quiet.count(binsize, ep=time_quiet_test)

#choose spike history window
hist_window_sec = 0.05
hist_window_size = int(hist_window_sec * count.rate)

n_bat = 1000
batch_size = count_train.time_support.tot_length() / n_bat

def batcher(start):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)
    start = end
    counts = count_train.restrict(ep)
    X = basis.compute_features(counts)

    return X, counts, start

logl = np.zeros((len(hist_window_size), n_bat))

# define  basis
basis = nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

X_test = basis.compute_features(count_test)

# define and initialize model
model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=1.0, solver_name="LBFGS"))
start = count_train.time_support.start[0]
params, state = model.initialize_solver(*batcher(start))

score_train = np.zeros(n_bat)
score_test = np.zeros(n_bat)

# train model
for i in range(n_bat):
    # Get a batch of data
    X, Y, start = batcher()

    # Do one step of gradient descent.
    params, state = model.update(params, state, X, Y)

    # Score the model along the time axis
    score_train[i] = model.score(X, Y, score_type="pseudo-r2-McFadden")
    score_test[i] = model.score(X_test, count_test.squeeze(), score_type="pseudo-r2-McFadden")


# compute score
print("Score(train data):", score_train[-1])
print("Score(test data):", score_test[-1])

# model output
weights = model.coef_.reshape(count_train.shape[1], basis.n_basis_funcs, count_train.shape[1])
filters = np.einsum("jki,tk->ijt", weights, basis_kernels)
spike_pred = model.predict(X_test)
intercept = model.intercept_
coef = model.coef_
results_dict = {"weights": weights, "filters": filters, "intercept": intercept, "coef": coef,
                "spike_pred": spike_pred, "time": time}

np.save("results.npy", results_dict)
