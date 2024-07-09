import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt

nap.nap_config.suppress_conversion_warnings = True

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

audio_on = audio_segm[:149]
training_end = off_time * 0.7
time_on_train = nap.IntervalSet(0, training_end)
time_on_test = nap.IntervalSet(training_end, off_time)
time_quiet_train = time_on_train.set_diff(audio_on)
time_quiet_train = time_quiet_train.drop_short_intervals(1,'s')
time_quiet_test = time_on_test.set_diff(audio_on)
time_quiet_test = time_quiet_test.drop_short_intervals(1,'s')

binsize = 0.005   # in seconds
count_train = spike_times_quiet.count(binsize, ep=time_quiet_train)
count_test = spike_times_quiet.count(binsize, ep=time_quiet_test)

n_neurons = count_train.shape[1]

hist_window_sec = 0.03
# hist_window_size = [int(hist_window_sec[i] * count_train.rate) for i in range(len(hist_window_sec))]
hist_window_size = int(hist_window_sec * count_train.rate)

n_fun = 5

# mask for group lasso
n_groups = n_neurons
n_features = n_neurons * n_fun

mask = np.zeros((n_groups, n_features))
for i in range(n_groups):
    mask[i, i*n_fun:i*n_fun+n_fun] = np.ones(n_fun)

n_bat = 100
batch_size = count_train.time_support.tot_length() / n_bat

def batcher(start):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)
    start = end
    counts = count_train.restrict(ep)
    X = basis.compute_features(counts)

    return X, counts, start

reg_strength = [8e-5, 5e-5, 2e-5]

n_ep = 100
logl = np.zeros((len(reg_strength), n_ep))
zero_fraction = np.zeros(len(reg_strength))

for l, reg in enumerate(reg_strength):
    basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)

    glm = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.GroupLasso(
        solver_name="ProximalGradient", mask=mask, solver_kwargs={"stepsize": 0.1, "acceleration": False},
        regularizer_strength=reg))

    start = count_train.time_support.start[0]

    params, state = glm.initialize_solver(*batcher(start))

    for ep in range(n_ep):
        start = count_train.time_support.start[0]
        for i in range(n_bat):
            # Get a batch of data
            X, Y, start = batcher(start)

            # Do one step of gradient descent.
            params, state = glm.update(params, state, X, Y)

            # Score the model along the time axis
        logl[l, ep] = glm.score(X, Y, score_type="log-likelihood")

        if ep%5 ==0:
            print("Epoch:", ep, ", score:", logl[l,ep])

    zero_fraction[l] = glm.coef_[glm.coef_==0].size / glm.coef_.size
    X_test = basis.compute_features(count_test)
    score_test = glm.score(X_test, count_test.squeeze(), score_type="log-likelihood")
    print(reg)
    print("Score(train data):", logl[l, -1])
    print("Score(test data):", score_test)
    print("Fraction of weights set to 0:", zero_fraction[l])

plt.figure()
for i in range(len(reg_strength)):
    plt.plot(logl[i], label=reg_strength[i], alpha=0.3)
plt.xlabel("Epoch")
plt.ylabel("pseudo-r2")
plt.legend()
plt.show()