import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt

nap.nap_config.suppress_conversion_warnings = True
np.random.seed(16)

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

hist_window_sec = [0.02, 0.05, 0.1, 0.9]
hist_window_size = [int(hist_window_sec[i] * count_train.rate) for i in range(len(hist_window_sec))]
# hist_window_size = int(hist_window_sec * count_train.rate)

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

for l, ws in enumerate(hist_window_size):
    basis = nmo.basis.RaisedCosineBasisLog(5, mode="conv", window_size=ws)

    glm = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(solver_name="GradientDescent",solver_kwargs={"stepsize": 0.1, "acceleration": False}))

    start = count_train.time_support.start[0]

    params, state = glm.initialize_solver(*batcher(start))

    for i in range(n_bat):
        # Get a batch of data
        X, Y,start = batcher(start)

        # Do one step of gradient descent.
        params, state = glm.update(params, state, X, Y)

        # Score the model along the time axis
        logl[l, i] = glm.score(X, Y, score_type="log-likelihood")

    X_test = basis.compute_features(count_test)
    score_test = glm.score(X_test, count_test.squeeze(), score_type="log-likelihood")
    print("Score(train data):", logl[l, -1])
    print("Score(test data):", score_test)

plt.figure()
for i in range(len(hist_window_size)):
    plt.plot(logl[i], label=hist_window_size[i], alpha=0.3)
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.legend()
plt.show()