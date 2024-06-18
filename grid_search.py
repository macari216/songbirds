import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
from sklearn.utils import gen_batches
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
time_on = nap.IntervalSet(0, off_time)
time_quiet = time_on.set_diff(audio_on)
time_quiet = time_quiet.drop_short_intervals(1,'s')

binsize = 0.005   # in seconds
count = spike_times_quiet.count(binsize, ep=time_quiet)

duration = count.time_support.tot_length("s")
start = count.time_support["start"]
end = count.time_support["end"]
training = nap.IntervalSet(start[0], start[0] + duration * 0.7)
testing = nap.IntervalSet(start[0] + duration * 0.7, end[-1])

count_train = count.restrict(training)
count_train = count_train.restrict(time_quiet.intersect(training))

hist_window_sec = 0.05
# hist_window_size = [int(hist_window_sec[i] * count.rate) for i in range(len(hist_window_sec))]
hist_window_size = int(hist_window_sec * count.rate)

reg = [nmo.regularizer.Ridge(solver_name="GradientDescent",solver_kwargs={"stepsize": 0.001, "acceleration": False}),
       nmo.regularizer.UnRegularized(solver_name="GradientDescent",solver_kwargs={"stepsize": 0.001, "acceleration": False})]

batch_size = 5 #sec

def batcher():



    t = np.random.uniform(count_train.time_support[0, 0], count_train.time_support[0, 1]-batch_size)

    while ~(t.isin(count_train.time_support)):
        t = np.random.uniform(count_train.time_support[0, 0], count_train.time_support[0, 1] - batch_size)

    ep = nap.IntervalSet(t, t+batch_size)
    counts = count_train.restrict(ep)
    X = basis.compute_features(counts)

    return X, counts


n_ep = 25000

logl = np.zeros((len(reg), n_ep))

for l, regul in enumerate(reg):
    basis = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=hist_window_size)

    glm = nmo.glm.PopulationGLM(regularizer=regul)

    params, state = glm.initialize_solver(*batcher())

    for i in range(n_ep):

        # Get a batch of data
        X, Y = batcher()

        # Do one step of gradient descent.
        params, state = glm.update(params, state, X, Y)

        # Score the model along the time axis
        logl[l,i] = glm.score(X, Y, score_type="log-likelihood")

plt.figure()
for i in range(len(hist_window_size)):
    plt.plot(logl[i], label=hist_window_size[i], alpha=0.3)
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.legend()
plt.show()