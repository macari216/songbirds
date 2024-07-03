import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--Neuron", help="Specify GLM input neuron (0-194)")
args = parser.parse_args()

nap.nap_config.suppress_conversion_warnings = True

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57EI.mat')['c57EI']

#convert times to Interval Sets and spikes to TsGroups
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_quiet = {int(args.Neuron): nap.Ts(spikes_quiet[int(args.Neuron), 0].flatten())}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add E/I label
spike_times_quiet["EI"] = ei_labels[int(args.Neuron)]

# time intervals for CV
kf = 5

time_on = nap.IntervalSet(0, off_time).set_diff(audio_segm)
tests = []
t_st = 0
one_int = off_time*0.2
for i in range(kf):
    t_end = t_st + one_int
    test = nap.IntervalSet(t_st, t_end).set_diff(audio_segm)
    tests.append(test)
    t_st = t_end

#training epochs
n_ep = 30

# output lists
score_train = np.zeros((kf, n_ep))
score_test = np.zeros((kf, n_ep))
weights = []
filters = []
intercepts = []

# compute train and test counts
binsize = 0.0001

for k, test_int in enumerate(tests):
    count_test = spike_times_quiet.count(binsize, ep=test_int)

    train_int = time_on.set_diff(test_int)
    count_train = spike_times_quiet.count(binsize, ep=train_int)

    # choose spike history window
    hist_window_sec = 0.004
    hist_window_size = int(hist_window_sec * count_train.rate)

    # define  basis
    n_fun = 9
    basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
    time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
    time *= hist_window_sec

    X_test = basis.compute_features(count_test)


    # implement minibatching
    n_bat = 500
    batch_size = train_int.tot_length() / n_bat

    def batcher(start):
        end = start + batch_size
        ep = nap.IntervalSet(start, end)

        while not count_train.time_support.intersect(ep):
            start += batch_size
            end += batch_size
            ep = nap.IntervalSet(start, end)
            if end > count_train.time_support.end[-1]:
                break

        start = end
        counts = count_train.restrict(ep)
        X = basis.compute_features(counts)
        return X, counts.squeeze(), start

    # define and initialize model
    model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(
        solver_name="GradientDescent", solver_kwargs={"stepsize": 0.1, "acceleration": False}))
    start = train_int.start[0]
    params, state = model.initialize_solver(*batcher(start))

    # train model
    for ep in range(n_ep):
        start = train_int.start[0]
        for i in range(n_bat):
            # Get a batch of data
            X, Y, start = batcher(start)

            # Do one step of gradient descent.
            params, state = model.update(params, state, X, Y)

        # Score the model along the time axis
        score_train[k,ep] = model.score(X, Y, score_type="log-likelihood")
        score_test[k,ep] = model.score(X_test, count_test.squeeze(), score_type="log-likelihood")

        if ep%10 ==0:
            print(f"Fold: {k}, epoch: {ep}, train ll:{score_train[k,ep]}, test ll:{score_test[k,ep]}")

    # model output
    weights.append(model.coef_)
    filters.append(np.matmul(basis_kernels, np.squeeze(model.coef_)))
    intercepts.append(model.intercept_)


results_dict = {"weights": weights, "filters": filters, "intercept": intercepts, "type": spike_times_quiet["EI"],
                "time": time, "basis_kernels": basis_kernels, "train_ll": score_train, "test_ll": score_test}

np.save("/Users/macari216/Desktop/glm_songbirds/songbirds/results_0.npy", results_dict)




