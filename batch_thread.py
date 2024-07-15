import threading
import queue
from time import perf_counter
import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import argparse
#import psutil

nap.nap_config.suppress_conversion_warnings = True

# PREPARE DATA
# select row (post synaptic neuron)
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--Neuron", help="Specify GLM receiver neuron (0-194)")
args = parser.parse_args()
rec = int(args.Neuron)

# load data
t0 = perf_counter()
audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']
t1 = perf_counter()
print(f"loaded data (thread 0): {t1-t0}")

t0 = perf_counter()
#convert times to Interval Sets and spikes to TsGroups
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

# create a TsGroup for spike times
ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add E/I label
spike_times_quiet["EI"] = ei_labels
t1 = perf_counter()
print(f"created TsGroup (thread 0): {t1-t0}")

# PARALLELIZE BATCHING
# set shutdown flag for batch thread
shutdown_flag = threading.Event()

def prepare_batch(start, train_int):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)

    while not train_int.intersect(ep):
        start += batch_size
        end += batch_size
        ep = nap.IntervalSet(start, end)
        if end > train_int.end[-1]:
            break

    start = end
    t_xy0 = perf_counter()
    X_counts = spike_times_quiet.count(binsize, ep=ep)
    Y_counts = spike_times_quiet[rec].count(binsize, ep=ep)
    t_xy1 = perf_counter()
    print(f"computed counts (thread 1): {t_xy1-t_xy0}")
    t_xy0 = perf_counter()
    X = basis.compute_features(X_counts)
    t_xy1 = perf_counter()
    print(f"convolved counts (thread 1): {t_xy1 - t_xy0}")
    return (X, Y_counts.squeeze()), start

def batch_loader(batch_queue, batch_qsize, shutdown_flag, start, train_int):
    while not shutdown_flag.is_set():
        if batch_queue.qsize() < batch_qsize:
            tb0 = perf_counter()
            batch, start = prepare_batch(start, train_int)
            tb1 = perf_counter()
            print(f"prepared batch (thread 0): {tb1-tb0}")
            batch_queue.put(batch)

# parameters
batch_qsize = 10  # Number of pre-loaded batches
batch_queue = queue.Queue(maxsize=batch_qsize)

# SET UP MODEL
# parameters
kf = 1
n_ep = 1
n_bat = 30
binsize = 0.0001

# create time intervals for CV
time_on = nap.IntervalSet(0, off_time).set_diff(audio_segm)
tests = []
t_st = 0
one_int = off_time*0.2
for i in range(kf):
    t_end = t_st + one_int
    test = nap.IntervalSet(t_st, t_end).set_diff(audio_segm)
    tests.append(test)
    t_st = t_end

# choose spike history window
hist_window_sec = 0.004
mean_rate = 9999  # pre-computed, insignificant variance between batches
hist_window_size = int(hist_window_sec * mean_rate)

# define  basis
n_fun = 9
basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

# MODEL STEP (1 epoch)
def model_update(batch_queue, shutdown_flag, max_iterations, params, state):
    iteration = 0
    while iteration < max_iterations and not shutdown_flag.is_set():
        try:
            print(f"queue size (thread 0): {batch_queue.qsize()}")
            batch = batch_queue.get(timeout=1)
            X, Y = batch

            tm0 = perf_counter()
            params, state = model.update(params, state, X, Y)
            score_train[k, ep] = state._error
            tm1 = perf_counter()
            print(f"model step (thread 0): tm1-tm0")
            # score_train[k, ep] = model.score(X, Y, score_type="log-likelihood")
            # score_test[k, ep] =

            batch_queue.task_done()
            print(f"end of iteration {iteration} -------------")
            iteration += 1
        except queue.Empty:
            continue

# PERFORM CROSS VALIDATION
# output lists
score_train = np.zeros((kf, n_ep))
score_test = np.zeros((kf, n_ep))
weights = []
filters = []
intercepts = []

print(f"before epoch 0: {perf_counter()}")
for k, test_int in enumerate(tests):
    train_int = time_on.set_diff(test_int)

    batch_size = train_int.tot_length() / n_bat

    tinit0 = perf_counter()
    init_ep = nap.IntervalSet(train_int.start[0], train_int.start[0]+batch_size)
    init_Y_counts = (spike_times_quiet[rec].count(binsize, ep=init_ep)).squeeze()
    init_X = basis.compute_features(spike_times_quiet.count(binsize, ep=init_ep))
    model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(
        solver_name="GradientDescent", solver_kwargs={"stepsize": 0.2, "acceleration": False}))
    params, state = model.initialize_solver(init_X, init_Y_counts)
    tinit1 = perf_counter()
    print(f"model initialization: {tinit1-tinit0}")

    # train model
    for ep in range(n_ep):
        start = train_int.start[0]
        shutdown_flag.clear()
        # start the batch loader thread
        loader_thread = threading.Thread(target=batch_loader,
                                         args=(batch_queue, batch_qsize, shutdown_flag, start, train_int))
        loader_thread.daemon = True  # This makes the batch loader a daemon thread
        loader_thread.start()
        print(f"started loader thread: {perf_counter()}")

        # update model
        try:
            model_update(batch_queue, shutdown_flag, n_bat, params, state)
        finally:
            # set the shutdown flag to stop the loader thread
            shutdown_flag.set()
            # wait for the loader thread to exit
            loader_thread.join()

        # log score
        print(f"epoch {ep}  completed: {perf_counter()}")
        print(f"K: {k}, Ep: {ep}, train ll: {score_train[k, ep]}, test ll: {score_test[k, ep]}")

    # save model parameters
    weights.append(model.coef_.reshape(n_fun,-1))
    filters.append(np.matmul(basis_kernels, model.coef_.reshape(n_fun,-1)))
    intercepts.append(model.intercept_)

# SAVE RESULTS
results_dict = {"weights": weights, "filters": filters, "intercept": intercepts, "type": spike_times_quiet["EI"],
                "time": time, "basis_kernels": basis_kernels, "train_ll": score_train, "test_ll": score_test}

#np.save(f"/mnt/home/amedvedeva/ceph/songbird_output/results_n{args.Neuron}.npy", results_dict)
