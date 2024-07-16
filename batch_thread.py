import threading
import queue
from time import perf_counter
import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import argparse
import psutil
import os

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

def set_thread_affinity(core_id):
    psutil.Process(os.getpid()).cpu_affinity([core_id])
    print(f"Thread {threading.current_thread().name} running on core(s): {psutil.Process(os.getpid()).cpu_affinity()}")

def get_allocated_cores():
    allocated_cores = []
    # Get the core IDs from the SLURM_CPUS_ON_NODE environment variable
    cpus_on_node = os.getenv('SLURM_CPUS_ON_NODE')
    if cpus_on_node:
        allocated_cores = list(range(int(cpus_on_node)))
    print(allocated_cores)

get_allocated_cores()

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
    X_counts = spike_times_quiet.count(binsize, ep=ep)
    Y_counts = spike_times_quiet[rec].count(binsize, ep=ep)
    X = basis.compute_features(X_counts)
    return (X, Y_counts.squeeze()), start

def batch_loader(batch_queue, batch_qsize, shutdown_flag, start, train_int, core_id):
    set_thread_affinity(core_id)
    while not shutdown_flag.is_set():
        if batch_queue.qsize() < batch_qsize:
            batch, start = prepare_batch(start, train_int)
            batch_queue.put(batch)

# parameters
batch_qsize = 5  # Number of pre-loaded batches
batch_queue = queue.Queue(maxsize=batch_qsize)

# SET UP MODEL
# parameters
kf = 1
n_ep = 30
n_bat = 500
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
def model_update(batch_queue, shutdown_flag, max_iterations, params, state, core_id):
    set_thread_affinity(core_id)
    iteration = 0
    while iteration < max_iterations and not shutdown_flag.is_set():
        try:
            tmt0 = perf_counter()
            print(f"queue size (thread 0): {batch_queue.qsize()}")
            batch = batch_queue.get(timeout=1)
            X, Y = batch

            tm0 = perf_counter()
            params, state = model.update(params, state, X, Y)
            tm1 = perf_counter()
            print(f"model update: {tm1-tm0}")
            batch_queue.task_done()
            tmt1 = perf_counter()
            print(f"end of iteration {iteration}, total time: {tmt1-tmt0}  -------------")
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

for k, test_int in enumerate(tests):
    train_int = time_on.set_diff(test_int)

    batch_size = train_int.tot_length() / n_bat

    start = train_int.start[0]

    # start the batch loader threads
    loader_threads = []
    n_lthreads = 9
    for i in range(n_lthreads):
        loader_thread = threading.Thread(target=batch_loader,
                                         args=(batch_queue, batch_qsize, shutdown_flag, start, train_int, i))
        loader_thread.daemon = True  # This makes the batch loader a daemon thread
        loader_threads.append(loader_thread)
        loader_thread.start()

    tinit0 = perf_counter()
    init_ep = nap.IntervalSet(train_int.start[0], train_int.start[0]+batch_size)
    init_Y_counts = (spike_times_quiet[rec].count(binsize, ep=init_ep)).squeeze()
    init_X = basis.compute_features(spike_times_quiet.count(binsize, ep=init_ep))
    model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(
        solver_name="GradientDescent", solver_kwargs={"stepsize": 0.4, "acceleration": False}))
    params, state = model.initialize_solver(init_X, init_Y_counts)
    tinit1 = perf_counter()
    print(f"model initialization: {tinit1-tinit0}")

    # train model
    for ep in range(n_ep):
        tep0 = perf_counter()
        start = train_int.start[0]
        shutdown_flag.clear()

        # update model
        try:
            model_update(batch_queue, shutdown_flag, n_bat, params, state, n_lthreads)
        finally:
            # set the shutdown flag to stop the loader thread
            shutdown_flag.set()
            print("shutdown flag set")
            # wait for the loader thread to exit
            for loader_thread in loader_threads:
                loader_thread.join(1)
            print("threads joined")

        # log score
        tsc0 = perf_counter()
        score_train[k, ep] = model.score(init_X, init_Y_counts, score_type="log-likelihood")
        tsc1 = perf_counter()
        print(f"computed ll: {tsc1-tsc0}")
        # score_test[k, ep] =
        tep1 = perf_counter()
        print(f"epoch {ep}  completed: {tep1-tep0}")
        print(f"K: {k}, Ep: {ep}, train ll: {score_train[k, ep]}, test ll: {score_test[k, ep]}")

    # save model parameters
    # weights.append(model.coef_.reshape(n_fun,-1))
    # filters.append(np.matmul(basis_kernels, model.coef_.reshape(n_fun,-1)))
    # intercepts.append(model.intercept_)

# SAVE RESULTS
results_dict = {"weights": weights, "filters": filters, "intercept": intercepts, "type": spike_times_quiet["EI"],
                "time": time, "basis_kernels": basis_kernels, "train_ll": score_train, "test_ll": score_test}

#np.save(f"/mnt/home/amedvedeva/ceph/songbird_output/results_n{args.Neuron}.npy", results_dict)
