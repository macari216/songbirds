import multiprocessing as mp
from time import perf_counter, time
import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import argparse
import os
import random

nap.nap_config.suppress_conversion_warnings = True

def prepare_batch(batch_size, seed, starts, spike_times, rec, basis):
    np.random.seed(seed)
    binsize = 0.0001
    start = random.choice(starts)
    ep = nap.IntervalSet(start, start+batch_size)
    X_counts = spike_times.count(binsize, ep=ep)
    Y_counts = spike_times[rec].count(binsize, ep=ep)
    X = basis.compute_features(X_counts)
    return (X.d, (Y_counts.d).squeeze()), start

def batch_loader(batch_queue, queue_semaphore, server_semaphore, shutdown_flag,
                 starts, seed, core_id, counter, batch_size, spike_times, rec, basis):
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    import jax
    print(f"Worker {core_id} with seed: {seed}")
    while not shutdown_flag.is_set():
        batch, start = prepare_batch(batch_size, seed, starts, spike_times, rec, basis)

        if queue_semaphore.acquire(timeout=1):
            with counter.get_lock():
                sequence_number = counter.value
                counter.value += 1
            batch_queue.put((sequence_number, batch))
            print(f"Worker {os.getpid()} with ID {core_id} added batch {sequence_number}: {start}")
            server_semaphore.release()

def model_update(batch_queue, queue_semaphore, server_semaphore, shutdown_flag, max_iterations, params, state, model):
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    import jax.numpy as jnp
    counter = 0
    while counter < max_iterations and not shutdown_flag.is_set():
        if server_semaphore.acquire(timeout=1):  # Wait for a signal from a worker
            print(f"Server {os.getpid()} iter: {counter}")
            try:
                tmt0 = perf_counter()
                print(f"queue size (thread 0): {batch_queue.qsize()}")
                sequence_number, batch  = batch_queue.get(timeout=1)
                X, Y = batch

                tm0 = perf_counter()
                params, state = model.update(params, state, X, Y)
                tm1 = perf_counter()
                print(f"model update: {tm1-tm0}")
                queue_semaphore.release()
                tmt1 = perf_counter()
                print(f"end of iteration {counter}, total time: {tmt1-tmt0}  -------------")
                counter += 1
            except batch_queue.Empty:
                print("EMPTY")
                pass
    shutdown_flag.set()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Use 'spawn' start method

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
    t1 = perf_counter()
    print(f"created TsGroup (thread 0): {t1-t0}")

    # set parameters
    shutdown_flag = mp.Event()
    batch_qsize = 3  # Number of pre-loaded batches
    batch_queue = mp.Queue(maxsize=batch_qsize)
    queue_semaphore = mp.Semaphore(batch_qsize)
    server_semaphore = mp.Semaphore(0)
    # model
    kf = 1
    n_ep = 2
    n_bat = 500
    binsize = 0.0001

    # create time intervals for CV
    train_int = nap.IntervalSet(0, off_time*0.8).set_diff(audio_segm)
    test_int = nap.IntervalSet(off_time * 0.8, off_time).set_diff(audio_segm)

    batch_size = train_int.tot_length() / n_bat

    starts = []
    start = 0.0
    for i in range(n_bat):
        starts.append(start)
        end = start + batch_size
        ep = nap.IntervalSet(start, end)
        while not train_int.intersect(ep):
            start += batch_size
            end += batch_size
            ep = nap.IntervalSet(start, end)
            if end > train_int.end[-1]:
                break
        else:
            start += batch_size

    # choose spike history window
    hist_window_sec = 0.004
    mean_rate = 9999  # pre-computed, insignificant variance between batches
    hist_window_size = int(hist_window_sec * mean_rate)

    # define  basis
    n_fun = 9
    basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
    b_time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
    b_time *= hist_window_sec

    # PERFORM CROSS VALIDATION
    # output lists
    score_train = np.zeros((kf, n_ep))
    score_test = np.zeros((kf, n_ep))
    weights = []
    filters = []
    intercepts = []

    for k in range(kf):
        mp.set_start_method('spawn', force=True)

        start = train_int.start[0]

        counter = mp.Value('i', 0)

        # start the batch loader threads
        processes = []
        n_proc = 3
        for id in range(1, n_proc+1):
            seed = ((id+k) * int(time())) % 123456789
            p = mp.Process(target=batch_loader,
                           args=(batch_queue, queue_semaphore, server_semaphore, shutdown_flag,
                                 starts, seed, id, counter, batch_size, spike_times_quiet, rec, basis))
            p.start()
            processes.append(p)


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
            server_process = mp.Process(target=model_update,
                                        args=(batch_queue, queue_semaphore, server_semaphore,
                                              shutdown_flag, 30, params, state, model))
            server_process.start()
            server_process.join()  # Wait for the server process to finish

            shutdown_flag.set()

            for p in processes:
                p.join()

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
    # results_dict = {"weights": weights, "filters": filters, "intercept": intercepts, "type": spike_times_quiet["EI"],
    #                 "time": b_time, "basis_kernels": basis_kernels, "train_ll": score_train, "test_ll": score_test}

    #np.save(f"/mnt/home/amedvedeva/ceph/songbird_output/results_n{args.Neuron}.npy", results_dict)
