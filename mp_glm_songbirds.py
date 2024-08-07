import multiprocessing as mp
from time import perf_counter, time
import numpy as np
import pynapple as nap
import nemos as nmo
import os
import random

nap.nap_config.suppress_conversion_warnings = True

def prepare_batch(batch_size, seed, starts, spike_times, rec, basis):
    np.random.seed(seed)
    binsize = 0.01
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
    import nemos as nmo
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

def model_update(batch_queue, queue_semaphore, server_semaphore, shutdown_flag, max_iterations, params, state, model,
                 score_train, score_test, k):
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    import jax.numpy as jnp
    import nemos as nmo
    score_train_pass = []
    score_test_pass = []
    counter = 0
    tep0 = perf_counter()
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

                if counter%500==0:
                    tsc0 = perf_counter()
                    score_train_pass.append(model.score(X, Y, score_type="log-likelihood"))
                    tsc1 = perf_counter()
                    print(f"computed ll: {tsc1 - tsc0}")

                    tep1 = perf_counter()
                    print(f"pass {counter%500}  completed: {tep1 - tep0}, train_ll: {score_train[-1]}")
                    tep0 = perf_counter()

            except batch_queue.Empty:
                score_train[k,:] = score_train_pass
                print("EMPTY")
                pass
    shutdown_flag.set()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Use 'spawn' start method

    n_neurons = 5
    n_sec =100.0
    spikes = [np.random.uniform(0.0, n_sec, 20) for i in range(n_neurons)]
    spikes = np.array((spikes))
    ts_dict = {key: nap.Ts(spikes[key, :].flatten()) for key in range(spikes.shape[0])}
    spike_times = nap.TsGroup(ts_dict)

    gap_starts = np.random.uniform(5, 95, 3)
    gaps = nap.IntervalSet(gap_starts, gap_starts + 3)
    time_quiet = spike_times.time_support.set_diff(gaps)

    # set parameters
    shutdown_flag = mp.Event()
    batch_qsize = 3  # Number of pre-loaded batches
    batch_queue = mp.Queue(maxsize=batch_qsize)
    queue_semaphore = mp.Semaphore(batch_qsize)
    server_semaphore = mp.Semaphore(0)
    # model
    kf = 1
    n_ep = 2
    n_bat = 10
    n_passes = 10
    max_iter = n_bat * n_passes
    binsize = 0.01

    batch_size = time_quiet.tot_length() / n_bat

    starts = []
    start = 0.0
    for i in range(n_bat):
        starts.append(start)
        end = start + batch_size
        ep = nap.IntervalSet(start, end)
        while not time_quiet.intersect(ep):
            start += batch_size
            end += batch_size
            ep = nap.IntervalSet(start, end)
            if end > time_quiet.end[-1]:
                break
        else:
            start += batch_size

    # choose spike history window
    hist_window_sec = 0.4
    mean_rate = np.mean(spike_times.rates) / binsize
    hist_window_size = int(hist_window_sec * mean_rate)

    # define  basis
    n_fun = 9
    basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
    b_time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
    b_time *= hist_window_sec

    # PERFORM CROSS VALIDATION
    # output lists
    score_train = np.zeros((kf, n_passes))
    score_test = np.zeros((kf, n_passes))
    weights = []
    filters = []
    intercepts = []

    for k in range(kf):
        shutdown_flag.clear()

        counter = mp.Value('i', 0)

        # start the batch loader threads
        processes = []
        n_proc = 3
        for id in range(1, n_proc+1):
            seed = ((id+k) * int(time())) % 123456789
            p = mp.Process(target=batch_loader,
                           args=(batch_queue, queue_semaphore, server_semaphore, shutdown_flag,
                                 starts, seed, id, counter, batch_size, spike_times, 0, basis))
            p.start()
            processes.append(p)


        tinit0 = perf_counter()
        init_ep = nap.IntervalSet(starts[0], starts[0]+batch_size)
        init_Y_counts = (spike_times[0].count(binsize, ep=init_ep)).squeeze()
        init_X = basis.compute_features(spike_times.count(binsize, ep=init_ep))
        model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(
            solver_name="GradientDescent", solver_kwargs={"stepsize": 0.2, "acceleration": False}))
        params, state = model.initialize_solver(init_X.d, init_Y_counts)
        tinit1 = perf_counter()
        print(f"model initialization: {tinit1-tinit0}")

        # train model
        server_process = mp.Process(target=model_update,
                                    args=(batch_queue, queue_semaphore, server_semaphore,
                                          shutdown_flag, max_iter, params, state, model,
                                          score_train, score_test, k))
        server_process.start()
        server_process.join()  # Wait for the server process to finish

        shutdown_flag.set()

        for p in processes:
            p.join()

        # save model parameters
        weights.append(model.coef_.reshape(n_fun,-1))
        filters.append(np.matmul(basis_kernels, model.coef_.reshape(n_fun,-1)))
        intercepts.append(model.intercept_)

