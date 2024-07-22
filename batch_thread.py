import multiprocessing as mp
import os
import random
import argparse
from time import perf_counter

import numpy as np
import scipy.io as sio
import pynapple as nap

nap.nap_config.suppress_conversion_warnings = True

class Server:
    def __init__(self, conns, semaphore_dict, shared_arrays, stop_event, num_iterations, shared_results, array_shape,
                 n_basis_funcs=9, bin_size=None, hist_window_sec=None, neuron_id=0):
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        import nemos
        import jax
        self.jax = jax
        self.nemos = nemos
        self.model = nemos.glm.GLM(
            regularizer=nemos.regularizer.UnRegularized(
                solver_name="GradientDescent",
                solver_kwargs={"stepsize": 0, "acceleration": False},
            )
        )

        # set mp attributes
        self.conns = conns
        self.semaphore_dict = semaphore_dict
        self.batch_queue = batch_queue
        self.stop_event = stop_event
        self.num_iterations = num_iterations
        self.shared_results = shared_results
        self.bin_size = bin_size
        self.hist_window_size = int(hist_window_sec / bin_size)
        self.basis = self.nemos.basis.RaisedCosineBasisLog(
            n_basis_funcs, mode="conv", window_size=self.hist_window_size
        )
        self.array_shape = array_shape
        self.neuron_id = neuron_id
        self.shared_arrays = shared_arrays
        print(f"ARRAY SHAPE {self.array_shape}")

    def run(self):
        params, state = None, None
        train_ll = []
        counter = 0
        tep0 = perf_counter()
        while not self.stop_event.is_set() and counter < self.num_iterations:
            if conn.poll(1):  # Wait for a signal from a worker
                try:
                    t0 = perf_counter()
                    worker_id = conn.recv()
                    print(f"control message worker {worker_id} loaded, time: {np.round(perf_counter() - t0, 5)}")

                    t0 = perf_counter()
                    x_count = np.frombuffer(self.shared_arrays[worker_id], dtype=np.float32).reshape(
                        self.array_shape)
                    print(f"data loaded, time: {np.round(perf_counter() - t0, 5)}")

                    self.semaphore_dict[worker_id].release() # Release semaphore after processing

                    #convolve x counts
                    y = x_count[:, self.neuron_id]
                    t0 = perf_counter()
                    X = self.basis.compute_features(x_count)
                    print(f"convolution performed, time: {np.round(perf_counter() - t0, 5)}")

                    # initialize at first iteration
                    if counter == 0:
                        params, state = self.model.initialize_solver(X.d,y)
                    # update
                    t0 = perf_counter()
                    params, state = self.model.update(params, state, X.d,y)
                    print(f"model step {counter}, time: {np.round(perf_counter() - t0, 5)}")
                    counter += 1

                    if counter%(self.num_iterations/10)==0:
                        train_score = self.model.score(X.d, y, score_type="log-likelihood")
                        train_ll.append(train_score)
                        print(f"train ll: {train_score}")

                except Exception as e:
                    print(f"Exception: {e}")
                    pass

        # stop workers
        print(f"all interations, time: {perf_counter() - tep0}")
        self.stop_event.set()
        # add the model to the manager shared param
        self.shared_results["params"] = params
        self.shared_results["state"] = state
        self.shared_results["train_ll"] = train_ll
        print("run returned for server")


class Worker:
    def __init__(self, conn, worker_id, spike_times, time_quiet, batch_size_sec, n_batches, shared_array, semaphore,
                 bin_size=0.0001, hist_window_sec=0.004, shutdown_flag=None, n_seconds=None):
        """Store parameters and config jax"""
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import nemos
        import jax
        self.nemos = nemos
        self.jax = jax

        # store worker info
        self.conn = conn
        self.worker_id = worker_id

        # store multiprocessing attributes
        self.shutdown_flag = shutdown_flag
        self.shared_array = shared_array
        self.semaphore = semaphore

        # store model design hyperparameters
        self.bin_size = bin_size
        self.hist_window_size = int(hist_window_sec / bin_size)
        self.batch_size_sec = batch_size_sec
        self.batch_size = int(batch_size_sec / bin_size)
        self.spike_times = spike_times
        self.epochs = self.compute_starts(n_bat=n_batches, time_quiet=time_quiet, n_seconds=n_seconds)

        # set worker based seed
        np.random.seed(123 + worker_id)

    def compute_starts(self, n_bat, time_quiet, n_seconds):
        iset_batches = []
        cnt = 0

        while cnt < n_bat:
            start = np.random.uniform(0, n_seconds-self.batch_size_sec)
            end = start + self.batch_size_sec
            ep = nap.IntervalSet(start, end).intersect(time_quiet)
            delta_t = self.batch_size_sec - ep.tot_length()

            while delta_t > 0:
                end += delta_t
                ep = nap.IntervalSet(start, end).intersect(time_quiet)
                delta_t = self.batch_size_sec - ep.tot_length()

            iset_batches.append(ep)
            cnt += 1

        return iset_batches

    def batcher(self):
        ep = self.epochs[np.random.choice(range(len(self.epochs)))]
        X_counts = self.spike_times.count(self.bin_size, ep=ep)
        return np.asarray(X_counts.d, dtype=np.float32)

    def run(self):
        try:
            while not self.shutdown_flag.is_set():
                if not self.semaphore.acquire(timeout=1):
                    continue
                t0 = perf_counter()
                x_count = self.batcher()
                print(f"worker {self.worker_id} batch ready, time: {np.round(perf_counter() - t0, 5)}")
                # Write data to shared memory using dedicated slice
                t0 = perf_counter()
                buffer_array = np.frombuffer(self.shared_array, dtype=np.float32)
                np.copyto(buffer_array, x_count.flatten())
                print(f"worker {self.worker_id} batch copied, time: {np.round(perf_counter() - t0, 5)}")

                self.conn.send(self.worker_id)
                # # Wait for confirmation from server
                # if not self.conn.recv():
                #     print(f"worker {self.worker_id} retrying to send control message...")
                #     continue
        finally:
            print(f"worker {self.worker_id} exits loop...")


def worker_process(*args, **kwargs):
    worker = Worker(*args, **kwargs)
    worker.run()
    print(f"run returned for worker {args[0]}")


def server_process(*args, **kwargs):
    server = Server(*args, **kwargs)
    server.run()
    print(f"run returned for server")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Use 'spawn' start method

    # set MP parameters
    shutdown_flag = mp.Event()
    batch_qsize = 3  # Number of pre-loaded batches
    batch_queue = mp.Queue(maxsize=batch_qsize)
    queue_semaphore = mp.Semaphore(batch_qsize)
    server_semaphore = mp.Semaphore(0)
    mp_counter = mp.Value('i', 0)

    # shared params
    manager = mp.Manager()
    shared_results = manager.dict()  # return the model to the main thread

    # get neuron id
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--Neuron", help="Specify GLM receiver neuron (0-194)")
    args = parser.parse_args()
    neuron_id = int(args.Neuron)

    # load data
    audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
    off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
    spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
    ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

    audio_segm = nap.IntervalSet(start=audio_segm[:, 0], end=audio_segm[:, 1])
    ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
    spike_times = nap.TsGroup(ts_dict_quiet)
    time_quiet_train = nap.IntervalSet(0, off_time*0.8).set_diff(audio_segm)
    time_quiet_test = nap.IntervalSet(off_time * 0.8, off_time).set_diff(audio_segm)

    # set the number of iteration and batches
    n_batches = 500
    n_sec = time_quiet_train.tot_length()
    batch_size_sec = n_sec / n_batches
    num_iterations = 10
    bin_size = 0.0001
    hist_window_sec = 0.004

    # set up workers
    num_workers = 3

    # shared arrays for data transfer
    array_shape = (int(batch_size_sec / bin_size), len(ts_dict_quiet))  # Adjust to match actual data size
    shared_arrays = {i: mp.Array('f', array_shape[0] * array_shape[1], lock=False) for i in range(num_workers)}

    # set up pipes, semaphores, and workers
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_workers)])

    semaphore_dict = {i: mp.Semaphore(1) for i in range(num_workers)}
    workers = []

    for i, conn in enumerate(child_conns):
        p = mp.Process(
            target=worker_process,
            args=(conn, i, spike_times, time_quiet_train, batch_size_sec, n_batches, shared_arrays[i], semaphore_dict[i]),
            kwargs=dict(
                bin_size=bin_size,
                shutdown_flag=shutdown_flag,
                hist_window_sec=hist_window_sec,
                n_seconds=n_sec
            )
        )
        p.start()
        workers.append(p)
        print(f"Worker id {i} pid = {p.pid}")



    # for worker_id in range(1, num_workers + 1):
    #     p = mp.Process(
    #         target=worker_process,
    #         args=(worker_id, neuron_id, spike_times, time_quiet_train, batch_size_sec, n_batches),
    #         kwargs=dict(
    #             bin_size=0.0001,
    #             n_basis_funcs=9,
    #             hist_window_sec=0.004,
    #             batch_queue=batch_queue,
    #             queue_semaphore=queue_semaphore,
    #             server_semaphore=server_semaphore,
    #             shutdown_flag=shutdown_flag,
    #             counter=mp_counter
    #         )
    #     )
    #     p.start()
    #     workers.append(p)
    #     print(f"Worker id {worker_id} pid = {p.pid}")

    server = mp.Process(
        target=server_process,
        args=(parent_conns, semaphore_dict, shared_arrays, shutdown_flag, num_iterations, shared_results, array_shape),
        kwargs=dict(n_basis_funcs=9, hist_window_sec=hist_window_sec, bin_size=hist_window_sec, neuron_id=neuron_id)
    )
    server.start()
    server.join()

    out = shared_results
    if out:
        params = out["params"]
        state = out["state"]
        score_train = out["train_ll"]
        print(type(params[0]), type(params[1]), type(state), type(score_train))
        print("final params", len(params))
    else:
        print("no shared model in the list...")

    # Signal workers to stop
    shutdown_flag.set()
    print("flag set")

    # Save results
    np.save(f"/mnt/home/amedvedeva/ceph/songbird_output/mp_results_n{neuron_id}.npy", shared_results)

    # Release all semaphores to unblock workers if they are waiting
    for _ in range(num_workers):
        queue_semaphore.release()

    # Join worker processes
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            print(f"Worker {p.pid} did not exit gracefully, terminating.")
            p.terminate()
        else:
            print(f"Joined worker {p.pid}")

    print("Script terminated")