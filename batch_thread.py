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
    def __init__(self, batch_queue, queue_semaphore, server_semaphore, stop_event, num_iterations, shared_results):
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        import nemos
        self.nemos = nemos
        self.model = nemos.glm.GLM(
            regularizer=nemos.regularizer.UnRegularized(
                solver_name="GradientDescent",
                solver_kwargs={"stepsize": 0.2, "acceleration": False},
            )
        )

        # set mp attributes
        self.batch_queue = batch_queue
        self.queue_semaphore = queue_semaphore
        self.server_semaphore = server_semaphore
        self.stop_event = stop_event
        self.num_iterations = num_iterations
        self.shared_results = shared_results

    def run(self):
        params, state = None, None
        train_ll = []
        counter = 0
        while not self.stop_event.is_set() and counter < self.num_iterations:
            if self.server_semaphore.acquire(timeout=1):  # Wait for a signal from a worker
                try:

                    tm0 = perf_counter()
                    # grab the batch (we are not using the seq number)
                    # at timeout it raises an exception
                    tb0 = perf_counter()
                    sequence_number, batch = self.batch_queue.get(timeout=1)
                    tb1 = perf_counter()
                    print(f"aqcuired batch from queue, time: {tb1-tb0}")
                    # initialize at first iteration
                    if counter == 0:
                        params, state = self.model.initialize_solver(*batch)
                    # update
                    params, state = self.model.update(params, state, *batch)

                    self.queue_semaphore.release()  # Release semaphore after processing
                    tm1 = perf_counter()
                    print(f"model step {counter}, time: {tm1-tm0}")
                    counter += 1

                    if counter%500==0:
                        tsc0 = perf_counter()
                        train_score = self.model.score(*batch, score_type="log-likelihood")
                        train_ll.append(train_score)
                        tsc1 = perf_counter()
                        print(f"train ll: {train_score}, time: {tsc1 - tsc0}")

                except Exception as e:
                    print(f"Exception: {e}")
                    pass
        # stop workers
        self.stop_event.set()
        # add the model to the manager shared param
        self.shared_results["params"] = params
        self.shared_results["state"] = state
        self.shared_results["train_ll"] = train_ll
        print("run returned for server")


class Worker:
    def __init__(self,
                 worker_id: int,
                 neuron_id: int,
                 spike_times: nap.TsGroup,
                 time_quiet: nap.IntervalSet,
                 batch_size_sec: float,
                 n_batches: int,
                 bin_size=None,
                 n_basis_funcs=None,
                 hist_window_sec=None,
                 batch_queue: mp.Queue = None,
                 queue_semaphore: mp.Semaphore = None,
                 server_semaphore: mp.Semaphore = None,
                 shutdown_flag: mp.Event = None,
                 counter: mp.Value = None):
        """Store parameters and config jax"""
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import nemos
        import jax

        # import jax dependent libs
        self.nemos = nemos
        self.jax = jax

        # store worker info
        self.worker_id = worker_id

        # store multiprocessing attributes
        self.batch_queue = batch_queue
        self.queue_semaphore = queue_semaphore
        self.server_semaphore = server_semaphore
        self.shutdown_flag = shutdown_flag
        self.counter = counter

        # store model design hyperparameters
        self.bin_size = bin_size
        self.hist_window_size = int(hist_window_sec / bin_size)

        self.batch_size_sec = batch_size_sec
        self.batch_size = int(batch_size_sec / bin_size)
        self.basis = self.configure_basis(n_basis_funcs)
        self.spike_times = spike_times
        self.neuron_id = neuron_id
        self.starts = self.compute_starts(n_bat=n_batches, time_quiet=time_quiet)

        # set worker based seed
        np.random.seed(123 + worker_id)

    def compute_starts(self, n_bat, time_quiet):
        starts = []
        start = 0.0
        for _ in range(n_bat):
            starts.append(start)
            end = start + self.batch_size
            ep = nap.IntervalSet(start, end)
            while not time_quiet.intersect(ep):
                start += self.batch_size
                end += self.batch_size
                ep = nap.IntervalSet(start, end)
                if end > time_quiet.end[-1]:
                    break
            else:
                start += self.batch_size
        return starts

    def configure_basis(self, n_basis_funcs):
        """Define basis and other computations."""
        basis = self.nemos.basis.RaisedCosineBasisLog(
            n_basis_funcs, mode="conv", window_size=self.hist_window_size
        )
        return basis

    def batcher(self):
        start = random.choice(self.starts)
        ep = nap.IntervalSet(start, start + self.batch_size_sec)
        X_counts = self.spike_times.count(self.bin_size, ep=ep)
        Y_counts = X_counts[:, self.neuron_id]
        X = self.basis.compute_features(X_counts)
        return (X.d, (Y_counts.d).squeeze())

    def run(self):
        compute_new_batch = True
        while not self.shutdown_flag.is_set():
            if compute_new_batch:
                print(f"worker {self.worker_id} preparing a batch...")
                batch = self.batcher()
                compute_new_batch = False
            if self.queue_semaphore.acquire(timeout=1):
                with self.counter.get_lock():
                    sequence_number = self.counter.value
                    self.counter.value += 1
                self.batch_queue.put((sequence_number, batch))
                self.server_semaphore.release()
                compute_new_batch = True

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
    n_batches = 300
    batch_size_sec = time_quiet_train.tot_length() / n_batches
    num_iterations = 300

    # set up workers
    num_workers = 3
    workers = []
    for worker_id in range(1, num_workers + 1):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, neuron_id, spike_times, time_quiet_train, batch_size_sec, n_batches),
            kwargs=dict(
                bin_size=0.0001,
                n_basis_funcs=9,
                hist_window_sec=0.004,
                batch_queue=batch_queue,
                queue_semaphore=queue_semaphore,
                server_semaphore=server_semaphore,
                shutdown_flag=shutdown_flag,
                counter=mp_counter
            )
        )
        p.start()
        workers.append(p)
        print(f"Worker id {worker_id} pid = {p.pid}")

    server = mp.Process(
        target=server_process,
        args=(batch_queue, queue_semaphore, server_semaphore, shutdown_flag, num_iterations, shared_results),
    )
    server.start()
    server.join()

    out = shared_results[0]
    if out:
        params, state = out
        print("final params", params)
    else:
        print("no shared model in the list...")

    # Signal workers to stop
    shutdown_flag.set()
    print("flag set")

    # Release all semaphores to unblock workers if they are waiting
    for _ in range(num_workers):
        print("releasing queue semaphore")
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