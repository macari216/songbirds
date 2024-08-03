import multiprocessing as mp
import os
import argparse
from time import perf_counter

import numpy as np
import scipy.io as sio
import pynapple as nap

nap.nap_config.suppress_conversion_warnings = True

class Server:
    def __init__(self, conns, semaphore_dict, shared_arrays, stop_event, num_iterations, shared_results, array_shape,
                 test_counts, block,
                 reg_strength=1e-05, step_size=0.001, n_basis_funcs=9, hist_window_sec=None, bin_size=None, n_ep=1): # nstart=0, nend=1):
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        import nemos
        import jax
        self.jax = jax
        self.nemos = nemos

        # set mp attributes
        self.array_shape = array_shape
        self.model = self.configure_model(n_basis_funcs ,reg_strength, step_size)
        self.conns = conns
        self.semaphore_dict = semaphore_dict
        self.stop_event = stop_event
        self.n_epochs = n_ep
        self.num_iterations = num_iterations
        self.shared_results = shared_results
        self.block = str(block)
        self.bin_size = bin_size
        self.hist_window_size = int(hist_window_sec / bin_size)
        self.basis = self.nemos.basis.RaisedCosineBasisLog(
            n_basis_funcs, mode="conv", window_size=self.hist_window_size
        )
        self.test_batch = test_counts
        # self.nstart = nstart
        # self.nend = nend
        self.shared_arrays = shared_arrays
        print(f"ARRAY SHAPE {self.array_shape}")

    def configure_model(self, n_basis_funcs, reg_strength, step_size):
        n_groups = self.array_shape[1]
        n_features = n_groups * n_basis_funcs
        mask = np.zeros((n_groups, n_features))
        for i in range(n_groups):
            mask[i, i * n_basis_funcs:i * n_basis_funcs + n_basis_funcs] = np.ones(n_basis_funcs)

        # ee_mask = np.zeros((9 * 195, 195))
        # for j in range(101):
        #     ee_mask[:101 * 9, j] = np.ones(101 * 9)
        #
        # ii_mask = np.zeros((9 * 195, 195))
        # for j in range(101, 195):
        #     ii_mask[101 * 9:, j] = np.ones(94 * 9)
        #
        # ie_mask = np.zeros((9 * 195, 195))
        # for j in range(101):
        #     ie_mask[101 * 9:, j] = np.ones(94 * 9)
        #
        # ei_mask = np.zeros((9 * 195, 195))
        # for j in range(101, 195):
        #     ei_mask[:101 * 9, j] = np.ones(101 * 9)

        model = self.nemos.glm.PopulationGLM(
            #feature_mask=ee_mask,
            regularizer=self.nemos.regularizer.GroupLasso(
                solver_name="ProximalGradient",
                mask=mask,
                solver_kwargs={"stepsize": step_size, "acceleration": False},
                regularizer_strength=reg_strength
            )
        )

        return model

    def select_block(self, x_c):
        if self.block == "ee":
            y = x_c[:,:101]
            x = x_c[:,:101*9]
            return x, y
        elif self.block == "ei":
            y = x_c[:,-94:]
            x = x_c[:,:101*9]
            return x, y

        elif self.block == "ii":
            y = x_c[:,-94:]
            x = x_c[:,-94*9:]
            return x, y

        elif self.block == "ie":
            y = x_c[:,:101]
            x = x_c[:,-94*9:]
            return x, y

    def run(self):
        params, state = None, None
        train_ll = []
        counter = 0
        tep0 = perf_counter()
        while not self.stop_event.is_set() and counter < self.num_iterations:
            for conn in self.conns:
                if conn.poll(1):  # Wait for a signal from a worker
                    try:
                        tt0 = perf_counter()
                        worker_id = conn.recv()
                        t0 = perf_counter()
                        x_count = np.frombuffer(self.shared_arrays[worker_id], dtype=np.float32).reshape(
                            self.array_shape)
                        print(f"data loaded, time: {np.round(perf_counter() - t0, 5)}")

                        self.semaphore_dict[worker_id].release() # Release semaphore after processing

                        #convolve x counts
                        #y = x_count[:, self.nstart:self.nend]
                        x, y = self.select_block(x_count)
                        t0 = perf_counter()
                        X = self.basis.compute_features(x)
                        print(f"convolution performed, time: {np.round(perf_counter() - t0, 5)}")

                        # initialize at first iteration
                        if counter == 0:
                            params, state = self.model.initialize_solver(X, y)
                        # update
                        t0 = perf_counter()
                        params, state = self.model.update(params, state, X,y)
                        print(f"model step {counter}, time: {np.round(perf_counter() - t0, 5)}, total time: {np.round(perf_counter() - tt0, 5)}")
                        counter += 1

                        if counter%50==0:
                            t0 = perf_counter()
                            train_score = self.model.score(X, y, score_type="log-likelihood")
                            train_ll.append(train_score)
                            print(f"train ll: {train_score}, time:{np.round(perf_counter() - t0, 5)}")

                        if counter == self.num_iterations:
                            X = self.basis.compute_features(self.test_batch)
                            y = self.test_batch
                            test_score = self.model.score(X, y, score_type="log-likelihood")
                            train_ll.append(test_score)
                            print(f"test ll: {test_score}")

                    except Exception as e:
                        print(f"Exception: {e}")
                        pass

        # stop workers
        print(f"all iterations, time: {perf_counter() - tep0}")
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
        self.epochs = self.compute_starts(n_bat=n_batches, time_quiet=time_quiet)

        # set worker based seed
        np.random.seed(123 + worker_id)

    def compute_starts(self, n_bat, time_quiet):
        iset_batches = []
        cnt = 0
        t0 = time_quiet.time_span().start
        tn = time_quiet.time_span().end
        while cnt < n_bat:
            start = np.random.uniform(t0, tn)
            end = start + self.batch_size_sec
            tot_time = nap.IntervalSet(end, tn).intersect(time_quiet)
            if tot_time.tot_length() < self.batch_size_sec:
                continue
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
        return nap.TsdFrame(X_counts.t, X_counts.d.astype(np.float32), time_support=X_counts.time_support)


    def run(self):
        try:
            while not self.shutdown_flag.is_set():
                if not self.semaphore.acquire(timeout=1):
                    continue
                x_count = self.batcher()
                splits = [x_count.get(a, b).d for a, b in x_count.time_support.values]
                padding = np.vstack([np.vstack((s, np.full((1, *s.shape[1:]), np.nan))) for s in splits])
                buffer_array = np.frombuffer(self.shared_array, dtype=np.float32)
                n_samp = int(buffer_array.shape[0] / 195)
                np.copyto(buffer_array, padding[:n_samp].flatten())

                self.conn.send(self.worker_id)

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
    batch_qsize = 5  # Number of pre-loaded batches
    batch_queue = mp.Queue(maxsize=batch_qsize)
    queue_semaphore = mp.Semaphore(batch_qsize)
    server_semaphore = mp.Semaphore(0)
    mp_counter = mp.Value('i', 0)

    # shared params
    manager = mp.Manager()
    shared_results = manager.dict()  # return the model to the main thread

    # get cv parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--RegStrength", help="Specify group lasso regularizer strength (float)")
    parser.add_argument("-j", "--JobID", help="Provide Slurm job ID for saving results")
    parser.add_argument("-s", "--StepSize", help="Provide step size for grad descent")
    parser.add_argument("-b", "--Block", help="Select a bloc to fit: ee, ii, ei or ie")
    args = parser.parse_args()
    reg_strength = float(args.RegStrength)
    step_size = float(args.StepSize)
    job_id = args.JobID
    block = args.Block

    # load data
    audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
    off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
    spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
    ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

    audio_segm = nap.IntervalSet(start=audio_segm[:, 0], end=audio_segm[:, 1])
    ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
    spike_times = nap.TsGroup(ts_dict_quiet)
    spike_times["EI"] = ei_labels
    inh = spike_times.getby_category("EI")[-1]
    exc = spike_times.getby_category("EI")[1]
    spike_times_sorted = nap.TsGroup.merge(exc, inh, reset_index=True)

    time_quiet_train = nap.IntervalSet(0, off_time*0.8).set_diff(audio_segm)
    time_quiet_test = nap.IntervalSet(off_time * 0.8, off_time).set_diff(audio_segm)
    #first_half = nap.IntervalSet(0, off_time*0.4).set_diff(audio_segm)
    #second_half = nap.IntervalSet(off_time * 0.4, off_time * 0.8).set_diff(audio_segm)

    # set the number of iteration and batches
    n_batches = 300
    n_epochs = 5
    n_sec = time_quiet_train.tot_length()
    batch_size_sec = n_sec / n_batches
    num_iterations = n_batches * n_epochs
    bin_size = 0.0001
    hist_window_sec = 0.004
    n_fun = 9
    n_presn = len(ts_dict_quiet)
    n_postsn = n_presn
    #n_postsn = len(np.arange(neuron_start, neuron_end))

    # create a test batch
    test_counts = spike_times.count(bin_size, ep=time_quiet_test[54])

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
            args=(conn, i, spike_times_sorted, time_quiet_train, batch_size_sec, n_batches, shared_arrays[i], semaphore_dict[i]),
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

    server = mp.Process(
        target=server_process,
        args=(parent_conns, semaphore_dict, shared_arrays, shutdown_flag, num_iterations, shared_results, array_shape,
              test_counts, block),
        kwargs=dict(reg_strength=reg_strength, step_size=step_size, n_basis_funcs=n_fun, hist_window_sec=hist_window_sec, bin_size=bin_size,
                    n_ep=n_epochs) #nstart=neuron_start, nend=neuron_end)
    )
    server.start()
    server.join()

    out = shared_results
    if out:
        score_train = out["train_ll"]
        model_coef = out["params"][0]
        weights_sum = (model_coef.reshape(n_presn, n_fun, n_postsn)).sum(axis=1)
        print(f"reg: {reg_strength}, step: {step_size}, iter: {num_iterations}")
        print("final params", np.array(score_train))
        print("fraction set to 0:", weights_sum[weights_sum==0].size / weights_sum.size)
    else:
        print("no shared model in the list...")

    # Signal workers to stop
    shutdown_flag.set()
    print("flag set")

    # Save results
    np.save(f"/mnt/home/amedvedeva/ceph/songbird_output/mp_results_{job_id}_{block}.npy", out.copy())

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
