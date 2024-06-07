import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt

# most active neurons
quiet_rates = spike_times_quiet['rate']
quiet_thr = np.median(quiet_rates)
hf_quiet = quiet_rates[quiet_rates > quiet_thr]
hfq_id = np.array(hf_quiet.index)

# plot coupling weights for 2 neurons
 plt.figure()
 plt.title("Coupling Filter Between Neurons 1 and 2")
 plt.plot(time, responses[0,1,:], "k", lw=2)
 plt.axhline(0, color="k", lw=0.5)
 plt.xlabel("Time from spike (sec)")
 plt.ylabel("Weight")

# plot spiking rates
spike_pred = model.predict(X.restrict(testing))
ep = nap.IntervalSet(testing.start, testing.start+30)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
firing_rate = spike_pred.restrict(ep).d[:,:10]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
ax1.set_ylabel("Neuron")
ax1.set_title("Predicted firing rate")
ax2 = fig.add_subplot(2,1,2)
firing_rate = count.restrict(ep).d[:,:10]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
ax2.set_ylabel("Neuron")
ax2.set_title("True firing rate")
fig.subplots_adjust(hspace=1)
plt.xlabel("Time (sec)")
plt.show()