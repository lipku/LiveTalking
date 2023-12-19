import numpy as np

net_output = np.load('french.ds.npy').reshape(-1, 29)
win_size = 16
zero_pad = np.zeros((int(win_size / 2), net_output.shape[1]))
net_output = np.concatenate((zero_pad, net_output, zero_pad), axis=0)
windows = []
for window_index in range(0, net_output.shape[0] - win_size, 2):
        windows.append(net_output[window_index:window_index + win_size])
print(np.array(windows).shape)
np.save('aud_french.npy', np.array(windows))
