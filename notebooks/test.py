import numpy as np

traj = np.load('/local_scratch/hoffmae99/bachelor/chign/radial/test_0/traj.npy')
traj1 = np.load('/local_scratch/hoffmae99/bachelor/chign/radial/test_0/traj1.npy')

print(np.shape(traj))
print(np.shape(traj1))

# print(traj)

# print(traj == traj1[:101, :326])
