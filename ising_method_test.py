import rbm_pytorch
import ising_methods
import numpy as np

training_data = "../../state02.27.txt"
data = np.loadtxt(training_data, delimiter=",",skiprows=1, dtype="float32")

temp = 2.27
N = 64
length = 8

newshape = data.reshape(-1, length**2)

print(newshape[0])



states = []

for x in range(len(newshape)):
	states.append(ising_methods.IsingState(newshape[x], length))

print(ising_methods.susceptibility(states, temp,N))
print(ising_methods.heat_capacity(states,temp,N))
