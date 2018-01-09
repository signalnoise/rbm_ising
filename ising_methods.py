import numpy as np


class IsingState:

	def __init__(self, state, ising_size):

		self.shape = state.shape
		state = np.array(state)
		self.state = state.reshape((ising_size, ising_size))

	def magnetisation(self):

		return np.mean(np.add(np.multiply(np.reshape(self.state, self.shape), 2), -1))

	def energy(self):

		e = 0
		for i in range(self.state.shape[0]):
			for j in range(self.state.shape[1]):
				e = e - self.spin(i, j)*(self.right(i, j) + self.below(i, j))
		return e

	def spin(self, i, j):
		return 2*self.state[i, j] - 1

	def right(self, i, j):

		if i == self.state.shape[0] - 1:
			index = 0
		else:
			index = i + 1
		return self.spin(index, j)

	def below(self, i, j):

		if j == self.state.shape[1] - 1:
			jndex = 0
		else:
			jndex = j + 1
		return self.spin(i, jndex)


def susceptibility(states, temperature, size):

	magnetisations = np.zeros(len(states))
	for i in range(len(states)):
		magnetisations[i] = states[i].magnetisation()
	return np.var(magnetisations)/(temperature*size)


def heat_capacity(states, temperature, size):

	energies = np.zeros(len(states))
	for i in range(len(states)):
		energies[i] = states[i].energy()
	return np.var(energies)/(size*temperature ** 2)


def avg_magnetisation(states):

	magnetisations = np.zeros(len(states))
	for i in range(len(states)):
		magnetisations[i] = np.absolute(states[i].magnetisation())
	return np.mean(magnetisations)