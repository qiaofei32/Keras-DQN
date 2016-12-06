import os
import gym
import numpy
import random
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import argparse

class DQN_CartPole():
	'''DQN To Play CartPole-v0'''

	def __init__(self):
		'''initialize'''

		self.EXPLORE_SIZE = 500 * 1
		self.MEMORY_SIZE  = 500 * 1
		self.TRAIN_FRAMES = 500 * 2
		self.BATCH_SIZE   = 50  * 1
		self.EPSILON	  = 1.0
		self.INIT_EPSILON = 1.0

		self.GAMMA        = 0.9
		self.TEST_SIZE	  = 100
		self.STEP_SIZE	  = 300
		self.NB_ACTIONS	  = 2
		self.GAME_STATE   = 4
		self.NUM_INPUT	  = 4
		self.MIN_EPSILON  = 0.01
		self.TRAINED_SIZE = 0
		self.MEMORY       = []
		# tuples of (State, Action, Reward, NewState, Done)

		self.env   = gym.make('CartPole-v0')
		self.model = self.create_model()

		self.X_train = []
		self.Y_train = []

	def create_model(self):
		'''create keras model'''

		model = Sequential()
		model.add(Dense(output_dim=128, input_shape=(self.NUM_INPUT,)))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(self.NB_ACTIONS))
		# model.add(Activation("linear"))
		# model.compile(RMSprop(), 'MSE')
		# model.compile('sgd', 'MSE')
		model.add(Activation("softmax"))
		model.compile('sgd', 'categorical_crossentropy')
		return model

	def process_minibatch(self, minibatch):
		"""This does the raining. """
		X_train = []
		Y_train = []

		# Loop through our batch and create arrays for X and y
		for i, memory in enumerate(minibatch):
			# Get stored values.
			old_state_m, action_m, reward_m, new_state_m, done = memory

			# Get prediction on old state.
			old_qval = self.model.predict(numpy.array([old_state_m]), batch_size=1)
			# print old_qval

			# Get prediction on new state.
			newQ = self.model.predict( numpy.array([new_state_m]), batch_size=1)
			# Get best action.
			maxQ = numpy.max(newQ)
			y = numpy.zeros((1, self.NB_ACTIONS))
			y[:] = old_qval[:]

			# Check for terminal state.
			if done:  # terminal state
				update = reward_m
			else:  # non-terminal state
				update = (reward_m + (self.GAMMA * maxQ))

			# Update the value for the action we took.
			y[0][action_m] = update

			X_train.append(old_state_m)
			Y_train.append(y.reshape(-1))

		X_train = numpy.array(X_train)
		Y_train = numpy.array(Y_train)

		# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

		return X_train, Y_train

	def train_model(self):
		'''train and test the model'''

		# Train The Model


		try:
			for epoch in range(self.TRAIN_FRAMES):
				'''Decrement EPSILON over time.'''
				if self.EPSILON > self.MIN_EPSILON or epoch < self.EXPLORE_SIZE:
					self.EPSILON -= (self.INIT_EPSILON / self.TRAIN_FRAMES)

				loss = 0.0
				done = False
				state = self.env.reset()
				win_cnt = 0
				while not done:
					state_input = state
					# Choose an action.
					# if random.random() <= self.EPSILON:
					if numpy.random.rand() <= self.EPSILON or epoch < self.EXPLORE_SIZE:
						action = numpy.random.randint(0, self.NB_ACTIONS)
					else:
						qval = self.model.predict(numpy.array([state_input]), batch_size=1)
						action = numpy.argmax(qval[0])  # best action

					# Take action, observe new state and get reward.
					state, reward, done, _ = self.env.step(action)
					reward = -1.0 if done else +1.0
					if not done:
						win_cnt += 1

					# Experience replay storage.
					self.MEMORY.append((state_input, action, reward, state, done))
					if len(self.MEMORY) > self.MEMORY_SIZE:
						self.MEMORY.pop(0)

					# if len(self.MEMORY) >= self.BATCH_SIZE and epoch >= self.EXPLORE_SIZE:
					if len(self.MEMORY):
						# Randomly sample our experience replay memory
						# minibatch = random.sample(self.MEMORY, self.BATCH_SIZE)
						minibatch = random.sample(self.MEMORY, min(self.BATCH_SIZE, len(self.MEMORY)))

						# Get training values.
						X_train, Y_train = self.process_minibatch(minibatch)
						self.X_train = X_train
						self.Y_train = Y_train

						# Train the model on this batch.
						# checkpointer = ModelCheckpoint(filepath="saved-models/weights.hdf5", verbose=1, save_best_only=True)
						# history = self.model.fit(
						# 	X_train,
						# 	Y_train,
						# 	batch_size=self.BATCH_SIZE,
						# 	nb_epoch=1,
						# 	verbose=0,
						# 	# callbacks=[checkpointer]
						# )

						loss += self.model.train_on_batch(X_train, Y_train)

					# Update the starting state with S'.
					# state_input = new_state

				print("Win: %-5d | Loss: %.5f at %-5d | epsilon %f" % (win_cnt, loss, epoch+1, self.EPSILON))

				# Save the model, and test every 1000 epochs.
				# if epoch % 1000 == 0:
				# 	print("Saving model %d" % (epoch))
				# 	self.model.save_weights('saved-models/' + str(epoch+1) + '.h5', overwrite=True)

		except KeyboardInterrupt:
			pass
		except Exception, e:
			print e

		self.model.save_weights('saved-models/cartpole.h5', overwrite=True)

	def test_model(self):
		'''Test The Model'''

		if os.path.exists('saved-models/cartpole.h5'):
			self.model.load_weights("saved-models/cartpole.h5")
		else:
			self.train_model()

		total_reward = 0
		for i in range(self.TEST_SIZE):
			state = self.env.reset()
			done = False
			c = 0
			while not done:
				# self.env.render()
				qval = self.model.predict(numpy.array([state]), batch_size=1)
				action = numpy.argmax(qval)
				new_state, reward, done, _ = self.env.step(action)
				total_reward += reward
				state = new_state
				c += 1
			print c,
		avg_reward = total_reward / self.TEST_SIZE
		print("\nAvg: %.5f" % (avg_reward))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("func", default="train")
	args = parser.parse_args()

	game = DQN_CartPole()

	if args.func == "train":
		game.train_model()
	else:
		game.test_model()







