import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import csv
import torch
from belief_network_util import * 

from belief_network_util import create_dataset

import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt 
import random
import csv
import torch

class BeliefWindowBuffer():
	"""Class to control the buffer which contains the current window of observations and actions to be used in the belief
	network's prediction of current timepoint's gradient and trajectory error. 
	"""
	def __init__(self, observation_size, action_size, buffer_length):
		self.observation_size = observation_size
		self.action_size = action_size
		self.buffer_length = buffer_length

		# initialize to all 0's. When using just update (append and remove) 1 element at a time
		self.observation_v = []
		self.action_v = []
		self.observed_error_v = []
		self.observed_kspace_v = []

		for iter in range(self.buffer_length):
			self.observation_v.append(np.zeros(self.observation_size))
			self.action_v.append(np.zeros(self.action_size))
			self.observed_error_v.append(0)
			self.observed_kspace_v.append(0)

	def add(self, observation, action, observed_error, observed_kspace, index=-1):
		# quick check if type of observed_error is numpy array. SHOULD be scalar
		if type(observed_error).__module__ == 'numpy':
			observed_error = observed_error.item()
		if type(observed_kspace).__module__ == 'numpy':
			observed_kspace = observed_kspace.item()

		if index == -1:
		# put a new observation and action in the buffer, and remove the oldest entry
		# add to end and remove 0th entry
			self.observation_v.append(observation)
			self.action_v.append(action)
			self.observed_error_v.append(observed_error)
			self.observed_kspace_v.append(observed_kspace)

			self.observation_v.pop(0)
			self.action_v.pop(0)
			self.observed_error_v.pop(0)
			self.observed_kspace_v.pop(0)
		else:
			# add at specifed index
			self.observation_v[index] = observation
			action_v_holder = np.zeros(self.action_size)
			action_v_holder[0] = action[0][0]
			self.action_v[index] = action_v_holder
			self.observed_error_v[index] = observed_error
			self.observed_kspace_v[index] = observed_kspace

	def add_transitions_from_csv(self, file_path):
		# Does not normalize at all. Simply adds CSV data
		# make new belief model object
		num_lines = pd.read_csv(file_path)

		# reset the buffer
		self.buffer_length = len(num_lines)
		self.reset()
		    
	    # add all of the data to the buffer: 
		with open(file_path) as file:
			reader_obj = csv.reader(file)
			index = 0
			for row in reader_obj:
				# add at specifed index
				# observations: Gamp, Gslew, precomp amp, precomp slew, scale_factor,
				all_observations = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[7])]
				self.observation_v[index] = all_observations
				self.action_v[index] = float(row[4])
				self.observed_error_v[index] = float(row[5])
				self.observed_kspace_v[index] = float(row[6])

				index += 1

	def reset(self):
		# initialize to all 0's. When using just update (append and remove) 1 element at a time
		self.observation_v = []
		self.action_v = []
		self.observed_error_v = []
		self.observed_kspace_v = []

		for index in range(self.buffer_length+1):
			self.observation_v.append(np.zeros(self.observation_size))
			self.action_v.append(np.zeros(self.action_size))
			self.observed_error_v.append(0)
			self.observed_kspace_v.append(0)

	def return_X_Y_train_test(self, train_test_split, up_to_index=None, normalize=False, rl_env=None, feature_indices=[0,3]):
		# return all of the recorded transitions as a single long timeseries
		# This is 0:4 on indices of observation, because we want to EXCLUDE the error belief in the training,
		# even though we will use that as a state in the RL training 

		observation_array = np.array(self.observation_v)
		observation_features = feature_indices # just the ideal and precomp amplitude [0, 3]
		observation_array = observation_array[:,observation_features]
		if normalize:
			observation_array = observation_array/rl_env.gmax
		action_array = np.array(self.action_v)
		if action_array.ndim == 1:
			action_array = np.expand_dims(action_array,1)

		# features include both observations (amp and slew of precomp and ideal) and actions
		all_x_data = np.concatenate((observation_array, action_array),1)

		observed_error_array = np.array(self.observed_error_v).astype('double')

		# normalize. Actions are already between [-1, 1]. Just want to divide by max amp, slew. 
		if normalize:
			observed_error_array = observed_error_array/rl_env.gmax

		# finally, split all into train and test datasets. If specify index, use that. Otherwise,
		# do by train/test split %
		if up_to_index is None and train_test_split is not None:
			train_size = int(len(self.observed_error_v)*train_test_split)
			test_size = len(self.observed_error_v) - train_size

			X_train = all_x_data[:train_size,:]
			X_test = all_x_data[train_size:,:]

			Y_train = np.expand_dims(observed_error_array[:train_size],1)
			Y_test = np.expand_dims(observed_error_array[train_size:],1)


		elif up_to_index is not None and train_test_split is None:
			train_size = up_to_index
			test_size = len(self.observed_error_v) - train_size
			X_train = all_x_data[:train_size,:]
			X_test = all_x_data[train_size:,:]

			Y_train = np.expand_dims(observed_error_array[:train_size],1)
			Y_test = np.expand_dims(observed_error_array[train_size:],1)

		else:
			print("Should specify either up_to_index or train_test_split as None")

		return X_train, X_test, Y_train, Y_test

	def get_action_v_list(self):
		return self.action_v

	def get_observation_v_list(self):
		return self.observation_v

	def get_action_v_array(self):
		return np.array(self.action_v)

	def get_observation_v_array(self):
		return np.array(self.observation_v)

	def get_observation_v_tensor(self):
		#TODO: this is terrible and inefficient
		return torch.tensor(np.array(self.observation_v))
