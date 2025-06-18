import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import csv
import torch
from belief_network_util import * 

from util import random_walk

from sklearn.preprocessing import normalize
from belief_network_util import create_dataset

import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt 
import random
import csv
import torch

from sklearn.preprocessing import normalize
from belief_network_util import create_dataset

class BeliefWindowBuffer():
	"""Class to control the buffer which contains the current window of observations and actions to be used in the belief
	network's prediction of current timepoint's gradient and trajectory error. 
	"""
	def __init__(self, observation_space, action_space, buffer_length):
		self.observation_space = observation_space
		self.action_space = action_space
		self.buffer_length = buffer_length

		# initialize to all 0's. When using just update (append and remove) 1 element at a time
		self.observation_v = []
		self.action_v = []
		self.observed_error_v = []
		self.observed_kspace_v = []

		for iter in range(self.buffer_length):
			self.observation_v.append(np.zeros(self.observation_space.shape[0]))
			self.action_v.append(np.zeros(self.action_space.shape[0]))
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
			action_v_holder = np.zeros(self.action_space.shape[0])
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
			self.observation_v.append(np.zeros(self.observation_space.shape[0]))
			self.action_v.append(np.zeros(self.action_space.shape[0]))
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





class GradEnvContBelief(gym.Env):
	"""Custom Environment for modeling gradient waveforms that follows gym interface. Continuous action space.
	Belief system for updating the error 

	Note that this problem is in general partially observable. When a pulse is played out on the scanner, the agent
	in general does not have much information about the actual, current error between the ideal waveform and 
	the "real" waveform at a given timepoint - the only real feedback is the reward that each timepoint achieves at
	the end of the episode. So we need to think about belief, and the partial observability of the problem.

	In general, the class of problem is OFF POLICY and PARTIALLY OBSERVABLE. It would be a "standard" POMDP as defined
	by Ni et. all ICML 2022. 


    ### Observation Space

    The observation is a `ndarray` with shape `(3,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | Current gradient amplitude			 | -Inf | Inf | T/m 		 |
    | 1   | Gradient slew rate (velocity)        | -Inf | Inf | T/m/s        |
	| 2   | Gradient acceleration rate           | -Inf | Inf | T/m/s^2      |

	The ACTUALLY OBSERVED amplitude, slew, and acceleration are from GIRF(g + gprecomp). g is fixed, 
	gprecomp can vary based on action. 

	### Action Space

    The action space is continuous. It is Box(-1.0, 1.0, (1,), float32)

    ### Transition Dynamics
	
	The transition dynamics are applied to a superimposed system: The underlying gradient waveform and the 
	designed preemphasis. Initially, the premphasis is 0. 

    Given an action, the gradients follow the following transition dynamics: 

	*amplitude<sub>t+1</sub> = amplitude<sub>t</sub> + slew<sub>t+1</sub>*

	*slew<sub>t+1</sub> = slew<sub>t</sub> + (action - 1) * force 


	### Reward:

    The goal is to provide a precompensation such that the gradients produce the desired waveform in the presence
    of a nonideal impulse response (ideal = a delta function). The agent is provided a reward for each step that its
    net value is with a certain tolerance of the desired waveform, within a certain tolerance. 

	### Starting State

	The gradients are initialized "off". Amplitude, slew and acceleration are 0. 
	Belief network is pretrained on an undistorted set of gradient waveforms, then updated. 

	## Episode End

	The episode ends if either of the following happens:
	1) Truncation: The end of the pulse is reached.
	2) Constraint reached: gradient amplitude/slew/acceleration greater than some value. However, at least 
		initially, we probably want to allow the agent to explore constraint-violating action sequences

	"""


	metadata = {"render_modes": ["human"], "render_fps": 30}


	def __init__(self,
				g_ideal_m,
				scanner_girf,
				scanner_girf_t,
				gradient_model,
				belief_model=None, 
				belief_lookback=40, 
				render_mode="human", 
				normalize=False,
				gmax=np.inf,
				smax=np.inf,
				kmax=np.inf,
				apply_constraint=True,
				randomize_waveform=True,
				max_amplitude_range=[40, 100],
				dt=4e-6,
				):
			# gmax = maximum gradient amplitude. Scalar.
			# smax = maximum slew rate. Scalar.
			# girf = gradient impulse response function. Vector.
			# max_amplitude_range: the scaling factors applied to normalized waveforms, [mT/m] 
			super(GradEnvContBelief, self).__init__()
           
			# initialize variables. Initial pulse is ideal g desired 
			# g_ideal_m waveforms are all normalized! 
			self.g_ideal_m = g_ideal_m

			self.dt = dt # dwell time, [s]
			
			# just initialize with the first pulse in the dataset 
			self.num_distinct_waveforms = len(self.g_ideal_m)
			self.waveform_index = 0
			self.g_ideal = self.g_ideal_m[self.waveform_index][0] # UNNORMALIZED, [mT/m]

			# check the maximum of both amplitude and slew across all provided waveforms: 
			max_input_g = 0
			max_input_s = 0
			for ii in range(self.num_distinct_waveforms):
				waveform_max_g = np.amax(np.abs(self.g_ideal_m[ii][0]))
				waveform_max_s = np.amax(np.abs(np.diff(self.g_ideal_m[ii][0])))
				if waveform_max_g > max_input_g:
					max_input_g = waveform_max_g
				if waveform_max_s > max_input_s:
					max_input_s = waveform_max_s

			self.s_ideal = np.diff(self.g_ideal)
			self.a_ideal = np.diff(self.s_ideal)
			self.t = 0 # 0 timepoint
			self.tmax = self.g_ideal.size-1
			self.kout_precomp_t = 0
			self.kout_noprecomp_t = 0

			self.scanner_girf = scanner_girf 
			self.scanner_girf_t = scanner_girf_t # time vector associated with the girf 
			self.g_history = [] # variable to have memory of played out waveform
			self.g_history_noprecomp = [] # variable to have memory of played out waveform
			self.action_history = [0] # variable to have memory of acceleration actions chosen
			self.precomp_amp_history = [0]
			self.belief_error_record = []
			self.actual_error_record = []
			self.state_history = []
			self.normalize = normalize

			# choose to randomize the sequence of waveforms or not
			self.randomize_waveform = randomize_waveform
			self.max_amplitude_range = max_amplitude_range

			# scale the normalized waveforms by some amount in the specified range.
			# This gives unscaled units of [mT/m]
			self.g_ideal_random_scale = random.uniform(self.max_amplitude_range[0], 
														self.max_amplitude_range[1])

			self.gradient_model = gradient_model

			self.force = 15 

			# This variable is to track kspace. If 
			self.k_baseline = 0

			# observation space: g, slew, acceleration. Limits size of observation space vs allowing to be infinite
			# additionally, any deviation exceeding these bounds is considered unacceptable error, and system should stop.
			self.gmax = gmax
			self.smax = smax
			self.kmax = kmax
			self.apply_constraint = apply_constraint

			# STATE SPACE: (ideal_amp, ideal_slew, precomp_slew, precomp_amp, belief_error)
			if self.normalize:
				self.low_state = np.array([-1, -1, -1, -1, -1], dtype=np.double)
				self.high_state = np.array([1, 1, 1, 1, 1], dtype=np.double)
			else:
				self.low_state = np.array([-self.gmax, -self.smax, -self.smax, -self.gmax, -self.gmax], dtype=np.double)
				self.high_state = np.array([self.gmax, self.smax, self.smax, self.gmax, self.gmax], dtype=np.double)

			# OBSERVATION SPACE: same dimension as state space
			self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.double)

			# Continuous implementation of action space. 3 options
			self.action_space = spaces.Box(
			    low=-1, high=1, shape=(1,), dtype=np.double
			)

			# belief network parameters
			self.belief_lookback = belief_lookback
			self.belief_training_buffer_size = int(1e4)
			self.belief_model = belief_model #preconstructed, and ideally pretrained

			# just reset the whole environment upon initialization
			self.reset()

			# initialize the belief lookback buffer, which will control what gets fed into the belief prediction network 
			self.BeliefWindowBuffer = BeliefWindowBuffer(self.observation_space, self.action_space, self.belief_lookback)
			print(f'belief window buffer initialized error_v = {self.BeliefWindowBuffer.observed_error_v}')

			# initialize a buffer to store transitions for training the belief network
			#TODO: specify the length of this buffer
			self.BeliefTrainingBuffer = BeliefWindowBuffer(self.observation_space, self.action_space, self.belief_training_buffer_size)

	def step(self, action, write_to_csv=False, csv_path=None):
		# A NOTE ON THE VARIABLE CONVENTION INSIDE THIS FUNCTION
		# I was getting confused as to what variable corresponds to the previous timepoint
		# Versus the current time point - either side of the transition so:
		# _tm1 variables refer to the previous timepoint (t-1),
		# _t variables refer to the current timepoint (t)
		# _tp1 variables refer to the next timepoint (t+1)

		truncated, terminated = False, False

		# array form, just get the scalar
		action = action[0]
		self.action_history.append(action)

		# get the state - what the amplitude, velocity of the previous timestep were. Also what currently demanded
		# input sample is
		last_timestep_error, last_timestep_slew, precomp_velocity_tm1, precomp_amplitude_tm1, belief_error = self.state

		# TODO: get the last precomp amplitude and the current gin amplitude(where we are in the track)
		precomp_amplitude_tm1 = self.precomp_amp_history[-1]
		self.gin_samp_t = self.g_ideal[self.t]
		gin_slew_t = self.s_ideal[self.t]
		# Unnormalize the amplitudes
		if self.normalize:
			precomp_velocity_tm1 *= self.smax

		# TODO: this is sloppy. But apply whatever the global scale of the current pulse is here
		self.gin_samp_t = self.g_ideal_random_scale*self.gin_samp_t

		# Here, action IS scaled acceleration. 
		# update velocity by acceleration
		precomp_velocity_t = precomp_velocity_tm1 + action * self.force
		precomp_velocity_t = np.squeeze(precomp_velocity_t)

		# We want to clip the velocity if excessive
		# check constraints on velocity and impose 
		if precomp_velocity_t > self.smax:
			truncated = True
			precomp_velocity_t = self.smax
		if precomp_velocity_t < -self.smax:
			truncated = True
			precomp_velocity_t = -self.smax

		# update precomp amp by velocity
		precomp_amplitude_t = precomp_amplitude_tm1 + precomp_velocity_t
		self.precomp_amp_history.append(precomp_amplitude_t)
		net_amplitude_t = self.gin_samp_t + precomp_amplitude_t
        		
        # We want to clip the acceleration if excessive
		# check constraints on amplitude and impose
		if self.apply_constraint:
			if net_amplitude_t > self.gmax:
				truncated = True
				net_amplitude_t = self.gmax
			if net_amplitude_t < -self.gmax:
				truncated = True
				net_amplitude_t = -self.gmax

		# update the "actual waveform" history. Append sum of ideal waveform + precomp:
		self.g_history.append(net_amplitude_t)
		self.g_history_noprecomp.append(self.gin_samp_t)

		# Calculate the net output gradient waveform: distortion of (ideal + precomp):
		# also calculate the net output gradient waveform of just the 
		# ALSO calculate where in kspace we are at this sample (assumes start @0 @beginning)
		self.net_amplitude_distorted_t, self.kout_precomp_t = self.apply_gradient_model(self.g_history, model=self.gradient_model)
		self.noprecomp_amplitude_distorted_t, self.kout_noprecomp_t = self.apply_gradient_model(self.g_history_noprecomp, model=self.gradient_model)

		# IF just using last sample of previous pulse as kspace baseline for next, add to kspace
		# baseline is 0 by default
		self.kout_precomp_t += self.k_baseline
		self.kout_noprecomp_t += self.k_baseline

		# increment time. Terminate episode if end of pulse is reached
		self.t += 1
		if self.t >= self.tmax:
			terminated = True
		else:
			terminated = False

		# calculate the error at this timestep
		self.delta_g_err = (self.net_amplitude_distorted_t - self.gin_samp_t)

		# update the buffers 
		action_out = np.zeros(self.action_space.shape[0])
		observed_error_out = action_out
		action_out[0]= action
		observed_error_out[0] = self.delta_g_err
		observed_error_out = self.get_current_observed_error(normalize=True)

		self.BeliefWindowBuffer.add(self.state, action_out, observed_error_out, self.kout_precomp_t)
		self.BeliefTrainingBuffer.add(self.state, action_out, observed_error_out, self.kout_precomp_t)

		# PREDICT the error at this timestep, from a sequence of length lookback

		if self.belief_model == None:
			belief_error = 0
		else:
			_, X_test, _, y_test = self.BeliefWindowBuffer.return_X_Y_train_test(train_test_split=0, normalize=True, predicting='kspace', rl_env=self)
			X_test, y_test = create_dataset(X_test, y_test, lookback=self.belief_lookback)
			belief_model_local=self.belief_model

			self.belief_model.eval()

			# WITHIN PREDICTION: ALL NORMALIZED
			with torch.no_grad():
				belief_error = np.ones_like(y_test)*0
				belief_error = belief_model_local(X_test)[:,-1,:]

				belief_error = belief_error[-1] 

			# print(f'timestep {self.t}')
			# print(f'Belief error = {belief_error*self.gmax}')
			self.belief_error_record.append(belief_error.item()*self.gmax)
			self.actual_error_record.append(self.delta_g_err)

			belief_error = belief_error.item()

		if terminated:
			self.state = [0.0, 0.0, 0.0, 0.0, 0.0]
			self.state = np.squeeze(np.array(self.state))
		else:
			if self.normalize:
				self.state = [self.g_ideal[self.t]/self.gmax, self.s_ideal[self.t]/self.smax,
								 precomp_velocity_t/self.smax, precomp_amplitude_t/self.gmax, belief_error/self.gmax]
				self.state = np.squeeze(np.array(self.state, dtype=np.double))
			else:
				self.state = [self.gin_samp_t, gin_slew_t, precomp_velocity_t, precomp_amplitude_t, belief_error]
				self.state = np.squeeze(np.array(self.state, dtype=np.double))

		# write state and action to csv file if doing so
		if write_to_csv:
			with open(csv_path, 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, dlimiter=',')
				writer.writerow(self.state)


		# calculate the reward associated with the error: Follows Cao et. al Path Following

		#c_e = 0.15 # constant determining sparsity of reward (shape of exponential). Larger = sparser rewards
		# reduce reward sparsity
		c_e = 0.15
		c_H = 10 # constant determing reward for successful completion
		c_F = 100 # constant determining reward for truncation failure
		c_dacc = 0.0075 # constant determining penalty for too much amplitude
		c_dslew = 0.00025 # constant determining penalty for too much slew
		c_adiff = 0.005 # constant determining the penalty for too quick change in acceleration

		# idea: penalize the derivative of the acceleration.

		reward_error = np.exp(-c_e*np.abs(self.delta_g_err))

		# Give reward just for "surviving" - not hitting the bounds of the system
		reward_survive = 0.1
		if terminated:
			reward_survive += c_H
		if truncated:
			reward_survive += -c_F*((self.tmax-self.t)/self.tmax) # discount this (-) reward for surviving longer

		# reward low effort
		reward_effort_acceleration = -c_dacc * np.abs(action * self.force)
		reward_effort_acc_diff = -c_adiff * np.abs(self.action_history[self.t-1]-self.action_history[self.t])*self.force
		reward_effort = reward_effort_acceleration + reward_effort_acc_diff

		reward = reward_error + reward_survive + reward_effort


		# optionally can pass additional info, we are not using for now
		info = {}


		return self.state, reward, terminated, truncated, info

	def calc_residual(self, g):

		residual = np.abs(self.g_ideal - self.g)
		return residual

	def apply_gradient_model(self, gin, model='linear'):
		# applies whatever the gradient model is, and returns the gradient and kspace sample that is
		# produced by the distorting model

		# girf assumes that gin is the whole history of played out gradient waveforms
		if model == 'linear':
			# This is a super simple linear distortion. Our output is simply a scaled version of our input
			scalefact = 0.75
			gout = gin[-1] * scalefact
		elif model == 'girf':
			conv_result = np.convolve(np.array(self.g_ideal*self.g_ideal_random_scale), self.scanner_girf)
			conv_result_k = np.cumsum(conv_result/1000)*42580*self.dt
			t0 = np.where(self.scanner_girf_t==0)[0] 

			conv_result = conv_result[int(t0)-1:] # trim the initial convolution pad
			conv_result_k = conv_result_k[int(t0)-1:]

			gout = np.squeeze(conv_result[self.t])
			kout = np.squeeze(conv_result_k[self.t])

		elif model == 'ideal':
			# just return the unmodified input - an ideal girf
			gout = gin[-1]
		else:
			print('invalid girf chosen. passing ideal output ')
			gout = gin[1]

		return gout, kout

	def plot_belief_error(self):
		plt.plot(np.array(self.belief_error_record))
		plt.plot(np.array(self.actual_error_record))
		plt.legend(['belief error', 'actual_error'])
		plt.show()



	def reset(self, seed=None, options=None, add_noise=False, set_k_baseline_0=False):
		"""
		Important: the observation must be a numpy array
		:return: (np.array)
		"""
		super().reset(seed=seed, options=options)

		# get the new gin
		self.g_ideal = self.g_ideal_m[self.waveform_index][0]

		self.t = 0
		self.tmax = self.g_ideal.size-1

		# set reference kspace for next pulse to 0, unless considering these pulses to be
		# played out back to back, in which last sample is used as baseline for next
		if set_k_baseline_0:
			self.k_baseline = 0
		else:
			self.k_baseline = self.kout_precomp_t

		if add_noise:
			# superimpose a random walk on the underlying waveform
			self.g_ideal = self.g_ideal + 0.005*random_walk(self.tmax+1,N=50)
			# plt.plot(self.g_ideal)
			# plt.show()
		self.s_ideal = np.diff(self.g_ideal)


		self.belief_error_record = []
		self.actual_error_record = []

		# to diversify training data, use a random scale/shift on gradient waveform desired
		# Just doing a positive random scale. Flipped waveforms are already included in waveform dataset, so no need
		# scale units are in mT/m
		self.g_ideal_random_scale = random.uniform(self.max_amplitude_range[0], 
														self.max_amplitude_range[1])
		self.g_ideal_max = max(np.abs(self.g_ideal_random_scale * self.g_ideal))

		# reset history variables
		self.g_history = [] # variable to have memory of played out waveform
		self.g_history_noprecomp = [] # variable to have memory of played out waveform
		self.action_history = [0]
		self.precomp_amp_history = [0]
		self.state_history = []

		# update the new waveform index for the next reset
		if self.randomize_waveform:
			self.waveform_index = random.randint(0,self.num_distinct_waveforms-1)
		else:
			self.waveform_index = self.waveform_index + 1

		# reset state information
		self.t = 0
		self.tmax = self.g_ideal.size-1
		precomp_amplitude = 0
		precomp_velocity = 0
		gin_samp = self.g_ideal[0]
		self.state = np.squeeze([0, 0, 0, 0, 0]) # no error, no velocity, 0 error

		# reset returns observation, info
		return (np.array(self.state, dtype=np.double), {})

	def get_random_scale(self):
		return self.g_ideal_random_scale

	def get_current_observed_error(self, normalize=True):
		# units are mT/m
		if normalize:
			out = self.delta_g_err/self.gmax
		else:
			out = self.delta_g_err
		return out

	def get_current_observed_kspace(self):
		out = self.kout_precomp_t
		return out


	def set_new_belief_model(self, belief_model):
		self.belief_model = belief_model

# Functions for Belief network system and control. 
	def predict_current_error(self):
		print("TODO: unimplemented")


	def close(self):
		pass