import gym
import random
import numpy
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset();
goal_steps = 500
score_requirement = 50
initial_games = 100

#def random_games():
#	for episode in range(5):
#		env.reset();
#		for t in range(goal_steps):
#			env.render() # performance killer
#			action = env.action_space.sample() # takes random action
#			observation, reward, done, info = env.step(action)
#			print(observation)
#			if (done):
#				break				
#random_games();

def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)
			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])
				
			prev_observation = observation
			score += reward
			if done:
				break
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				observation = data[0]
				scalar_action = data[1]
				if scalar_action == 1:
					vector_action = [1,0]					
				elif scalar_action == 0:
					vector_action = [0,1]
				
				training_data.append([data[0], vector_action])
				
		env.reset()
		scores.append(score)
	
	print(training_data)
	
	training_data_save = numpy.array(training_data)
	
	numpy.save('saved.npy', training_data_save)
	
	print('Average accepted score', mean(accepted_scores))
	print('Median accepted score', median(accepted_scores))
	print(Counter(accepted_scores))
	
	return training_data
	
initial_population();
		