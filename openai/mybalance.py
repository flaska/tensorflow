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
initial_games = 10000

def random_games():
	for episode in range(5):
		env.reset();
		for t in range(goal_steps):
			env.render() # performance killer
			action = env.action_space.sample() # takes random action
			observation, reward, done, info = env.step(action)
			print(observation)
			if (done):
				break
				
#random_games();

def initial_population():
	training_data = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		for _ in range(goal_steps)
			action = random.randrange(0,2)			
			observation, reward, done, info = env.step(action)
			game_memory.append([observation, action])
			score += reward
			if done:
				break
		if score >= score_requirement			
			training_data.append([data[0], output])
				
		env.reset()
		scores.append(score)
	
	training_data_save = np.array(training_data)
	
	np.save('saved.npy', training_data_save)
	
	print('Average accepted score', mean(accepted_scores))
	print('Median accepted score', median(accepted_scores))
	print(Counter(accepted_scores))
		