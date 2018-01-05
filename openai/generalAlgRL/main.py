import gym
import random
import numpy
import tflearn
import play
 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


LR = 1e-3
env = gym.make('CartPole-v0')
input_size = 4
goal_steps = 500
score_requirement = 50
initial_games = 100

def transform_actions(training_data, game_memory):
	for data in game_memory:
		observation = data[0]
		scalar_action = data[1]
		if scalar_action == 1:
			vector_action = [0,1]					
		elif scalar_action == 0:
			vector_action = [1,0]				
		training_data.append([data[0], vector_action])
	return training_data


def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score, game_memory = play.play(env = env, model = False, production = False)
		scores.append(score)
		if score >= score_requirement:
			accepted_scores.append(score)
			training_data = transform_actions(training_data, game_memory)

	
	# training_data_save = numpy.array(training_data)	
	# numpy.save('observation_action_responses.npy', training_data_save)
	
	print('Average accepted score', mean(accepted_scores))
	print('Median accepted score', median(accepted_scores))
	print(Counter(accepted_scores))
	
	return training_data
	

def get_network_spec(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)	

	network = fully_connected(network, 2, activation='softmax')
	
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
	
	return network
	
	
def train_model(training_data, model):
	X = numpy.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	Y = [i[1] for i in training_data]		
	model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')	
	return model
	
def play_games(model):
	scores = []
	for each_game in range(10):
		score, game_memory = play.play(env = env, model = model, production = False)
		scores.append(score)
	print('Average Score', sum(scores)/len(scores))
	
	
training_data = initial_population()

model = tflearn.DNN(get_network_spec(input_size), tensorboard_dir='log')

model = train_model(training_data, model)	
model.save('./model')

#model.load('./model')

play_games(model)


	
	