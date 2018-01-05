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
initial_games = 300

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


def play_games(model):
	game_records = []
	scores = []
	for _ in range(initial_games):
		score, game_memory = play.play(env = env, model = model, production = False)
		scores.append(score)
		game_records.append([score, game_memory])
					

	scores.sort(reverse = True)
	score_requirement = scores[int(len(scores)/3)]
	training_data = []		
	acceptable_scores = []
	for record in game_records:
		score = record[0]		
		game_memory = record[1]
		if score >= score_requirement:
			training_data = transform_actions(training_data, game_memory)	
			acceptable_scores.append(score)
	print("Average score", sum(scores)/len(scores))
	print("Average acceptable score", sum(acceptable_scores)/len(acceptable_scores))
	print("Training set length", len(training_data))
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
	
	
def train_model(training_data):
	X = numpy.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	Y = [i[1] for i in training_data]		
	model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=500, show_metric=False, run_id='openaistuff')	
	return model	
	
training_data = play_games(False)

model = tflearn.DNN(get_network_spec(input_size), tensorboard_dir='log')

model = train_model(training_data)	

training_data = play_games(model)

model = train_model(training_data)	

training_data = play_games(model)
	
	