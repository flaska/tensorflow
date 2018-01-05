import gym
import random
import numpy

goal_steps = 500

def play(env, model):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range (goal_steps):
		env.render();
		if len(prev_obs) == 0 or model==False:
			action = random.randrange(0,2)
		else:
			action = numpy.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
		
		new_observation, reward, done, info = env.step(action)
		game_memory.append([prev_obs, action])
		score += reward
		prev_obs = new_observation
		#if done:
		#	break
	return score, game_memory