from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import os
import shutil
import tempfile
import zipfile
import csv

import reverb
import copy

import tensorflow as tf
import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple
#from tf_agents.agents.dqn import dqn_agent
import dqn_epsilon_greedy as dqn_agent
import dqn_epsilon_greedy_shape as dqn_agent_shape

from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.environments import py_environment

import WaterSortEnv as wsp

'''
Functions used in the Watersort DQN RL
'''

###### The agent
'''
RL Agent - DQN
- This NN will use the Dense layer type
- The QNetwork is composed of sequential keras layers
- RL Agent chosen for this tutorial is the DQN

env: python environment
all_layers_param: Tuple contaning the list of parameters of each layer of the NN including the input layer :
    NumberOfElements(all_layers_param)==NumberOfLayers
    FistElement== Input Layer parameters List
    LastElement== Output Layer parameters List/ q_values_layer
    Each Element == Layer parameters List containing:
        Number of neurons
        Activation: instance of tf.keras.activations (or None)
        Kernel initializer: tf.keras.initializers (or default_kernel_initializer)
        Bias initializer: tf.keras.initializers (or default_bias_initializer)

The first output: the QNetwork (input to the DQN Agent)
The second output: the DQN Agent (RL model)
'''
#The QNetwork 
water_level =4

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      env: py_environment.PyEnvironment, 
      all_layers_params: List):
    """Initialize."""
    super().__init__()

    self.common = q_network_sequential_ac(all_layers_params)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    self.actor = tf.keras.layers.Dense(num_actions) 
    self.critic = tf.keras.layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)

def env_step(action: np.ndarray, env):
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32), env)


def tf_env_step(action: tf.Tensor,env) :
  return tf.numpy_function(env_step, [action, env], 
                           [tf.float32, tf.int32, tf.int32, env])

def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps, env) :
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done, env = tf_env_step(action,env)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards, env
  
def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor, huber_loss):
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

def get_expected_return(
    rewards: tf.Tensor,  max_steps: int) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward +  discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  #if standardize:
  returns = ((returns ) / 
               ( max_steps))

  return returns

@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    max_steps: int, 
    max_steps_per_episode: int, env, huber_loss):
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards, env = run_episode(
        initial_state, model, max_steps_per_episode, env) 

    # Calculate expected returns
    returns = get_expected_return(rewards, max_steps_per_episode)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns, huber_loss)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward, env

def q_network_conv1D(env,number_vials,cnn_activation_function1,cnn_activation_function2):
    model = [] #sequential.Sequential()
    model.append(tf.keras.layers.Conv1D(filters=4, kernel_size=5, activation=cnn_activation_function1, input_shape=(20,1)))
    #model.append(tf.keras.layers.Conv1D(filters=4, kernel_size=5, activation=cnn_activation_function1))
    model.append(tf.keras.layers.Dropout(0.2))
    model.append(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.append(tf.keras.layers.Flatten())

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    model.append(tf.keras.layers.Dense(num_actions*20, activation=cnn_activation_function1))
    model.append(tf.keras.layers.Dense(num_actions, activation=cnn_activation_function2))
    #model.compile(loss=cnn_loss_function, optimizer=cnn_optimizer, metrics=['accuracy'])
    q_net = sequential.Sequential(model)
    return q_net


def q_network_sequential(env,all_layers_params):
    q_net = sequential.Sequential(neural_network_dense(env,all_layers_params))
    return q_net

def q_network_sequential_ac(all_layers_params):
    q_net = sequential.Sequential(neural_network_dense_ac(all_layers_params))
    return q_net

def neural_network_dense(env,all_layers_params):
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    all_layers_params[len(all_layers_params)-1][0] = num_actions
    dense_layers = [dense_layer(layer_params) for layer_params in all_layers_params]
    return dense_layers

def neural_network_dense_ac(all_layers_params):
    dense_layers = [dense_layer(layer_params) for layer_params in all_layers_params]
    return dense_layers  
    
def dense_layer(layer_params):
    return tf.keras.layers.Dense(
      layer_params[0],
      activation=layer_params[1],
      kernel_initializer=layer_params[2],
      bias_initializer=layer_params[3])

#The DQN RL Agent

def DQN(env,q_net,optimizer,loss_function,train_step_counter,epsilon, observation_and_action_constrain_splitter):
    
    return dqn_agent.DqnEpsilonGreedyAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=observation_and_action_constrain_splitter,
        td_errors_loss_fn=loss_function,
        train_step_counter=train_step_counter,
        epsilon_greedy=epsilon
        )

def DQNS(env,q_net,optimizer,loss_function,train_step_counter,epsilon, observation_and_action_constrain_splitter):
    
    return dqn_agent_shape.DqnEpsilonGreedyAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=observation_and_action_constrain_splitter,
        td_errors_loss_fn=loss_function,
        train_step_counter=train_step_counter,
        epsilon_greedy=epsilon
        )


def DDQN(env,q_net,optimizer,loss_function,train_step_counter,epsilon, observation_and_action_constrain_splitter):
    #q_net_target=copy.deepcopy(q_net)
    return dqn_agent.DdqnEpsilonGreedyAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        #target_q_network=q_net_target,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=observation_and_action_constrain_splitter,
        td_errors_loss_fn=loss_function,
        train_step_counter=train_step_counter,
        epsilon_greedy=epsilon
        )

###### The evaluation: average returns
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  evaluation_list =[]
  #print("inside ....")
  for _ in range(num_episodes):
    #print("avg iteration ....")
    #print(_)
    time_step = environment.reset()
    #print(time_step)
    #print(time_step)
    
    episode_return = 0.0
    i=0

    evaluation_list.append([_,i,"null",time_step,episode_return])

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      #print([i,time_step.is_last()])
      episode_return += time_step.reward
      i+=1
      evaluation_list.append([_,i,action_step,time_step,episode_return])
      
      
    total_return += episode_return

  avg_return = total_return / num_episodes

  return [avg_return.numpy()[0], evaluation_list]

###### The experiance: replay buffer 
def replay_buffer_observer_reverb(agent,replay_buffer_max_length):
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)
    return [replay_buffer, rb_observer]

def replay_buffer_observer_reverb_ac(agent,replay_buffer_max_length):
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)
    return [replay_buffer, rb_observer, reverb_server]


def create_zip_file(dirname, base_filename):
  return shutil.make_archive(base_filename, 'zip', dirname)

# Save the policy, model in local storage
def save_model(agent,replay_buffer,checkpoint_dir,policy_dir,global_step,max_to_keep): #1
  
  train_checkpointer = common.Checkpointer(
      ckpt_dir=checkpoint_dir,
      max_to_keep=max_to_keep,
      agent=agent,
      policy=agent.policy,
      replay_buffer=replay_buffer,
      global_step=global_step
  )
  
  tf_policy_saver = policy_saver.PolicySaver(agent.policy)

  train_checkpointer.save(global_step)
  tf_policy_saver.save(policy_dir)
  #In the case we want to create compressed files out of the saved models adn policies
  checkpoint_zip_filename = ""#create_zip_file(checkpoint_dir, os.path.join(checkpoint_dir, 'exported_cp'))
  policy_zip_filename = ""#create_zip_file(policy_dir, os.path.join(policy_dir, 'exported_policy'))

  return [train_checkpointer, checkpoint_zip_filename, tf_policy_saver, policy_zip_filename]

def read_file_eval_results(file):
  my_data = np.genfromtxt(file, delimiter=", ")
  my_data =my_data.tolist()
  return my_data

def save_file_eval_results(file,eval_results):
  np.savetxt(file, 
                eval_results,
                delimiter =", ", 
                fmt ='%s')

###### Visualizations
def run_episodes(policy, eval_tf_env, eval_py_env):
  num_episodes = 10
  frames = []
  #state = [1,2,3,1,2,2,1,3,3,1,3,2,0,0,0,0,0,0,0,0]
  #capacity = [4,4,4,0,0]
  for _ in range(num_episodes):
    time_step = eval_tf_env.reset()
    #eval_tf_env.set_state(state,capacity)
    #frames.append(eval_py_env.render())
    episode_return =0
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      episode_return += time_step.reward
      #print(time_step)
    print(" return episode ", [_,episode_return])
    #frames.append(eval_py_env.render())
    #gif_file = io.BytesIO()
    #imageio.mimsave(gif_file, frames, format='gif', fps=60)
    #IPython.display.display(embed_gif(gif_file.getvalue()))

# Videos
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def create_policy_eval_video(eval_env,eval_py_env,policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
  return embed_mp4(filename)

def create_policy_eval_video_actions(eval_env,eval_py_env,policy, filename, states,num_episodes=5, fps=30):
  #filename = filename + ".mp4"
  #with imageio.get_writer(filename, fps=fps) as video:
  actions =[]
  for i in range(num_episodes):
    actions_now=[]
    time_step = eval_env.reset()
    #video.append_data(eval_py_env.render(states[i]))
    while not time_step.is_last():
      action_step = policy.action(time_step)
      #print(action_step.action.numpy()[0])
      actions_now.append(action_step.action.numpy()[0])
      time_step = eval_env.step(action_step.action)
      #video.append_data(eval_py_env.render())
    actions.append(actions_now)
  return actions
#Plot
def save_eval_plot(plot_file,x,y,legends,top=250):
        fig=plt.figure()
        plt.ion()
        for i in range (0,len(x)):
          plt.plot(x[i],y[i])
        if len(legends) >0 :
          plt.legend(legends)
        plt.title("Evaluation of Average returns per iterations")
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        #plt.ylim(top=top)
        #plt.show()
        plt.savefig(plot_file)

###### Training Process
def train(steps_file,agent,train_py_env, eval_env, buffer_observer, iterator, num_iterations,num_eval_episodes,collect_steps_per_iteration,eval_interval, log_interval):
            # (Optional) Optimize by wrapping some of the code in a graph using TF function.
            #agent.train = common.function(agent.train)

            # Reset the train step.
            #print("train 1 : ")
            agent.train_step_counter.assign(0)

            # Evaluate the agent's policy once before training.
            #print("train 1 : average")
            avg_return, steps = compute_avg_return(eval_env, agent.policy, num_eval_episodes)

            with open(steps_file, "a", newline="") as f:
              writer = csv.writer(f)
              writer.writerows(steps)

            returns = [avg_return]
            #p#rint("return : ")
            #print(returns)

            # Reset the environment.
            time_step = train_py_env.reset()

            # Create a driver to collect experience.
            collect_driver = py_driver.PyDriver(
                train_py_env,
                py_tf_eager_policy.PyTFEagerPolicy(
                agent.collect_policy, use_tf_function=True),
                [buffer_observer],
                max_steps=collect_steps_per_iteration)
            #print("driver done : ")
            for _ in range(num_iterations):
                #print("iteration n :")
                #print(_)
                # Collect a few steps and save to the replay buffer.
                time_step, _ = collect_driver.run(time_step)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss

                step = agent.train_step_counter.numpy()

                if step % log_interval == 0:
                    print('step = {0}: loss = {1}'.format(step, train_loss))

                if step % eval_interval == 0:
                    avg_return, steps = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                    with open(steps_file, "a", newline="") as f:
                      writer = csv.writer(f)
                      writer.writerows(steps)
                    print('step = {0}: Average Return = {1}'.format(step, avg_return))
                    returns.append(avg_return)
            return [agent, returns, eval_env] 


###### For action splitting
def observation_action_splitter(obs):
  return (obs['observation'], obs['valid_actions'])

