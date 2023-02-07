from __future__ import absolute_import, division, print_function

import base64
import csv
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

import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import policy_saver
#from tf_agents.policies import random_tf_policy
import masked_random_tf_policy as random_tf_policy


from tf_agents.policies import epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import copy
########################################################################
'''
Hyperparameters and other variables
'''

#The experience / buffer / file saver
initial_collect_steps = 100#20 # 
replay_buffer_max_length = 100000 #1000#10000  #
batch_size = 64 #16#32# 

#tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())
tempdir = "/home/ubuntu/workspace/results"
tempdir = os.path.join(tempdir, 'water-newdeep-masked-reward-actions-5')


#The training
num_iterations = 10000#20000#1000#2000 
learning_rate = 1e-3 
num_eval_episodes = 10  
collect_steps_per_iteration =  1
log_interval = 20#200#200#25#50 # 
eval_interval = 500#1000#1000#50#100# 

#The DQN Agent

loss_function=  common.element_wise_squared_loss#common.element_wise_huber_loss
train_step_counter = tf.Variable(0)


num_vials = 5
#The Q_network
default_param_value=None
default_kernel_initializer=tf.keras.initializers.HeUniform()
default_bias_initializer=tf.keras.initializers.Zeros

activation_hl=tf.keras.activations.relu

kernel_initializer_hl= tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal')
kernel_initializer_output=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03)
bias_initializer_output=tf.keras.initializers.Constant(-0.2)

all_layers_params = ([num_vials*(num_vials-1)*25,activation_hl,default_kernel_initializer,default_bias_initializer],
                    [num_vials*(num_vials-1)*10,activation_hl,default_kernel_initializer,default_bias_initializer],
                    [num_vials*(num_vials-1),default_param_value,default_kernel_initializer,bias_initializer_output])



import WaterSortAllEnv as wsp
import game_water_utils as cpu

def wsp_dqn_learning(params):
        prefix = params[0]
        env_values = params[1]
        num_iterations = params[2]
        learning_rate = params[3]
        num_eval_episodes = params[4]
        collect_steps_per_iteration = params[5]
        eval_interval = params[6]
        replay_buffer_max_length = params[7]
        batch_size = params[8]
        '''
        Environment
        '''
        #Initialization
        '''env_name = 'CartPole-v0'
        env = suite_gym.load(env_name)
        train_py_env = suite_gym.load(env_name)
        eval_py_env = suite_gym.load(env_name)
        '''
        env =  wsp.WaterSortEnv(num_vials,number_empty_vials=2,water_level=4,max_steps = 500, initial_state_reuse=env_values[1], reward_type=env_values[2], max_number_initial_state_reuse=env_values[3])

        train_py_env = wsp.WaterSortEnv(num_vials,number_empty_vials=2,water_level=4,max_steps = 500, initial_state_reuse=env_values[1], reward_type=env_values[2], max_number_initial_state_reuse=env_values[3])
        eval_py_env = wsp.WaterSortEnv(num_vials,number_empty_vials=2,water_level=4,max_steps = 500, initial_state_reuse=env_values[1], reward_type=env_values[2], max_number_initial_state_reuse=env_values[3])

        #TF Conversion
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        '''
        print(env.time_step_spec().observation)
        print('Reward Spec:')
        print(env.time_step_spec().reward)
        print('Action Spec:')
        print(env.action_spec())
        time_step = env.reset()
        print('Time step:')
        print(time_step)

        action = np.array(1, dtype=np.int32)

        next_time_step = env.step(action)
        print('Next time step:')
        print(next_time_step)
        '''
        ##########DQN Agent
        
        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        q_net = cpu.q_network_sequential(env,all_layers_params)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        agent = cpu.DQN(train_env,q_net,optimizer,loss_function,train_step_counter,env_values[0],cpu.observation_action_splitter)
        agent.initialize()

        '''
        ##########Policy
        print(eval_policy)
        print(collect_policy)
        policy = copy.deepcopy(agent.collect_policy)
        agent.policy.update(epsilon_greedy_policy.EpsilonGreedyPolicy(policy,epsilon=0.9))
        eval_policy = agent.policy
        print(eval_policy)
        '''
        eval_policy = agent.policy #Greedy Policy 
        collect_policy = agent.collect_policy #GEpsilon reedy Policy 
        #agent.policy = copy.deepcopy(agent.collect_policy)
        #print (train_env.time_step_spec())
        random_policy = random_tf_policy.MaskedRandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec(),
                                                        observation_and_action_constraint_splitter = cpu.observation_action_splitter
                                                     )

        #print(eval_policy)
        
        #print (train_env.time_step_spec())
        
        ##########Evaluation
        

        print(cpu.compute_avg_return(eval_env, random_policy, num_eval_episodes))

        
        ##########Experience: Replay Buffer
        
        replay_buffer, rb_observer = cpu.replay_buffer_observer_reverb(agent,replay_buffer_max_length)
        print("after rewardddddddddddddddddd")
        py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(
            random_policy, use_tf_function=True),
            [rb_observer],
            max_steps=initial_collect_steps).run(train_py_env.reset())
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        iterator = iter(dataset)
        print("before trainnnnnnnnnnnnnnnnnnnnnnn")
        
        ##########ML Approach
        #tSaving state
        file_step =prefix
        tempdir_it = os.path.join(tempdir, file_step)
        policy_dir = os.path.join(tempdir_it, 'policy')
        checkpoint_dir = os.path.join(tempdir_it, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(policy_dir, exist_ok=True)

        steps_file = tempdir_it + "/steps.csv"
        #training
        agent, returns, eval_env  = cpu.train(steps_file,agent,train_py_env, eval_env, rb_observer, iterator, num_iterations,num_eval_episodes,collect_steps_per_iteration,eval_interval,log_interval)
        
 
        

        train_checkpointer, checkpoint_zip_filename, tf_policy_saver, policy_zip_filename = cpu.save_model(agent,replay_buffer,checkpoint_dir,policy_dir,train_step_counter,1)
        returns_file = tempdir_it + "/returns.csv"
        
        cpu.save_file_eval_results(returns_file,returns)   

        ############# To reload the state
        #train_checkpointer.initialize_or_restore()
        #global_step = tf.compat.v1.train.get_global_step()
        #saved_policy = tf.saved_model.load(policy_dir)
        
        #file_step ="T" + str(stepping)
        #tempdir_it = os.path.join(tempdir, file_step)
        #returns_file = tempdir_it + "/returns.csv"
        #returns = cpu.read_file_eval_results(returns_file)
            
        
        ##########Visualization
        
        #plot
        plot_file = tempdir_it +"/avgReturns.png"
        iterations = range(0, num_iterations + 1, eval_interval)
        legends=[]
        legends.append('Avg')
        cpu.save_eval_plot(plot_file,[iterations], [returns], legends)
        
        
        '''
        #Video
        video_file1 = os.path.join(tempdir_it, 'trained-agent')
        video_file2 = os.path.join(tempdir_it, 'random-agent')
        cpu.create_policy_eval_video(eval_env,eval_py_env,agent.policy, video_file1)
        cpu.create_policy_eval_video(eval_env,eval_py_env,random_policy, video_file2)  
        '''

'''
num_iterations_list = [50000,500000,1000000]
learning_rate_list = [1e-4,1e-2,1e-1]
num_eval_episodes_list = [5,10,50]
collect_steps_per_iteration_list =[5,10]
eval_interval_list =[10,50,500]
replay_buffer_max_length_list=[10000,100000]
batch_size_list=[32,128]

epsilon_list=[0.1,0.5,0.7,0.9,1] #0.0,0.05,

initial_state_reuse_list=[0.0,0.1,0.2,0.5,0.7,0.9,1]

max_number_initial_state_reuse_list=[10,50,100,500,1000]
'''
reward_type_list=[0,1,2,3,4] #4 valid/invalid

num_iterations_default = 200000#20000#1000#2000 
learning_rate_default = 1e-3 
num_eval_episodes_default = 100  
collect_steps_per_iteration_default =  1
eval_interval_default = 100#1000#1000#50#100# 
replay_buffer_max_length_default=1000000
batch_size_default=64

epsilon_default = 0.2


initial_state_reuse_default =0.7
#reward_type_default =[0,1,2,3] #4 valid/invalid
max_number_initial_state_reuse_default =100


i=0
#number_empty_vials=2,water_level=4,max_steps = 5000, initial_state_reuse=0.0, reward_type=0, max_number_initial_state_reuse=50
env_values_default =[epsilon_default,initial_state_reuse_default,0,max_number_initial_state_reuse_default]
params_default = [env_values_default,num_iterations_default,learning_rate_default,num_eval_episodes_default,collect_steps_per_iteration_default,eval_interval_default,replay_buffer_max_length_default,batch_size_default]


reward_index = 2


def param_prep(i, reward_type):
            params = copy.deepcopy(params_default)
            params[0][reward_index] = reward_type
            '''if intern_param:
                params[0][index_param] = parameter
            else:
                params[index_param] = parameter
            '''
            print(["state**************************",i])
            prefix = "Water"+str(i)
            file_step =prefix
            tempdir_it = os.path.join(tempdir, file_step)
            os.makedirs(tempdir_it, exist_ok=True)
            config_file = tempdir_it + "/configuration.csv"

            with open(config_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows([['epsilon,initial_state_reuse,reward_type,max_number_initial_state_reuse','num_iterations','learning_rate','num_eval_episodes','collect_steps_per_iteration','eval_interval','replay_buffer_max_length','batch_size'],params])

            params.insert(0,prefix)
            return params

intern_param = False
index_param=1 


for reward_type in reward_type_list:
        params = param_prep(i, reward_type)
        print(params)
        wsp_dqn_learning(params)
        i+=1

