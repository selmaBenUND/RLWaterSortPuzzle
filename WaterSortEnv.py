from asyncio import base_tasks
import copy
import numpy as np
import random


from tf_agents.environments import utils
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts





#actions = {0: pour from vial 0 to vial 1, 1: pour from vial 1 to vial 2, ....}
#reward_type = {0: win=+1/lose=-1, 1:solved vials=+1, 3: -1 for each action (for number of movements), 4: +1 for correct move / -1 for invalid actions}

class WaterSortEnv(py_environment.PyEnvironment):

  #Initialization
  def __init__(self,number_vials, number_empty_vials=2,water_level=4,max_steps = 500, initial_state_reuse=0.0, reward_type=0, max_number_initial_state_reuse=50):

    super(WaterSortEnv,self).__init__()

    self.water_level=water_level
    self.number_empty_vials=number_empty_vials
    self.number_vials=number_vials
    self.number_colors=self.number_vials-self.number_empty_vials

    self.max_steps = max_steps
    self.reward_type = reward_type
    self._initial_state_reuse = initial_state_reuse
    self.initial_state_reuse = initial_state_reuse
    self.number_initial_state_reuse = 0
    self.max_number_initial_state_reuse = max_number_initial_state_reuse

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=number_vials*(number_vials-1)-1, name='action')

    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(number_vials*self.water_level,), dtype=np.int32, minimum=0, name='observation') #number_vials,self.water_level

    self._observation_spec = {
                                'observation': array_spec.BoundedArraySpec(
                                  shape=(self.number_vials*self.water_level,), dtype=np.int32, name='observation'), 
                                'valid_actions': array_spec.ArraySpec(
                                                                   name  = "valid_actions",
                                                                   shape = (number_vials*(number_vials-1), ), 
                                                                   dtype = np.bool_
                                            )
                                }
    self._state, self.vials_capacity, self.number_moves, self.number_actions = self.new_game(True)
    self.initial_state = copy.deepcopy(self._state)
    self._episode_ended = False

  #The spec of the actions
  def action_spec(self):
    return self._action_spec

  #The spec of the states
  def observation_spec(self):
    return self._observation_spec

  #For invalid actions masking
  def observation_and_action_splitter(self):
     action_mask = [0]*self.number_vials*(self.number_vials-1)
     #state_work = np.reshape(observation,(nbVials,water_level))
     
     for action in range(0,self.number_vials*(self.number_vials-1)):
        source_vial, destination_vial = self.action_to_vials(action)
        #print([action, source_vial, destination_vial])
        if (self.can_pour_from_vial(source_vial)):
          if(self.can_pour_from_source_to_destination_vial(source_vial,destination_vial)):
                  action_mask[action] = 1 #valid action      
     return action_mask
  
  #Reinitialisation of a state
  def _reset(self):
    self._state, self.vials_capacity, self.number_moves, self.number_actions = self.new_game(False)
    self.initial_state = copy.deepcopy(self._state)
    self._episode_ended = False
    obs ={}
    obs['observation'] = np.array(self._state, dtype=np.int32)
    obs['valid_actions'] = np.array(self.observation_and_action_splitter(), dtype=np.bool_)

    return ts.restart(obs)#ts.restart(np.array(obs, dtype=np.int32))

  def reset2(self,state):
    self._state, self.vials_capacity, self.number_moves, self.number_actions = self.new_game(False)
    self._state = copy.deepcopy(state)
    self.initial_state = copy.deepcopy(self._state)
    self._episode_ended = False
    obs ={}
    obs['observation'] = np.array(self._state, dtype=np.int32)
    obs['valid_actions'] = np.array(self.observation_and_action_splitter(), dtype=np.bool_)

    return ts.restart(obs)#ts.restart(np.array(obs, dtype=np.int32))


  #Return the current state
  def get_state(self):
    obs ={}
    obs['observation'] = np.array(self._state, dtype=np.int32)
    obs['valid_actions'] = np.array(self.observation_and_action_splitter(), dtype=np.bool_)
    return obs
  
  #Set the environment with a specific state (for test only)
  def set_state(self,state,capacity):
      self._state = state
      self.vials_capacity = capacity
  
  #Set the environment with a specific state (for test only)
  def set_environment(self, state,number_vials,number_empty_vials=2,water_level=4):
    self._state = state
    self.vials_capacity = self.calculate_capacity(state,number_vials,water_level)

  #Calculate the capacity of the vials
  def calculate_capacity(self, state,number_vials,water_level):
    capacity = [0]*number_vials
    state_work = np.reshape(state,(number_vials,water_level))
    for i in range(0,number_vials):
      color=0
      for j in range(0,water_level):
        if state_work[i][j] !=0 :
          color += 1
      capacity[i] = color
    return capacity
  
  #Return the vials capacity
  def get_vials_capacity(self):
      return self.vials_capacity

  #Generate a new game either randomly or by using the previous game
  def new_game(self,new):
    self._episode_ended=False
    reuse_state = random.random()
    if ((not new) and reuse_state <self.initial_state_reuse):
      game_board = copy.deepcopy(self.initial_state)
    else:
      game_board = self.generate_game()
    capacity = [self.water_level]*self.number_colors
    capacity = capacity +[0]*self.number_empty_vials
    self.initial_state_reuse = self._initial_state_reuse
    #matrix = np.genfromtxt('p1.csv', delimiter=',', dtype=int)
    #state = np.reshape(game_board,(self.number_vials,self.water_level))
    return [game_board, capacity, 0, 0]

  #Generate a random state
  def generate_game(self):
    game_board = [[0]*self.water_level for i in range(self.number_vials)]
    color_count =[self.water_level]*self.number_colors
    color_list = list(range(1,self.number_colors+1))

    for i in range(0, self.number_colors):
        for j in range(0, self.water_level):
            # any random numbers from 0 to 1000
            color= random.choice(color_list)
            color_count[color-1]= color_count[color-1]-1
            game_board[i][j] = color
            if(color_count[color-1]==0):
                color_list.remove(color)
    game_board=[3,7,12,7,7,5,8,9,11,1,6,3,5,5,4,3,2,8,3,11,6,11,10,2,1,9,10,4,10,12,11,2,8,1,12,6,7,2,6,9,12,1,10,9,4,4,8,5,0,0,0,0,0,0,0,0]
    state = np.reshape(game_board,(self.number_vials*self.water_level,))
    return state

  ################################
  #If there is an empty vial
  def exists_empty_vial(self):
    flag=False
    if (0 in self.vials_capacity):
      flag =True
    return flag

  #if vial is empty
  def is_empty_vial(self,vial):
    flag=False
    if (self.vials_capacity[vial] ==0):
      flag =True
    return flag

  #if pouring from vial is valid
  def can_pour_from_vial(self,vial):
    flag=True
    if (self.is_empty_vial(vial) or self.get_number_top_color(vial) == self.water_level):
      flag =False
    return flag

  #if pouring from vial source to destination is valid
  def can_pour_from_source_to_destination_vial(self,source_vial,destination_vial):
    flag =False
    destination_vials = self.destination_vials_to_pour_color(source_vial)
    if(destination_vial in destination_vials):
      flag =True
    return flag

  #if the game is not stuck in a state
  def can_pour_from_any_vial(self):
    flag=False
    if (self.exists_empty_vial()):
      return True

    for i in range(0,self.number_vials):
      if (self.can_pour_from_vial(i)):
        destination_vials = self.destination_vials_to_pour_color(i)
        if (len(destination_vials) !=0):
          return True

    return flag

  #Return number of colors in the top of a vial
  def get_number_top_color(self,vial):
    top_color_index = self.vials_capacity[vial]-1
    state_work = np.reshape(self._state,(self.number_vials,self.water_level))
    top_color = state_work[vial][top_color_index]
    number_top_color = 0
    #top_color_index -=1
    while(top_color_index>=0 and top_color== state_work[vial][top_color_index] and top_color!=0):
      number_top_color += 1
      top_color_index -=1

    return number_top_color

  #All possible destinations from source vial ---- must be preceded with can_pour_from_vial
  def destination_vials_to_pour_color(self,source_vial):
    destination_vials =[]
    state_work = np.reshape(self._state,(self.number_vials,self.water_level))
    for i in range(0,self.number_vials):
      if (i != source_vial and self.vials_capacity[i]!=self.water_level):
        if(self.vials_capacity[i] ==0):
          destination_vials.append(i)
        else:
          if (state_work[i][self.vials_capacity[i]-1] == state_work[source_vial][self.vials_capacity[source_vial]-1]):
            destination_vials.append(i)

    return destination_vials

  #if the game is not stuck in a state
  def is_stuck_color(self,source_vial,destination_vials):
    flag = True
    for destination_vial in destination_vials:
      if not ((self.get_number_top_color(source_vial)>(self.water_level-self.vials_capacity[destination_vial])) and (self.get_number_top_color(destination_vial)>(self.water_level-self.vials_capacity[source_vial])) ):
        return False
    return flag
  
  # Convert the action (integer) into a source and destination vials
  def action_to_vials(self,action):
    source_vial = action % self.number_vials
    destination_vial = ((action // self.number_vials)+1+source_vial)% self.number_vials
    return [source_vial,destination_vial]

  # Perform a pouring action---- must perform test in code
  def pour_color_action(self,source_vial,destination_vial):
    #if (self.can_pour_from_vial(source_vial)):
    #  if(self.can_pour_from_source_to_destination_vial(source_vial,destination_vial)):
    state_work = np.reshape(self._state,(self.number_vials,self.water_level))

    top_color_source_index = self.vials_capacity[source_vial]-1
    top_color_source = state_work[source_vial][top_color_source_index]
    destination_capacity = self.vials_capacity[destination_vial]


    while(top_color_source_index>=0 and top_color_source == state_work[source_vial][top_color_source_index] and destination_capacity<self.water_level):
      state_work[destination_vial][destination_capacity] = state_work[source_vial][top_color_source_index]
      state_work[source_vial][top_color_source_index] = 0
      top_color_source_index-=1
      destination_capacity+=1
      self.vials_capacity[destination_vial] = self.vials_capacity[destination_vial] +1
      self.vials_capacity[source_vial] = self.vials_capacity[source_vial] -1

    self._state = np.reshape(state_work,(self.number_vials*self.water_level,))
    self.number_moves +=1
  
  # Check if the game is won
  def game_won(self):
    flag=True
    state_work = np.reshape(self._state,(self.number_vials,self.water_level))
    for i in range(0, self.number_vials):
        color_vial = state_work[i][0]
        for j in range(0, self.water_level):
            if(state_work[i][j]!=color_vial):
                flag=False
                break

    return flag

  # Check if the game is over
  def game_over(self):
    flag=True
    destination_vials = [[]]*self.number_vials
    for i in range(0,self.number_vials):
      if (self.is_empty_vial(i)):
        return False
      else:
        if (self.can_pour_from_vial(i)):
            destination_vials[i] = self.destination_vials_to_pour_color(i)
            if (len(destination_vials) !=0):
              flag_stuck = self.is_stuck_color(i,destination_vials[i])
              if (not flag_stuck):
                return False
              else:
                flag =True
            else:
              flag =True
    return flag


  def _step(self, action):
    self.number_actions +=1
    reward1 = 0
    reward2 = 0
    
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      #print("resetttttttttttttttttt "+str(self.number_actions))
      return self.reset()


    #the continuation of the _step(self, action) function
    # Make sure episodes don't go on forever.
    source_vial, destination_vial = self.action_to_vials(action)
    #print("begin  ",self._state)
    #print("action  ",[action, source_vial, destination_vial])
    if (self.can_pour_from_vial(source_vial)):
      if(self.can_pour_from_source_to_destination_vial(source_vial,destination_vial)):
        self.pour_color_action(source_vial,destination_vial)
        reward1 =self.get_number_top_color(destination_vial) 
      else:
        reward2 = -1
    else:
      reward2 = -1

    if(self.reward_type ==0):
        reward1 =0
        reward2 =0
    else:
          if (self.reward_type ==1):
            reward2 =0
            if reward1 < 4:
              reward1=0
          else:
            if (self.reward_type ==2):
              reward1=-1
              reward2=0
            else:
                if (self.reward_type ==3):
                  if reward2 ==0:
                    reward1=1
                  else:
                    reward1=0

    reward = reward1+reward2
    #reward = penalty-self.number_actions
    #reward = -self.number_actions
    if self.game_won():
      self._episode_ended = True
      if(self.reward_type ==0):
        reward = 1
      print("****************** game won ",[self.number_actions, self.number_moves])
      #print(self._state)
      self.number_initial_state_reuse += 1
      if (self.number_initial_state_reuse >= self.max_number_initial_state_reuse):
        #self.initial_state_reuse = 0.9
      #else:
        self.initial_state_reuse = 0
        self.number_initial_state_reuse = 0

      obs ={}
      obs['observation'] = np.array(self._state, dtype=np.int32)
      obs['valid_actions'] = np.array(self.observation_and_action_splitter(), dtype=np.bool_)
      return ts.termination(obs, reward)
      #return ts.termination(np.array(obs, dtype=np.int32), reward)
    else:
      if self.game_over() or self.number_actions>self.max_steps:
        self._episode_ended = True
        if(self.reward_type ==0):
          reward = -1
        self.number_initial_state_reuse += 1
        if (self.number_initial_state_reuse >= self.max_number_initial_state_reuse):
          #self.initial_state_reuse = 0.9
        #else:
          self.initial_state_reuse = 0
          self.number_initial_state_reuse = 0
        
        print("****************** game over",[self.number_actions, self.number_moves])
        #print(self._state)
        #reward += -1
        obs ={}
        obs['observation'] = np.array(self._state, dtype=np.int32)
        obs['valid_actions'] = np.array(self.observation_and_action_splitter(), dtype=np.bool_)
        return ts.termination(obs, reward)
        #return ts.termination(np.array(self._state, dtype=np.int32), reward)
      else:
        obs ={}
        obs['observation'] = np.array(self._state, dtype=np.int32)
        obs['valid_actions'] = np.array(self.observation_and_action_splitter(), dtype=np.bool_)
        return ts.transition(obs,reward=reward, discount=1.0)
        #return ts.transition(np.array(obs, dtype=np.int32), reward=reward, discount=1.0)






  
  