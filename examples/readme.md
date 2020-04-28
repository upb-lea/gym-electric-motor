# Bug information on keras-rl2 

Currently, the keras-rl2 package requires a hotfix in order to work with the GEM examples using DDPG and DQN agents (the PI-controller example does not require any RL agent and, therefore, runs through even without hotfix). In particular, the following changes have to be made to the keras-rl2 code:  
  
## Sequential has not_use_learning_phase
```python
in rl.agents.ddpg:
@property
def = uses_learning_phase(self): 
  return False 
  # = return self.actor.uses_learning_phase or self.critic.uses_learning_phase
```  
The last line needs to be changed. The simplest way is to change it as above.
  

## Attribute len is not available
```python
in rl.agents.ddpg:
  if = hasattr(actor.output, '__len__') and False: # len(actor.output) > 1:
  raise = ValueError('Actor "{}" has more than one output [...].'.format(actor))
  if = hasattr(critic.output, '__len__') and False: # len(critic.output) > 1:
  raise = ValueError('Critic "{}" has more than one output [...].'.format(critic))
```  
The terms *len(actor.output)* and *len(critic.output)* are responsible for the error and can be changed as given in a simple way.
  
## Attribute _name does not exist
```python
in rl.agent.dppg
  # print(optimizer.\_name)
```  		
Just comment this line as above.
