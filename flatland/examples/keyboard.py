#import necessary modules that this python scripts need.
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from pyglet.window import key
import time


#Initialize Flatland Railway environment
railWaySettings = complex_rail_generator(
                   nr_start_goal=10, # Number of start location and train station pairs
                   nr_extra=10, # Extra rails between each pair of start and goal.
                   min_dist=10, # Minimum distance between start and goal locations
                   max_dist=99999,
                   seed=999)
 
env = RailEnv(  width=15, #Width/columns of grids
               height=15, #Height/rows of grids
               rail_generator= railWaySettings,
               number_of_agents=1) # How many trains are enabled.
env.reset() # initialize railway env
 
# Initiate the graphic module
render = RenderTool(env,
                         show_debug=False,
                         screen_height=500,  # Window resolution height
                         screen_width=500)  # Window resolution width
render.render_env(show=True, frames=False, show_observations=False)
window=render.gl.window  #We will use this window object to capture user input

#Capture key press event from window and set pressed key to pressedKey variable
pressedKey = None #store pressed key here
keyMap={key.LEFT:1,#Turn left
       key.UP:2,#Go forward
       key.RIGHT:3,#Turn right
       key.DOWN:4#Stop
       }

@window.event
def on_key_press(symbol, modifiers):
    global pressedKey
    if symbol in keyMap.keys():
        pressedKey = keyMap[symbol]

#Define Controller
def my_controller(number_agents,first_makespan=False):
   _action = {}
   global pressedKey #declare pressedKey is a global variable
 
   if first_makespan:
       #In the first frame of flatland, agents aren disabled,
       #  use any action to enable agents.
       for i in range(0,number_agents):
           _action[i] = 2
       return _action
  
   #Retrieve pressed action and put it into an action dictionary.
   #We only have 1 agent in this practice,
   #  thus we assign the pressed action only to agent 0
   if pressedKey is not None:
       _action[0] = pressedKey
   else:
       _action[0] = 0
   pressedKey = None
   return _action


   #Main loop
cost_dict={} #Record cost for each agent
makespan = -1 #all agents are disabled in the first makespan, so we start from -1..
all_done = False
while not all_done:
    time.sleep(0.25)
    #Call controller function to get the action dictionary for all agents
    _action = my_controller(len(env.agents), makespan == -1)
    #Pass action dictionary to railway env and execute all actions.
    obs, all_rewards, done, info = env.step(_action)
    all_done = done["__all__"]
 
   #count cost for each agent
    for key,value in done.items():
        if value == False and key != "__all__":
            if key not in cost_dict.keys():
                cost_dict[key] = 1
            else:
                cost_dict[key] += 1
    makespan+=1
    render.render_env(show=True, frames=False, show_observations=False)
    time.sleep(0.15)
 
print("Sum of individual cost: ", sum(cost_dict.values()))
print("Total makespan: ", makespan)
