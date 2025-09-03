"""
This is the python script week 1 tutorial exercise.
"""


#########################
# Your Task
#########################

# Run this script, required start and goal locations will be printed to terminal.
# Then replace empty lists with lists of location tuple, so that each train can following the path to reach goal location.
# For example : [(2,3),(2,4),(3,4)]
train_1 = []
train_2 = []

# Turn debug to True to know what's wrong with your path.
debug = False



#########################
# You should not modify any codes below. You can read it know how we ran flatland environment.
########################
print("A locaiton (x,y) indicate a cell on x row and y column")
print("Train 1: Start  (0, 1)  Goal  (6, 7)")
print("Train 2: Start  (1, 5)  Goal  (6, 1)")

def get_path(agents, rail, max_ep_steps):
    return [train_1, train_2]

import glob, os

#import necessary modules that this python scripts need.
try:
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator
except Exception as e:
    print("Cannot load flatland modules! Make sure your flatland-rl is properly installed.")
    print(e)
    exit(1)

script_path = os.path.dirname(os.path.abspath(__file__))
test_cases = glob.glob(os.path.join(script_path,"test_0.pkl"))
evaluator(get_path,test_cases,debug,True,3)


















