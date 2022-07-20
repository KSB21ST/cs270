from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden

#################### to the TA's#############################
'''
I was coding in a file named student_t.py, and only after I decided to submit did I realize that the logs folder was only saving student.py.
I changed the name student_t.py into student.py, but somehow the code doesn't work anymore in my cmd.
However, the due hour is too near to modify such things.
'''

import numpy as np
import time

show_animation(True)
set_speed(3)          # This line is only meaningful if animations are enabled.
 
#####################################
#### Implement steps 1 to 3 here ####
#####################################
class icewalker():
    def __init__(self, pos, ori, sensor, done):
        self.visited_node = []
        self.ice_list = set([])
        self.max_x = 6
        self.max_y = 6
        self.cr_x = pos[0]
        self.cr_y = pos[1]
        self.ori = self.ori_chng(ori)
        self.sensor = sensor
        self.done = done

    def ori_chng(self, ori):
        if(ori == 'S'):
            return 1
        if(ori == 'N'):
            return 3
        if(ori == 'E'):
            return 0
        if(ori == 'W'):
            return 2
            
    def change_dir(self, cur_dir, next_dir): 
        (x, y), ori, sensor, done = (self.cr_x, self.cr_y), self.ori, self.sensor, self.done
        print("change_dir: ", (self.cr_x, self.cr_y), self.ori, self.sensor, self.done)
        if(cur_dir == 0):
            if(next_dir == 0):
                turn_left()
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 1):
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 2):
                turn_right()
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 3):
                (x, y), ori, sensor, done = turn_left()
        if(cur_dir == 1):
            if(next_dir == 1):
                turn_left()
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 0):
                (x, y), ori, sensor, done = turn_left()
            if(next_dir == 2):
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 3):
                turn_left()
                (x, y), ori, sensor, done = turn_left()
        if(cur_dir == 2):
            if(next_dir == 2):
                turn_left()
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 0):
                turn_left()
                (x, y), ori, sensor, done = turn_left()
            if(next_dir == 1):
                (x, y), ori, sensor, done = turn_left()
            if(next_dir == 3):
                (x, y), ori, sensor, done = turn_right()
        if(cur_dir == 3):
            if(next_dir == 3):
                turn_left()
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 0):
                (x, y), ori, sensor, done = turn_right()
            if(next_dir == 1):
                turn_left()
                (x, y), ori, sensor, done = turn_left()
            if(next_dir == 2):
                (x, y), ori, sensor, done = turn_left()
        return next_dir, (x, y), self.ori_chng(ori), sensor, done
    
    def search_map(self, parent):
        if(self.cr_x == 6 and self.cr_y == 6):
            print("done!")
            return
        if(self.done == True):
            print("done!")
            return
        self.visited_node.append((self.cr_x, self.cr_y))
	    # print_passed(passed_list)
        to_child = -1
        adj_node = [(self.cr_x+1, self.cr_y), (self.cr_x, self.cr_y+1), (self.cr_x-1, self.cr_y), (self.cr_x, self.cr_y-1)]
        for j in range(4):
            next_node = adj_node[j]
            next_x, next_y = next_node[0], next_node[1]
            if(next_x <0) or (next_y < 0):
                continue
            if(next_x > 6 or next_y > 6):
                continue
            if((next_x, next_y) in self.visited_node):
                continue
            print("before: ", (self.cr_x, self.cr_y), self.ori, self.sensor, self.done)
            self.ori, (self.cr_x, self.cr_y), self.ori, self.sensor, self.done = self.change_dir(self.ori, j)
            print("search map: ", (self.cr_x, self.cr_y), self.ori, self.sensor, self.done)
            if(self.sensor[1] == 1):
                continue
            if(j == 0): #move right
                if(self.cr_x >= 6):
                    continue
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
                self.ori = self.ori_chng(t_ori)
                to_child = 2
            if(j == 1): #move down
                if(self.cr_y >= 6):
                    continue
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
                self.ori = self.ori_chng(t_ori)
                to_child = 3
            if(j == 2): #move left
                if(self.cr_x <= 0):
                    continue
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
                self.ori = self.ori_chng(t_ori)
                to_child = 0
            if(j == 3): #move up
                if(self.cr_y <= 0):
                    continue
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
                self.ori = self.ori_chng(t_ori)
                to_child = 1
            self.search_map(to_child)

        if(parent == 0):
            self.change_dir(self.ori, 0)
            (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
            self.ori = self.ori_chng(t_ori)
        if(parent == 1):
            self.change_dir(self.ori, 1)
            (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
            self.ori = self.ori_chng(t_ori)
        if(parent == 2):
            self.change_dir(self.ori, 2)
            (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
            self.ori = self.ori_chng(t_ori)
        if(parent == 3):
            self.change_dir(self.ori, 3)
            (self.cr_x, self.cr_y), t_ori, self.sensor, self.done = move_forward()
            self.ori = self.ori_chng(t_ori)
        return



grid_size   = (8, 8)             # Size of the map
goal        = (6, 6)             # Coordinates of the goal
orientation = ['N','E','S','W']  # List of orientations
# N : up
# E: right
# S: Down
# W: left

# Hyperparameters: Feel free to change all of these!
actions = [move_forward, turn_left, turn_right]
num_epochs = 3000
alpha = 0
gamma = 0
epsilon = 0
# q_table = np.zeros([6*6, 8, 4, len(orientation)]) #shape each for [state, ori, sensor, action]
q_table = np.zeros([6*6, 8, 4])
# q_table = np.zeros([6*6, 2, 4]) #middle: check only the front sensor
# q_table = np.zeros([6*6, 4])
lr = .8   # learning rate
gamma = .99   # discount factor
#create lists to contain total rewards and steps per episode
rList = [] # reword list
# sList = [] # state list
#end

def sort_sensor(sensor):
    if(sensor[0] == 0):
        if(sensor[1] == 0):
            if(sensor[2] == 0):#[0, 0, 0]
                return 0
            else: #[0, 0, 1]
                return 1
        else: #sensor[1] == 1
            if(sensor[2] == 0): #[0, 1, 0]
                return 2
            else: #[0, 1, 1]
                return 3
    else: #sensor[0] == 1
        if(sensor[1] == 0):
            if(sensor[2] == 0):#[1, 0, 0]
                return 4
            else: #[1, 0, 1]
                return 5
        else: #sensor[1] == 1
            if(sensor[2] == 0): #[1, 1, 0]
                return 6
            else: #[1, 1, 1]
                return 7


# def change_dir(ori, pre_ori): #ori 가 원래 position
#     orientation = ['N','E','S','W'] 
#     if(ori == orientation.index('N')):
#         if(pre_ori ==orientation.index('W')):
#             print("N, W")
#             turn_left()
#         elif(pre_ori == orientation.index('S')):
#             turn_left()
#             turn_left()
#         elif(pre_ori == orientation.index('E')):
#             print("N, E")
#             turn_right()
#         # elif(pre_ori == 'N'):
#     if(ori == 'E'):
#         if(pre_ori ==orientation.index('W')):
#             turn_left()
#             turn_left()
#         elif(pre_ori == orientation.index('S')):
#             turn_right()
#         # elif(pre_ori == 'E'):
#         elif(pre_ori == orientation.index('N')):
#             print("e, n")
#             turn_left()
#     if(ori == orientation.index('S')):
#         if(pre_ori == orientation.index('W')):
#             turn_right()
#         # elif(pre_ori == 'S'):
#         elif(pre_ori == orientation.index('E')):
#             turn_left()
#         elif(pre_ori == orientation.index('N')):
#             turn_left()
#             turn_left()
#     if(ori == orientation.index('W')):
#         # if(pre_ori == 'W'):
#         if(pre_ori == orientation.index('S')):
#             turn_left()
#         elif(pre_ori == orientation.index('E')):
#             turn_right()
#             turn_right()
#         elif(pre_ori == orientation.index('N')):
#             turn_right()
#     return

def ori_chng(ori):
    if(ori == 'S'):
        return 1
    if(ori == 'N'):
        return 3
    if(ori == 'E'):
        return 0
    if(ori == 'W'):
        return 2

def change_dir(cur_dir, next_dir): 
    if(cur_dir == 0):
        # if(next_dir == 0):
        #     turn_left()
        #     (x, y), ori, sensor, done = turn_right()
        if(next_dir == 1):
            turn_right()
        if(next_dir == 2):
            turn_right()
            turn_right()
        if(next_dir == 3):
            turn_left()
    if(cur_dir == 1):
        # if(next_dir == 1):
        #     turn_left()
        #     (x, y), ori, sensor, done = turn_right()
        if(next_dir == 0):
            turn_left()
        if(next_dir == 2):
            turn_right()
        if(next_dir == 3):
            turn_left()
            turn_left()
    if(cur_dir == 2):
        # if(next_dir == 2):
        #     turn_left()
        #     (x, y), ori, sensor, done = turn_right()
        if(next_dir == 0):
            turn_left()
            turn_left()
        if(next_dir == 1):
            turn_left()
        if(next_dir == 3):
            turn_right()
    if(cur_dir == 3):
        # if(next_dir == 3):
        #     turn_left()
        #     (x, y), ori, sensor, done = turn_right()
        if(next_dir == 0):
            turn_right()
        if(next_dir == 1):
            turn_left()
            turn_left()
        if(next_dir == 2):
            turn_left()
    return


# Define your reward function
def train():
    d_cnt = 0
    for i in range(num_epochs):
        set_speed(200)  
        # (x, y), ori, sensor, done = reset_map()
        (x, y), ori, sensor, d = reset_map()
        idx_1 = sort_sensor(sensor)
        # idx_1 = sensor[1]
        ori_1 = ori
        s = 6*(y-1) + (x-1)
        rAll = 0        # total reward
        j = 0           # step
        e = 1./((i/50)+10)

        orientation = ['N','E','S','W']  # List of orientations
# N : up
# E: right
# S: Down
# W: left

        #The Q-Table learning algorithm
        while j < 1000:
            j+=1
            if np.random.rand(1) < e:
                # a = np.argmax(q_table[s, :] + np.random.randn(1, 4)*(1./(i+1))) 
                a = np.argmax(q_table[s][idx_1] + np.random.randn(1, 4)*(1./(i+1))) 
            else:
                # a = np.argmax(q_table[s, :])
                a = np.argmax(q_table[s][idx_1])
            # a = np.argmax(q_table[s][idx_1] + np.random.randn(1, 4)*(1./(i+1))) 
# 0: right
# 1: down
# 2: left
# 3: up
            # change_dir(ori_chng(ori), a)   
            # (x, y), ori, sensor, d = move_forward()
            if(a == 0):
                (x, y), ori, sensor, d = move_forward()
            elif(a == 1):
                (x, y), ori, sensor, d = turn_left()
                (x, y), ori, sensor, d = move_forward()
            elif(a==2):
                (x, y), ori, sensor, d = turn_right()
                (x, y), ori, sensor, d = move_forward()
            else:
                (x, y), ori, sensor, d = turn_right()
                (x, y), ori, sensor, d = turn_right()
                (x, y), ori, sensor, d = move_forward()

            s1 = 6*(y-1) + (x-1)
            idx_2 = sort_sensor(sensor)
            # idx_2 = sensor[1]
            ori_2 = ori
            if((x, y) == (6, 6)):
                r = 36
            else:
                if(d):
                    if(x > 6 or y > 6 or x < 1 or y < 1):
                        s1 = s
                        # if(x<0 or y<0):
                        #     r = -(x+1)*(y+1)
                        # else:
                        #     r = -(x)*(y)
                    r = -1
                    # else:
                    # if(sensor[1] == 1 and a == 0):
                    #     r = -(x*y) - 3
                    # elif(sensor[0] == 1 and a == 1):
                    #     r = -(x*y) - 3
                    # elif(sensor[2] == 1 and a == 2):
                    #     r = -(x*y) - 3
                    # else:
                    #     r = -(x*y)
                # elif(ori == 'E' or ori == 'S'):
                #     r = -5
                # elif(ori == 'N' or ori == 'W'):
                #     r = -0.5
                # elif(ori == 'E' or ori == 'S'):
                #     r = x*y
                else:
                    # r = 0
                    r = x*y
            print("(x, y): ", (x, y), r)
            
            # Update Q-Table with new knowledge(=reward)
            # Q[s,a] = Q[s,a] + lr*(r + gamma*np.max(Q[s1,:]) - Q[s,a]) 
            # q_table[s,a] = q_table[s,a] + lr*(r + gamma*np.max(q_table[s1,:]) - q_table[s,a]) 
            q_table[s][idx_1][a] = q_table[s][idx_1][a] + lr*(r + gamma*np.max(q_table[s1]) - q_table[s][idx_1][a]) 
            # q_table[s, idx_1, [orientation.index(ori_1)], a] = q_table[s, idx_1, orientation.index(ori_1), a] + lr*(r + gamma*np.max(q_table[s1,:]) - q_table[s, idx_1, orientation.index(ori_1), a]) 
            
            rAll += r # add reward 
            s = s1    # move to next state
            idx_1 = idx_2
            ori_1 = ori_2

            # Check some conditions in console
            
            # check the end of process
            if (x, y) == (6, 6):
                d_cnt += 1
                break
        print("_____epoch : ", i)
        rList.append(rAll)
    return d_cnt

# action 0 : Left
# action 1 : Down
# action 2 : Right
# action 3 : UP
def test_table():
    data = np.load('q_table_v3.npy')
    for i in data:
        print(i)

    set_speed(3)
    test()
    (x, y), ori, sensor, done = reset_map()

    set_speed(1) 
    while not done:
        idx = sort_sensor(sensor)
        s = 6*(y-1) + (x-1)
        # a = np.argmax(data[s])
        a = np.argmax(data[s][idx])
        print("----------------------")
        print(data[s][idx])
        print(a)
        # change_dir(ori_chng(ori), a)   
        # (x, y), ori, sensor, done = move_forward()
        # ''' for q_v1, q_v2
        if(a == 0):
            (x, y), ori, sensor, d = move_forward()
        elif(a == 1):
            (x, y), ori, sensor, d = turn_left()
            (x, y), ori, sensor, d = move_forward()
        elif(a==2):
            (x, y), ori, sensor, d = turn_right()
            (x, y), ori, sensor, d = move_forward()
        else:
            (x, y), ori, sensor, d = turn_right()
            (x, y), ori, sensor, d = turn_right()
            (x, y), ori, sensor, d = move_forward()
        # '''


#####################################
#####training part#########
# tr_cnt = train()
# print("true number: ", tr_cnt)
# np.save("q_table_v5", q_table)

(x, y), ori, sensor, done = reset_map()
test()
myIcewalker = icewalker((x, y), ori, sensor, done)
myIcewalker.search_map(-1)

test_table()

# set_speed(3)
# test()
# (x, y), ori, sensor, done = reset_map()

###############################
#### Implement step 4 here ####
###############################
# test_table()

###############################

#### If you want to try moving around the map with your keyboard, uncomment the below lines 
# import pygame
# set_speed(5)
# show_animation(True)
# while True:
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			exit("Closing...")
# 		if event.type == pygame.KEYDOWN:
# 			if event.key == pygame.K_LEFT: print(turn_left())
# 			if event.key == pygame.K_RIGHT: print(turn_right())
# 			if event.key == pygame.K_UP: print(move_forward())
# 			if event.key == pygame.K_t: test()
# 			if event.key == pygame.K_r: print(reset_map())
# 			if event.key == pygame.K_q: exit("Closing...")
