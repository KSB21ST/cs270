from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math

show_animation(True)
set_speed(5000)          # This line is only meaningful if animations are enabled.
 
# print(reset_map())
# ((1, 1), 'S', [1, 0, 1], False, False, (10, 1), 'S', [1, 0, 0], False, False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class icewalker():
    def __init__(self, pos, ori, sensor, done, goal):
        self.visited_node = []
        self.ice_list = set([])
        self.max_x = 10
        self.max_y = 10
        self.cr_x = pos[0]
        self.cr_y = pos[1]
        self.ori = self.ori_chng(ori)
        self.sensor = sensor
        self.done = done
        self.goal_x = goal[0]
        self.goal_y = goal[1]

    def ori_chng(self, ori):
        if(ori == 'S'):
            return 1
        if(ori == 'N'):
            return 3
        if(ori == 'E'):
            return 0
        if(ori == 'W'):
            return 2
            
    def change_dir(self, cur_dir, next_dir, robot): 
        (x, y), ori, sensor, done = (self.cr_x, self.cr_y), self.ori, self.sensor, self.done
        if(cur_dir == 0):
            if(next_dir == 0):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 1):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 2):
                turn_right(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 3):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
        if(cur_dir == 1):
            if(next_dir == 1):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 0):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
            if(next_dir == 2):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 3):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
        if(cur_dir == 2):
            if(next_dir == 2):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 0):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
            if(next_dir == 1):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
            if(next_dir == 3):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
        if(cur_dir == 3):
            if(next_dir == 3):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 0):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_right(robot)
            if(next_dir == 1):
                turn_left(robot)
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
            if(next_dir == 2):
                (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2 = turn_left(robot)
        if robot == 1:
            return next_dir, (x1, y1), self.ori_chng(ori1), sensor1, status1
        else:
            return  next_dir, (x2, y2), self.ori_chng(ori2), sensor2, status2
    
    def search_map(self, parent, robot):
        # if(self.cr_x == 10 and self.cr_y == 10):
        if(self.cr_x == self.goal_x and self.cr_y == self.goal_y):
            print("robot ", robot, " done!")
            if (robot ==1):
                print("wait for robot 2...")
            return
        if(self.done == True) and (self.cr_x, self.cr_y) != (self.goal_x, self.goal_y):
            print("killed!")
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
            if(next_x > 10 or next_y > 10):
                continue
            if((next_x, next_y) in self.visited_node):
                continue
            self.ori, (self.cr_x, self.cr_y), self.ori, self.sensor, self.done = self.change_dir(self.ori, j, robot)
            if(self.sensor[1] >= 1):
                continue
            if(j == 0): #move right
                if(self.cr_x >= 10):
                    continue
                if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
                else:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
                self.ori = self.ori_chng(t_ori)
                to_child = 2
            if(j == 1): #move down
                if(self.cr_y >= 10):
                    continue
                if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
                else:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
                self.ori = self.ori_chng(t_ori)
                to_child = 3
            if(j == 2): #move left
                if(self.cr_x <= 0):
                    continue
                if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
                else:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
                self.ori = self.ori_chng(t_ori)
                to_child = 0
            if(j == 3): #move up
                if(self.cr_y <= 0):
                    continue
                if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
                else:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
                self.ori = self.ori_chng(t_ori)
                to_child = 1
            self.search_map(to_child, robot)

        if(parent == 0):
            self.change_dir(self.ori, 0, robot)
            if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
            else:
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
            self.ori = self.ori_chng(t_ori)
        if(parent == 1):
            self.change_dir(self.ori, 1, robot)
            if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
            else:
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
            self.ori = self.ori_chng(t_ori)
        if(parent == 2):
            self.change_dir(self.ori, 2, robot)
            if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
            else:
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
            self.ori = self.ori_chng(t_ori)
        if(parent == 3):
            self.change_dir(self.ori, 3, robot)
            if robot == 1:
                    (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[:5]
            else:
                (self.cr_x, self.cr_y), t_ori, self.sensor, self.done, _done = move_forward(robot)[5:]
            self.ori = self.ori_chng(t_ori)
        return

grid_size = (12, 12)
goal1 = (10, 10)
goal2 = (1, 10)
orientation = ['N','E','S','W']
actions = [move_forward, turn_left, turn_right]
num_epochs = 30000
epsilon = 0.8
BATCH_SIZE = 64

def reward(precoor, coor, status, visited_val, same, steps, goal, sensor, move, o_i, other_coor, preother_coor, pre_act):
    """
    Task 6 (optional) - design your own reward function
    """
    reward = 50
    other = False
    x, y = coor
    prex, prey = precoor
    if coor == goal:
        reward += 200
    if coor ==  goal and (abs(precoor[0]-coor[0]) == 1 or abs(precoor[1]-coor[1]) == 1) and move == 0:
        reward += 200
    if status and coor != goal:
        reward = -10000
    if status and coor != goal and other_coor == coor:
        reward -= 1000
    if x<=0 or y<=0 or x>=10 or y>=10:
        reward -= 300
    if same>=2:
        reward -= 600
    if same >= 5 and move != 0:
        reward -= 1000
    elif visited_val>=3 and pre_act == move and visited_val < 5:
        reward -= 400
    if(sensor[1] >= 1 and move == 0):
        reward -= 700
    # if goal == (10, 10) and (sensor[0] == 2 or sensor[1] == 2 or sensor[2] == 2) and move ==  0:
    # 	reward -= 2000
    # if goal == (1, 10) and ((sensor[1] == 2 and move == 0) or (sensor[0] == 2 and move == 1) or (sensor[2] == 2 and move==2)):
    # 	reward -= 1000
    if (abs(precoor[0]-preother_coor[0]) <= 1 or abs(precoor[1]-preother_coor[1]) <= 1) and move == 0:
        other = True
        reward -= 2000
    if ((abs(precoor[0]-preother_coor[0]) <= 2 and precoor[1] == preother_coor[1] and (o_i == 0 or o_i == 2)) or (abs(precoor[1]-preother_coor[1]) <= 2 and precoor[0] == preother_coor[0]  and (o_i == 1 or o_i == 3))) and move == 0:
        other = True
        reward -= 2000
    # if(sensor[1] == 0 and move != 0):
    # 	reward -= 300
    # if (not status) and (sensor[1] == 0 and move == 0) and coor[0] < 11 and coor[1] < 11:
    # 	reward += 200
    # if(sensor[0] >= 1 and (sensor[1] == 0 or sensor[2] == 0) and move == 1):
    # 	reward -= 100
    # if(sensor[2] >= 1 and (sensor[1] == 0 or sensor[0] == 0) and move == 2):
    # 	reward -= 100
    if goal == (10, 10) and (not status): #빨간색
        if o_i == 1:
            if sensor[1] == 0:
                if visited_val < 3:
                    if move == 0:
                        reward += 300
                    elif (not other):
                        reward -= 300
                else:
                    # if move != 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward += 500
                    # elif move == 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward -= 500
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
            elif sensor[2] == 0:
                if visited_val < 3:
                    if move == 2:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[0] == 0 or sensor[1] == 0):
                        reward -= 300
        if o_i == 2:
            if sensor[1] == 0:
                if visited_val < 3:
                    if move == 0:
                        reward += 300
                    elif (not other):
                        reward -= 300
                else:
                    # if move != 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward += 500
                    # elif move == 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward -= 500
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
            elif sensor[0] == 0:
                if visited_val < 3:
                    if move == 1:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
        if o_i == 3:
            if sensor[0] == 0:
                if visited_val < 3:
                    if move == 1:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
        if o_i == 0:
            if sensor[2] == 0:
                if visited_val < 3:
                    if move == 2:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move ==pre_act and (sensor[0] == 0 or sensor[1] == 0):
                        reward -= 300
    elif goal == (1, 10) and (not status): #파란색
        if o_i == 3:
            if sensor[1] == 0:
                if visited_val < 3:
                    if move == 0:
                        reward += 300
                    elif (not other):
                        reward -= 300
                else:
                    # if move != 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward += 500
                    # elif move == 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward -= 500
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
            elif sensor[0] == 0:
                if visited_val < 3:
                    if move == 1:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
        if o_i == 2:
            if sensor[1] == 0:
                if visited_val < 3:
                    if move == 0:
                        reward += 300
                    elif (not other):
                        reward -= 300
                else:
                    # if move != 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward += 500
                    # elif move == 0 and (sensor[0] == 0 or sensor[2] == 0):
                    # 	reward -= 500
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
            elif sensor[2] == 0:
                if visited_val < 3:
                    if move == 2:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[0] == 0 or sensor[1] == 0):
                        reward -= 300
        if o_i == 1:
            if sensor[2] == 0:
                if visited_val < 3:
                    if move == 2:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[0] == 0 or sensor[1] == 0):
                        reward -= 300
        if o_i == 0:
            if sensor[0] == 0:
                if visited_val < 3:
                    if move == 1:
                        reward += 300
                    else:
                        reward -= 300
                else:
                    if move == pre_act and (sensor[2] == 0 or sensor[1] == 0):
                        reward -= 300
    return reward


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):

        """
        Task 3 - 
        push input: "transition" into replay meory
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position+1)%self.capacity

        return

    def sample(self, batch_size):
        """
        Task 3 - 
        give a batch size, pull out batch_sized samples from the memory
        """
        random_sample = random.sample(self.memory, batch_size)
        return random_sample

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        """
        Task 1 -
        generate your own deep neural network
        """

        self.robot = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )



    def forward(self, x):
        """
        Task 1 - 
        generate your own deep neural network
        """
        x = self.robot(x)
        return x

def optimize_model():
    if len(memory1) < BATCH_SIZE or len(memory2) < BATCH_SIZE:
        return
    transition1 = memory1.sample(BATCH_SIZE)
    transition2 = memory2.sample(BATCH_SIZE)
    """
    Task 4: optimize model
    """
    state1 = []
    action1 = []
    reward1 = []
    new_state1 = []

    state2 = []
    action2 = []
    reward2 = []
    new_state2 = []

    for t1 in transition1:	
        state1.append(t1[0])
        action1.append(t1[1])
        reward1.append(t1[2])
        new_state1.append(t1[3])

    for t2 in transition2:
        state2.append(t2[0])
        action2.append(t2[1])
        reward2.append(t2[2])
        new_state2.append(t2[3])

    # min_max_scaler = MinMaxScaler()

    state1 = torch.stack(state1).to(device)
    action1 = torch.stack(action1)
    reward1 = torch.stack(reward1)
    new_state1 = torch.stack(new_state1).to(device)

    state2 = torch.stack(state2).to(device)
    action2 = torch.stack(action2)
    reward2 = torch.stack(reward2)
    new_state2 = torch.stack(new_state2).to(device)

    state = torch.cat([state1, state2], dim=1)
    new_state = torch.cat([new_state1, new_state2], dim=1)

    temp_q = policy_net(state)
    state_action_values1 = temp_q[:, :3].gather(1, action1)
    state_action_values2 = temp_q[:, 3:].gather(1, action2)
    state_action_values = torch.cat([state_action_values1, state_action_values2], dim=1).to(device)

    state_values = target_net(new_state)
    state_values1 = state_values[:, :3]
    state_values2 = state_values[:, 3:]

    y_values1= []
    y_values2= []
    for i in range(len(transition1)):
        if transition1[i][4]==True:
            y_value1 = reward1[i]
        else:
            y_value1 = reward1[i] + 0.999 * torch.max(state_values1[i])

        if transition2[i][4]==True:
            y_value2 = reward2[i]
        else:
            y_value2 = reward2[i] + 0.999 * torch.max(state_values2[i])
        y_values1.append(y_value1.type(torch.FloatTensor).unsqueeze(0))
        y_values2.append(y_value2.type(torch.FloatTensor).unsqueeze(0))

    expected_state_action_values1 = torch.stack(y_values1).detach()
    expected_state_action_values2 = torch.stack(y_values2).detach()
    expected_state_action_values = torch.cat([expected_state_action_values1, expected_state_action_values2], dim=1).to(device)

    expected_state_action_values = expected_state_action_values


    # loss = F.mse_loss(state_action_values, expected_state_action_values)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


policy_net = DQN().to(device)
# policy_net.load_state_dict(torch.load('policy_net.pth'))
policy_net.train()
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory1 = ReplayMemory(1000)
memory2 = ReplayMemory(1000)



def select_action(state):
    """
    Task 2: select action
    """
    global epsilon
    if random.uniform(0, 1) < epsilon:
        action1 = random.choices([0, 1, 2])
        action1 = torch.from_numpy(np.array(action1))
        action2 = random.choices([0, 1, 2])
        action2 = torch.from_numpy(np.array(action2))
        if epsilon>0.003:
            epsilon-=0.0001
    else:
        with torch.no_grad():
            q_values = policy_net(state)
            action1 = torch.argmax(q_values[:3])
            action2 = torch.argmax(q_values[3:])
    return action1, action2
	
TARGET_UPDATE = 10
same1 = 0
same2 = 0
step1 = 0
step2 = 0

def _train():
    for i in range(num_epochs):
        if (i % 10 == 0): 
            print(i)
        if(i > 2950):
            set_speed(10)
        elif(i > 2000):
            set_speed(500)
        steps = 0
        step1 = 0
        step2 = 0
        same1 = 0
        same2 = 0
        # surv1 = 1
        # surv2 = 1
        visited1 = np.zeros((13, 13))
        visited2 = np.zeros((13, 13))
        pre_act1 = np.full_like(np.zeros((13, 13)), -1)
        pre_act2 = np.full_like(np.zeros((13, 13)), -1)
        (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2  = reset_map()
        while (status1 == False or done1 == True) and (status2 == False or done2 == True):
            o_i1 = orientation.index(ori1)
            o_i2 = orientation.index(ori2)

            cur_state = np.array((x1, y1, sensor1[0], sensor1[1], sensor1[2], o_i1, same1, visited1[x1][y1], pre_act1[x1][y1], x2, y2, sensor2[0], sensor2[1], sensor2[2], o_i2, same2, visited2[x2][y2], pre_act2[x2][y2]), dtype=np.float32) 
            cur_state = torch.from_numpy(cur_state).to(device)
            idx_action = select_action(cur_state)
            idx_action1, idx_action2 = idx_action[0].item(), idx_action[1].item()

            action1 = actions[idx_action1]
            action2 = actions[idx_action2]
            (new_x1, new_y1), new_ori1, new_sensor1, status1, done1, (new_x2, new_y2), new_ori2, new_sensor2, status2, done2 = action1(1)
            (new_x1, new_y1), new_ori1, new_sensor1, status1, done1, (new_x2, new_y2), new_ori2, new_sensor2, status2, done2 = action2(2)
            steps+=1

            if(status1 == False) and (new_x1, new_y1) != (x1, y1):
                step1+=1
            if(status2 == False) and (new_x2, new_y2) != (x2, y2):
                step2+=1
            # if(status1 == True) and (done1 == False):
            # 	surv1 = 0
            # if(status1 == True) and (done1 == False):
            # 	surv2 = 0
            
            # if(new_x2 > 11 or new_y2 >11 or new_x1 > 11 or new_y1 > 11):
            # 	break
                
            # if(new_x2 < 0 or new_y2 <0 or new_x1 <0 or new_y1 <0):
            # 	break

            # if (visited1[new_x1][new_y1] < 5):
            if (new_x1, new_y1) != (x1, y1):
                visited1[new_x1][new_y1] +=1
            if (new_x1, new_y1) == (x1, y1):
                same1+=1
            else:
                same1=0
            if visited1[new_x1][new_y1] > 1:
                pre_act1[x1][y1] = idx_action1

            if (new_x2, new_y2) != (x2, y2):
                visited2[new_x2][new_y2] +=1
            if (new_x2, new_y2) == (x2, y2):
                same2+=1
            else:
                same2=0
            if visited2[new_x2][new_y2] > 1:
                pre_act2[x2][y2] = idx_action2
                
            new_o_i1 = orientation.index(new_ori1)
            visited_val1 = visited1[new_x1][new_y1]
            action_val1 = torch.tensor([idx_action1], device=device)
            reward_val1 = reward((x1, y1), (new_x1,new_y1), status1, visited_val1, same1, step1, goal1, sensor1, idx_action1, o_i1, (new_x2, new_y2), (x2, y2), pre_act1[x1][y1])
            reward_val1 = torch.tensor(reward_val1, device=device)
            new_state1 = np.array((new_x1, new_y1, new_sensor1[0], new_sensor1[1], new_sensor1[2], new_o_i1, same1, visited_val1, pre_act1[new_x1][new_y1]), dtype=np.float32)
            new_state1 = torch.from_numpy(new_state1)
            transition1 = (cur_state[:9], action_val1, reward_val1, new_state1, status1)# generate your own transition form

            new_o_i2 = orientation.index(new_ori2)
            visited_val2 = visited2[new_x2][new_y2]
            action_val2 = torch.tensor([idx_action2], device=device)
            reward_val2 = reward((x2, y2), (new_x2,new_y2), status2, visited_val2, same2, step2, goal2, sensor2, idx_action2, o_i2, (new_x1, new_y1), (x1, y1), pre_act2[x2][y2])
            reward_val2 = torch.tensor(reward_val2, device=device)
            new_state2 = np.array((new_x2, new_y2, new_sensor2[0], new_sensor2[1], new_sensor2[2], new_o_i2, same2, visited_val2, pre_act2[new_x2][new_y2]), dtype=np.float32)
            new_state2 = torch.from_numpy(new_state2)
            transition2 = (cur_state[9:], action_val2, reward_val2, new_state2, status2)# generate your own transition form

            memory1.push(transition1)
            memory2.push(transition2)
            (x1, y1), ori1, sensor1 = (new_x1, new_y1), new_ori1, new_sensor1
            (x2, y2), ori2, sensor2 = (new_x2, new_y2), new_ori2, new_sensor2
            optimize_model()
            # if steps>5000:
            # 	break
            if (done1 == True) and (done2 == True):
                print("success!")
                break
            if (x1, y1) == (x2, y2):
                print(".")
                break
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), 'policy_net.pth')


	
"""
Task 5 - save your policy net
"""
# torch.save(policy_net.state_dict(), 'policy_net.pth')
def _test_network():
    test()
    set_speed(10)
    (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2  = reset_map()
    myIcewalker = icewalker((x1, y1), ori1, sensor1, done1, goal1)
    myIcewalker2 = icewalker((x2, y2), ori2, sensor2, done2, goal2)
    myIcewalker.search_map(-1, 1)
    myIcewalker2.search_map(-1, 2)

def select_action_testing(state, policy_net):
    with torch.no_grad():
        q_values = policy_net(state)
        action1 = torch.argmax(q_values[:3])
        action2 = torch.argmax(q_values[3:])
    return action1, action2

def test_network():
    """
    Task 5: test your network
    """
    same1 = 0
    same2 = 0
    set_speed(3)
    test()
    _test_network()
    (x1, y1), ori1, sensor1, status1, done1, (x2, y2), ori2, sensor2, status2, done2  = reset_map()

    policy_net = DQN().to(device)# load policy net
    policy_net.load_state_dict(torch.load('policy_net.pth', map_location=device))
    policy_net.eval()
    visited1 = np.zeros((13, 13))
    visited2 = np.zeros((13, 13))
    pre_act1 = np.full_like(np.zeros((13, 13)), -1)
    pre_act2 = np.full_like(np.zeros((13, 13)), -1)
    while not done1 or not done2:
        o_i1 = orientation.index(ori1)
        o_i2 = orientation.index(ori2)
        """
        fill this section to test your network
        """
        # cur_state = np.array((x, y, sensor[0], sensor[1], sensor[2], o_i, same, visited[x][y]), dtype=np.float32)#create your own state
        # cur_state = torch.from_numpy(cur_state)
        cur_state = np.array((x1, y1, sensor1[0], sensor1[1], sensor1[2], o_i1, same1, visited1[x1][y1], pre_act1[x1][y1], x2, y2, sensor2[0], sensor2[1], sensor2[2], o_i2, same2, visited2[x2][y2], pre_act2[x2][y2]), dtype=np.float32) 
        cur_state = torch.from_numpy(cur_state).to(device)

        idx_action = select_action_testing(cur_state, policy_net)
        idx_action1, idx_action2 = idx_action[0].item(), idx_action[1].item()
        action1 = actions[idx_action1]
        action2 = actions[idx_action2]
        # (new_x, new_y), ori, sensor, done = action()
        (new_x1, new_y1), new_ori1, new_sensor1, status1, done1 = action1(1)[:5]
        (new_x2, new_y2), new_ori2, new_sensor2, status2, done2 = action2(2)[5:]

        # if visited1[new_x1][new_y1]<3:
        visited1[new_x1][new_y1]+=1
        if (new_x1, new_y1) == (x1, y1):
            if same1<3:
                same1+=1
        else:
            same1=0
        
        # if visited2[new_x2][new_y2]<3:
        visited2[new_x2][new_y2]+=1
        if (new_x2, new_y2) == (x2, y2):
            if same2<3:
                same2+=1
        else:
            same2=0
        if visited1[new_x1][new_y1] > 1:
            pre_act1[x1][y1] = idx_action1
        if visited2[new_x2][new_y2] > 1:
            pre_act2[x2][y2] = idx_action2

        (x1, y1), ori1, sensor1 = (new_x1, new_y1), new_ori1, new_sensor1
        (x2, y2), ori2, sensor2 = (new_x2, new_y2), new_ori2, new_sensor2
        if (done1 == True) and (done2 == True):
            break

# _train()
test_network()


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
# 			if event.key == pygame.K_LEFT: print(turn_left(1))
# 			if event.key == pygame.K_RIGHT: print(turn_right(1))
# 			if event.key == pygame.K_UP: print(move_forward(1))
# 			if event.key == pygame.K_a: print(turn_left(2))
# 			if event.key == pygame.K_d: print(turn_right(2))
# 			if event.key == pygame.K_w: print(move_forward(2))
# 			if event.key == pygame.K_t: test()
# 			if event.key == pygame.K_r: print(reset_map())
# 			if event.key == pygame.K_q: exit("Closing...")