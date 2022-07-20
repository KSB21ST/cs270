from simulator import get_sensors, move_forward, move_backward, turn_left, turn_right, submit, set_map
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import "simulator_hidden" or any other functions from "simulator"
import math
from random import randint, uniform
import time
from collections import deque
from itertools import permutations
import sys

# Colors
black = (0,0,0)
white = (255,255,255)
gray = (100,100,100)
blue = (0,180,255)
brown = (120,70,0)


iter_cnt = 0
cr_x_r = [0, 0, 0] #x, y, r
# max_x = 11
# max_y = 7
global touched_down
touched_down = False
global touched_side 
touched_side = False
global max_x 
max_x = 0
global max_y 
max_y = 0
global block_list
block_list = set([])
global lake_list
lake_list = []
global visited_node
visited_node = []

##############################
#### Write your code here ####
##############################

# How to use the functions:
# (left_color, right_color), ir_sensor = get_sensors()
# move_forward(how_much)
# submit(lake_list=[(x1,y1),(x2,y2)...], building_list=[(x1,y1),(x2,y2)...])
# set_map(new_map_dim=(width, heigth), new_lake_list=[(x1,y1),(x2,y2)...], new_building_list=[(x1,y1),(x2,y2)...])
def bfs(x, y, w, h, a):
	global block_list, visited_node, lake_list
	dx = [1, -1, 0, 0]
	dy = [0, 0, 1, -1]
	k = []
	q = deque()
	c = [[0]*w for _ in range(h)]
	q.append([x, y])
	c[y][x] = 1
	while q:
		x, y = q.popleft()
		for i in range(4):
			nx = x + dx[i]
			ny = y + dy[i]
			if 0 <= nx < w and 0 <= ny < h:
				if a[ny][nx] != 'x' and not c[ny][nx]:
					c[ny][nx] = c[y][x] + 1
					q.append([nx, ny])
					k.append([nx, ny])
	return c

def draw_input(w, h):
	global block_list
	global lake_list
	global visited_node
	ans = [[0]*(w) for _ in range(h)]
	for row in range(h):
		for column in range(w):
			if((column, row) in block_list):
				ans[row][column] = 'x'
			else:
				ans[row][column] = '.'
	return ans


def shortest_path(robot_pos, starting):
	global block_list
	global lake_list
	global visited_node
	global max_x, max_y
	while True:
		w, h = max_x+1, max_y+1
		sx, sy = robot_pos[0], robot_pos[1]
		a = draw_input(w, h)
		a[sy][sx] = 'o'
		a[0][0] = '*'

		for i in a:
			print(i)
		d = [[0, 0]]
		c = bfs(sx, sy, w, h, a)
		break
	return c



def speed_check():
	global block_list
	global lake_list
	global visited_node
	i = 0
	right_color, left_color, block_dst = check_sensor()
	while(right_color == white and left_color == white):
		move_forward()
		i += 1
		right_color, left_color, block_dst = check_sensor()
	print(i)
	move_backward(i)
	turn_left(70)
	r = 70
	while True:
		# turn_left()
		# r += 1
		move_forward(i)
		right_color, left_color, block_dst = check_sensor()
		if(right_color == white):
			move_backward(i)
			turn_left()
			r += 1
			continue
		if(right_color == gray and left_color == gray):
			break
	print(r)
	move_backward(i)
	turn_right(r)

	turn_left(r*2)
	right_color, left_color, block_dst = check_sensor()
	t = 0
	while True:
		move_forward(i)
		right_color, left_color, block_dst = check_sensor()
		if(left_color != black):
			move_backward(i)
			turn_right()
			t += 1
			continue
		if(right_color != black):
			move_backward(i)
			turn_left()
			t -= 1
			continue
		if(left_color == right_color):
			break
	move_backward(i)
	turn_right(r*2 - t)
	if (t == 0):
		new_r = r
	else:
		new_r = r - (t//2) +1
	print("changed from ", r, "to " ,new_r)
	# turn_right(r)
	return i, new_r

def check_sensor():
	sensor = get_sensors()
	right_color = sensor[0][1]
	left_color = sensor[0][0]
	block_dst = sensor[1]
	return right_color, left_color, block_dst

def color_loop(color, base_r):
	right_color, left_color, block_dst = check_sensor()
	r_ch = 0
	l_ch = 0
	while True:
		if(right_color == left_color):
			break
		move_backward(base_i*2)
		if(right_color != color):
		# if(right_color == white or right_color == blue):
			turn_left()
			l_ch += 1
		if(left_color != color):
		# if(left_color == white or left_color == blue):
			turn_right()
			r_ch += 1
		move_forward(base_i*2)
		right_color, left_color, block_dst = check_sensor()
	if(r_ch > 3):
		r_ch = r_ch//2
	if(l_ch > 3):
		l_ch = l_ch //2
	if(r_ch>0):
		move_backward(base_i*2)
		turn_right(base_r+2)
		move_forward(r_ch*2)
		turn_left(base_r+2)
		move_forward(base_i*2)
	if(l_ch>0):
		move_backward(base_i*2)
		turn_left(base_r+2)
		move_forward(l_ch*2)
		turn_right(base_r+2)
		move_forward(base_i*2)
	move_backward(base_i)

def check_down(base_i, base_r, cr_x_r):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y

	if touched_down:
		return True
	move_cnt = 0
	check_dir(cr_x_r[2], 0, base_r)
	right_color, left_color, block_dst = check_sensor()
	if(right_color == blue):
		while(right_color == blue and left_color == blue):
			move_forward()
			move_cnt += 1
			right_color, left_color, block_dst = check_sensor()
	elif(right_color == white):
		while(right_color == white and left_color == white):
			move_forward()
			move_cnt += 1
			right_color, left_color, block_dst = check_sensor()
	if (right_color == black or left_color == black):
		# move_backward(move_cnt)
		color_loop(black, base_r)
		touched_down = True
		print("touched down! max_y ", max_y)
		return True
	# move_backward(move_cnt)
	color_loop(gray, base_r)
	max_y += 1
	print("checked down", max_y)
	return False

def check_right(base_i, base_r, cr_x_r):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y

	if touched_side:
		return True
	move_cnt = 0
	check_dir(cr_x_r[2], 3, base_r)
	right_color, left_color, block_dst = check_sensor()
	if(right_color == blue):
		while(right_color == blue and left_color == blue):
			move_forward()
			move_cnt += 1
			right_color, left_color, block_dst = check_sensor()
	elif(right_color == white):
		while(right_color == white and left_color == white):
			move_forward()
			move_cnt += 1
			right_color, left_color, block_dst = check_sensor()
	if (right_color == black or left_color == black):
		color_loop(black, base_r)
		touched_side = True
		print("touched side! max_x ", max_x)
		return True
	print("checked side", max_x)
	color_loop(gray, base_r)
	max_x += 1
	return False

def move_down(base_i, base_r, cr_x_r):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y
	if (touched_down) and (cr_x_r[1] >= max_y):
		return
		
	check_dir(cr_x_r[2], 0, base_r)
	right_color, left_color, block_dst = check_sensor()
	move_forward(base_i*2 + 2)
	cr_x_r[1] += 1
	right_color, left_color, block_dst = check_sensor()
	if(right_color == blue and left_color == blue):
		if((cr_x_r[0], cr_x_r[1]) not in lake_list):
			lake_list.append((cr_x_r[0], cr_x_r[1]))
		while(right_color == blue and left_color == blue):
			move_forward()
			right_color, left_color, block_dst = check_sensor()
		color_loop(gray, base_r)
		return
	while(right_color == white and left_color == white):
		move_forward()
		right_color, left_color, block_dst = check_sensor()
	if touched_down and (cr_x_r[1] >= max_y):
		color_loop(black, base_r)
		return
	color_loop(gray, base_r)
	return

def move_up(base_i, base_r, cr_x_r):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y
	if(cr_x_r[1] <= 0):
		return

	check_dir(cr_x_r[2], 1, base_r)
	move_forward(base_i*2 + 2)
	cr_x_r[1] -= 1
	right_color, left_color, block_dst = check_sensor()

	if(right_color == blue and left_color == blue):
		if((cr_x_r[0], cr_x_r[1]) not in lake_list):
			lake_list.append((cr_x_r[0], cr_x_r[1]))
		while(right_color == blue and left_color == blue):
			move_forward()
			right_color, left_color, block_dst = check_sensor()
		color_loop(gray, base_r)
		return

	while(right_color == white and left_color == white):
		move_forward()
		right_color, left_color, block_dst = check_sensor()
	if(cr_x_r[1] <= 0):
		color_loop(black, base_r)
		return
	color_loop(gray, base_r)
	return

def move_right(base_i, base_r, cr_x_r):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y
	if(touched_side) and (cr_x_r[0] >= max_x):
		return

	check_dir(cr_x_r[2], 3, base_r)
	move_forward(base_i*2+2)

	cr_x_r[0] += 1
	right_color, left_color, block_dst = check_sensor()

	if(right_color == blue and left_color == blue):
		if((cr_x_r[0], cr_x_r[1]) not in lake_list):
			lake_list.append((cr_x_r[0], cr_x_r[1]))
		while(right_color == blue and left_color == blue):
			move_forward()
			right_color, left_color, block_dst = check_sensor()
		color_loop(gray, base_r)
		return

	while(right_color == white and left_color == white):
		move_forward()
		right_color, left_color, block_dst = check_sensor()
	if touched_side and (cr_x_r[0] >= max_x):
		color_loop(black, base_r)
		return
	color_loop(gray, base_r)
	return


def move_left(base_i, base_r, cr_x_r):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y
	if(cr_x_r[0] <= 0):
		return
	check_dir(cr_x_r[2], 2, base_r)
	move_forward(base_i*2+2)
	cr_x_r[0] -= 1
	right_color, left_color, block_dst = check_sensor()

	if(right_color == blue and left_color == blue):
		if((cr_x_r[0], cr_x_r[1]) not in lake_list):
			lake_list.append((cr_x_r[0], cr_x_r[1]))
		while(right_color == blue and left_color == blue):
			move_forward()
			right_color, left_color, block_dst = check_sensor()
		color_loop(gray, base_r)
		return

	while(right_color == white and left_color == white):
		move_forward()
		right_color, left_color, block_dst = check_sensor()
	if(cr_x_r[0] <= 0):
		color_loop(black, base_r)
		return
	color_loop(gray, base_r)
	return

def fix_dir(base_r):
	if(cr_x_r[2] == 0):
		return
	elif(cr_x_r[2] == 1):
		turn_left(base_r*2)
	elif(cr_x_r[2] == 2):
		turn_left(base_r)
	else:
		turn_right(base_r)
	cr_x_r[2] = 0
	return

def check_block():
	global block_list
	global lake_list
	global visited_node
	right_color, left_color, block_dst = check_sensor()
	if(block_dst < 5 and block_dst > -1):
		if(cr_x_r[2] == 0):
			block_list.add((cr_x_r[0], cr_x_r[1]+1))
			# passed_list[cr_x_r[1]+1][cr_x_r[0]] = 1
			visited_node.append((cr_x_r[0], cr_x_r[1]))
		if(cr_x_r[2] == 1):
			block_list.add((cr_x_r[0], cr_x_r[1]-1))
			# passed_list[cr_x_r[1]-1][cr_x_r[0]] += 1
			visited_node.append((cr_x_r[0], cr_x_r[1]-1))
		if(cr_x_r[2] == 2):
			block_list.add((cr_x_r[0]-1, cr_x_r[1]))
			# passed_list[cr_x_r[1]][cr_x_r[0]-1] += 1
			visited_node.append((cr_x_r[0]-1, cr_x_r[1]))
		if(cr_x_r[2] == 3):
			block_list.add((cr_x_r[0]+1, cr_x_r[1]))
			# passed_list[cr_x_r[1]][cr_x_r[0]+1] += 1
			visited_node.append((cr_x_r[0]+1, cr_x_r[1]))
		return True
	return False

# def print_passed(passed_list):
# 	for i in range(len(passed_list)):
# 		print(passed_list[i])

def sum_passed():
	global block_list
	global lake_list
	global visited_node
	global max_x
	global max_y
	# total = 0
	# for i in range(len(passed_list)):
	# 	total += sum(passed_list[i])
	# if (total == (len(passed_list)*len(passed_list[0]))):
	# 	print(len(passed_list)*len(passed_list[0]))
	# 	return True
	# return False
	return len(visited_node) == (max_x+1)*(max_y+1)

def check_dir(cur_dir, next_dir, base_r):
	global block_list
	global lake_list
	global visited_node
	cr_x_r[2] = next_dir
	if(cur_dir == 0):
		# if(next_dir == 0):
		# 	return
		if(next_dir == 1):
			turn_right(base_r*2)
		if(next_dir == 2):
			turn_right(base_r)
		if(next_dir == 3):
			turn_left(base_r)
	if(cur_dir == 1):
		if(next_dir == 0):
			turn_right(base_r*2)
		# if(next_dir == 1):
		# 	return
		if(next_dir == 2):
			turn_left(base_r)
		if(next_dir == 3):
			turn_right(base_r)
	if(cur_dir == 2):
		if(next_dir == 0):
			turn_left(base_r)
		if(next_dir == 1):
			turn_right(base_r)
		# if(next_dir == 2):
		# 	return
		if(next_dir == 3):
			turn_right(base_r*2)
	if(cur_dir == 3):
		if(next_dir == 0):
			turn_right(base_r)
		if(next_dir == 1):
			turn_left(base_r)
		if(next_dir == 2):
			turn_right(base_r*2)
		# if(next_dir == 3):
		# 	return
	# right_color, left_color, block_dst = check_sensor()
	# if(right_color != left_color):
	# 	move_backward(3)
	# 	turn_right()
	# 	move_forward(3)
	# if(check_block()):
	# 	move_backward(3)
	# 	check_dir(next_dir, cur_dir, base_r)

	return

def opt_dir(cr_dir):
	if(cr_dir == 0):
		return [0, 2, 1, 3]
	if(cr_dir == 1):
		return [1, 2, 0, 3]
	if(cr_dir == 2):
		return [2, 0, 3, 1]
	if(cr_dir == 3):
		return [3, 0, 2, 1]

def surrounded(adj_node):
	global block_list
	global lake_list
	global visited_node
	global touched_down
	global touched_side
	global max_x
	global max_y
	blocked_sum = 0
	for k in range(4):
		i = adj_node[k]
		next_x = i[0]
		next_y = i[1]
		# if (next_x < 0) or (next_y < 0) or (next_x > max_x) or (next_y > max_y):
		# 	blocked_sum += 1
		# 	continue

		if (next_x < 0) or (next_y < 0):
			blocked_sum += 1
			continue
		if touched_down:
			if(next_y > max_y):
				blocked_sum += 1
				continue
		if touched_side:
			if(next_x > max_x):
				blocked_sum += 1
				continue

		if i in block_list:
			blocked_sum += 1
	if blocked_sum >= 4:
		return True
	return False


def dfs_node(cr_x_r, base_i, base_r, parent):
	global touched_down
	global touched_side
	global max_x
	global max_y
	global block_list
	global lake_list
	global visited_node
	if touched_down and touched_side:
		if sum_passed():
			print("all iterated!")
			return
	# if sum_passed(passed_list):
	# 	print("all iterated!")
	# 	return
	# fix_dir(base_r)
	cr_x = cr_x_r[0]
	cr_y = cr_x_r[1]
	cr_dir = cr_x_r[2]
	# opt_d = opt_dir(cr_dir)
	print("dfs node-------------", cr_x, cr_y, cr_dir)
	# passed_list[cr_y][cr_x] = 1
	visited_node.append((cr_x, cr_y))
	# print_passed(passed_list)
	adj_node = [(cr_x, cr_y+1), (cr_x, cr_y-1), (cr_x-1, cr_y), (cr_x+1, cr_y)]

	to_child = -1
	for j in range(4):
		# print("iteration ", j, (cr_x, cr_y))
		if (surrounded(adj_node)):
			print("nowhere to move!")
			return
		i = opt_d[j]
		next_node = adj_node[i]
		next_x, next_y = next_node[0], next_node[1]
		if (next_x < 0) or (next_y < 0):
			continue
		if touched_down:
			if(next_y > max_y):
				continue
		if touched_side:
			if(next_x > max_x):
				continue
		# continue
		# if (passed_list[next_y][next_x] > 0):
		if (next_x, next_y) in visited_node:
			continue

		if(j == 0):
			check_dir(cr_x_r[2], opt_d[j], base_r)
			if(check_block()):
				continue
			else:
				if (cr_x_r[0] == 0):
					continue
				if not touched_side:
					max_x -= 1
				move_left(base_i, base_r, cr_x_r)
				to_child = 3
		if(j == 1):
			check_dir(cr_x_r[2], opt_d[j], base_r)
			if(check_block()):
				continue
			else:
				if not touched_down:
					if(check_down(base_i, base_r, cr_x_r)):
						continue
				move_down(base_i, base_r, cr_x_r)
				to_child = 1
		if(j == 2):
			check_dir(cr_x_r[2], opt_d[j], base_r)
			if(check_block()):
				continue
			else:
				if not touched_side:
					if(check_right(base_i, base_r, cr_x_r)):
						continue
				move_right(base_i, base_r, cr_x_r)
				to_child = 2
		if(j == 3):
			check_dir(cr_x_r[2], opt_d[j], base_r)
			if(check_block()):
				continue
			else:
				if (cr_x_r[1] == 0):
					continue
				if not touched_down:
					max_y -= 1
				move_up(base_i, base_r, cr_x_r)
				to_child = 0
		dfs_node(cr_x_r, base_i, base_r, to_child)
	print("finished iterating ", (cr_x, cr_y))
	if touched_down and touched_side:
		if sum_passed():
			print("all iterated!")
			return
	print("going to parent: ", adj_node[parent])
	if(parent == 0):
		move_down(base_i, base_r, cr_x_r)
	if(parent == 1):
		move_up(base_i, base_r, cr_x_r)
	if(parent == 2):
		move_left(base_i, base_r, cr_x_r)
	if(parent == 3):
		move_right(base_i, base_r, cr_x_r)
	return



# iter_cnt = 0
# cr_x_r = [0, 0, 0] #x, y, r
# # max_x = 11
# # max_y = 7
# touched_down = False
# touched_side = False
# max_x = 0
# max_y = 0
# block_list = set([])
# lake_list = []
# passed_list = [[0 for i in range(max_x+1)] for i in range(max_y + 1)]
opt_d = [2, 0, 3, 1]
while True:
	if(iter_cnt==0):
		base_i, base_r = speed_check()
		# passed_list[0][0] = 1
		visited_node.append((0, 0))
		iter_cnt += 1
	if(iter_cnt == 1):
		dfs_node(cr_x_r, base_i, base_r, -1)
		print("came out of the loop")
		iter_cnt += 1
	if(iter_cnt == 2):
		# block_list = [(8, 0), (10, 2), (5, 5), (2, 7), (4, 7)]
		robot_pos = (cr_x_r[0], cr_x_r[1])
		# robot_pos = (11, 0)
		print("robot_pos: ", robot_pos)
		path = []
		dir_path = []
		c = shortest_path(robot_pos, (0, 0))
		for i in c:
			print(i)
		n_x = 0
		n_y = 0
		path.append((n_x, n_y))
		while True:
			if(c[n_y][n_x] == 1):
				print("at the end")
				break
			if (n_y < max_y) and (c[n_y+1][n_x] == c[n_y][n_x]-1):
				n_y = n_y+1
				print("first ", (n_x, n_y))
				dir_path.append(1)
			elif (n_x < max_x) and (c[n_y][n_x+1] == c[n_y][n_x]-1):
				n_x = n_x + 1
				print("second ", (n_x, n_y))
				dir_path.append(2)
			elif (n_x > 0) and (c[n_y][n_x-1] == c[n_y][n_x]-1):
				n_x = n_x - 1
				print("third ", (n_x, n_y))
				dir_path.append(3)
			elif (n_y > 0) and (c[n_y-1][n_x] == c[n_y][n_x]-1):
				n_y = n_y-1
				print("fourth ", (n_x, n_y))
				dir_path.append(0)
			path.append((n_x, n_y))
			continue
		print(path)
		print(dir_path)

		for move in range(len(dir_path)):
			dir_m = dir_path[len(dir_path) - move - 1]
			if(dir_m == 0):
				move_down(base_i, base_r, cr_x_r)
			if(dir_m == 1):
				move_up(base_i, base_r, cr_x_r)
			if(dir_m == 2):
				move_left(base_i, base_r, cr_x_r)
			if(dir_m == 3):
				move_right(base_i, base_r, cr_x_r)
		iter_cnt += 1
	if(iter_cnt == 3):
		lake_set = set(lake_list)
		lake_list = list(lake_set)
		submit(lake_list, block_list)
		iter_cnt += 1
		break






##############################


#### If you want to try moving around the map with your keyboard, uncomment the below lines 
# import pygame
# while True:
#    i = 0
#    pressed = pygame.key.get_pressed()
#    if pressed[pygame.K_UP]: move_forward()
#    if pressed[pygame.K_DOWN]: move_backward()
#    if pressed[pygame.K_LEFT]: turn_left()
#    if pressed[pygame.K_RIGHT]: turn_right()
#    if pressed[pygame.K_n]: set_map((10,5), [(8,0), (4,9), (2,0), (3,3), (4,1)], [(7,2), (0,1), (2,3)])
#    if pressed[pygame.K_c]: print(get_sensors())
#    # if pressed[pygame.K_n]: speed_check()
#    # if pressed[pygame.K_m]: move_forward(1)
#    for event in pygame.event.get():
#       if event.type == pygame.QUIT:
#          exit("Closing...")