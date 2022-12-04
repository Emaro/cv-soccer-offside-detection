import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
#from main import frame

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_player.pt', force_reload=True)

img = cv2.imread('soccer_0.jpg')
#img = Image.fromarray(main.frame) #Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

results = model(img)  # includes NMS

# For Debugging
#results.print()  # print results to screen
#results.show()  # display results

# Data
player_pos_full = results.pandas().xyxy[0]
print(player_pos_full)
player_pos_full = player_pos_full.to_numpy()

num_player = player_pos_full.shape[0]

player_pos = np.zeros((num_player,4))

for i in range(num_player):
	x1 = player_pos_full[i,0]
	x2 = player_pos_full[i,2]
	y1 = player_pos_full[i,1]
	y2 = player_pos_full[i,3]
	which_team = player_pos_full[i,5] # 0 for white team, 1 for blue team
	confidence = player_pos_full[i,4]
	x = (x1+x2)/2
	y = (y1+y2)/2
	player_pos[i,0] = x
	player_pos[i,1] = y
	player_pos[i,2] = which_team
	player_pos[i,3] = confidence

player_pos.view('f4,f4,f4,f4').sort(order=['f3'], axis=0)
player_pos = player_pos[::-1]
print(player_pos)




	
	
