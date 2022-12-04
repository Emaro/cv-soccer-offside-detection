import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from main import frame

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_ball.pt', force_reload=True)

#img = cv2.imread('soccer_0.jpg')
img = Image.fromarray(main.frame) #Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

results = model(img)  # includes NMS

# For Debugging
#results.print()  # print results to screen
#results.show()  # display results

# Data
ball_pos_full = results.pandas().xyxy[0]
ball_pos_full = ball_pos_full.to_numpy()

num_ball = ball_pos_full.shape[0]

ball_pos = np.zeros(2)

x1 = ball_pos_full[0]
x2 = ball_pos_full[2]
y1 = ball_pos_full[1]
y2 = ball_pos_full[3]
x = (x1+x2)/2
y = (y1+y2)/2

ball_pos[0] = x
ball_pos[1] = y

	
	
