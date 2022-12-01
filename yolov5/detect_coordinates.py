import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image

model = torch.hub.load('.', 'custom', 'best.pt', source='local')

# Images
img = Image.open('soccer/images/soccer_0.jpg')  # PIL image
  
imgs = [img]  # batched list of images [img1, img2]

# Inference
results = model(imgs)  # includes NMS

# Results
results.print()  # print results to screen
results.show()  # display results

# Data
#print('\n', results.xyxy[0])
coordinates = results.pandas().xyxy[0]
coordinates = coordinates.to_numpy()

num_obj = coordinates.shape[0]
coord_new = np.zeros((num_obj,3))
for i in range(num_obj):
	x1 = coordinates[i,0]
	x2 = coordinates[i,2]
	y1 = coordinates[i,1]
	y2 = coordinates[i,3]
	which_team = coordinates[i,5] # 0 for white team, 1 for blue team
	x = (x1+x2)/2
	y = (y1+y2)/2
	coord_new[i,0] = x
	coord_new[i,1] = y
	coord_new[i,2] = which_team

print(coord_new)
	
	
