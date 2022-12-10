import torch
import numpy as np

class YOLOModel:
	def __init__(self, yolo_path, model_path):
		self.model = torch.hub.load(yolo_path, 'custom', path = model_path, force_reload=True)

	def track_yolo(self, img):
		results = self.model(img)  # includes NMS

		# For Debugging
		#results.print()  # print results to screen
		#results.show()  # display results

		# Data
		player_pos_full = results.pandas().xyxy[0]
		#print(player_pos_full)
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
		#print(player_pos)

		return player_pos[:, 0:3]





	
	
