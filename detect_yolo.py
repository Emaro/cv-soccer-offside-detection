import torch
import torchvision
import numpy as np
import cv2

class YOLOModel:
	def __init__(self, yolo_path, model_path):
		# self.model = torch.hub.load(yolo_path, 'custom', path = model_path, force_reload=True)
		self.cnt = 1

	def track_yolo(self, img):
		f = open("labels/video_" + str(self.cnt) + ".txt", "r");
		self.cnt += 1;

		pos_list = [];
		
		while True:
			line = f.readline();
			if not line : break;
			
			team, x, y, w, h = line.split(" ");

			team = int(team);
			x = (float(x)) * img.shape[1];
			y = (float(y)) * img.shape[0];

			pos_list.append((x, y, team));

		f.close();

		return np.array(pos_list);

	# def track_yolo(self, img):
	# 	# img = np.pad(img, ((4, 4), (0, 0), (0, 0)), 'constant').astype(np.float32);
	# 	# img /= 255;
	# 	# img = torch.from_numpy(img).to("cuda:0");

	# 	img = cv2.resize(img, (640, 640));
	# 	tts = torchvision.transforms.ToTensor();
	# 	img = tts(img).to("cuda:0");
	# 	img = img[None]
	# 	results = self.model(img)

	# 	print(img.shape)
	# 	# For Debugging
	# 	# results[0].print()  # print results to screen
	# 	# results[0].show()  # display results

	# 	# Data
	# 	player_pos_full = results.xyxy[0]
	# 	print(player_pos_full)
	# 	player_pos_full = player_pos_full.to_numpy()

	# 	num_player = player_pos_full.shape[0]

	# 	player_pos = np.zeros((num_player,4))

	# 	for i in range(num_player):
	# 		x1 = player_pos_full[i,0]
	# 		x2 = player_pos_full[i,2]
	# 		y1 = player_pos_full[i,1]
	# 		y2 = player_pos_full[i,3]
	# 		confidence = player_pos_full[i,4]
	# 		x = (x1+x2)/2
	# 		y = (y1+y2)/2

	# 		#which_team = player_pos_full[i,5] # 0 for white team, 1 for blue team
	# 		if (player_pos_full[i, 5] == 0) : which_team = 0;
	# 		else :
	# 			x_idx = np.floor(x).astype(np.int);
	# 			y_idx = np.floor(y).astype(np.int);
	# 			color_sum = np.sum(img[y_idx - 1 : y_idx + 2, x_idx - 1 : x_idx + 2, :]);
	# 			which_team = 1 if color_sum > 3000 else 2;

	# 		player_pos[i,0] = x
	# 		player_pos[i,1] = y
	# 		player_pos[i,2] = which_team
	# 		player_pos[i,3] = confidence

	# 	player_pos.view('f4,f4,f4,f4').sort(order=['f3'], axis=0)
	# 	player_pos = player_pos[::-1]
	# 	#print(player_pos)

	# 	return player_pos[:, 0:3]





	
	
