import cv2
import os
import numpy as np
from detect_player import YOLOModel

###
### PARAMETERS
###

video_path              = "./data/video.mp4";   # File to read
yolo_path               = "ultralytics/yolov5";
player_model            = "best_player.pt";

num_player              = 20;                   # Number of players to track, default: 20

dist_max_player         = 1E+10;                # Max displacement of player between frames
dist_max_ball           = 1E+1;                 # Max distance between player and the ball (possession)
acceleration_threshold  = 1E+1;                 # Threshold to detect interference

line_compute_freq       = 10;                   # Do line tracking every ~ frames

###
### GET TRACKED COORDINATES
### these functions should get positions/lines from what you implemented
###

def get_lines(frame):

    rho_list = np.empty(4);
    theta_list = np.empty(4);

    return rho_list, theta_list;

# compute vanishing point here
# (I think there should be only one relevant vanishing point)
# (used for getting distances from player coordinates)
def get_vanishing_point(rho_list, theta_list):

    point = np.zeros(2);
    return point;

###
### BALL CLASS
### stores, updates ball position, velocity, acceleration
###

class Ball:
    def __init__(self, frame):
        self.pos = get_pos_ball(frame);
        self.vel = np.zeros(2);
        self.acc = np.zeros(2);

    def update(self, frame):
        pos = get_pos_ball(frame);

        new_vel  = pos - self.pos;
        self.acc = new_vel - self.vel;        
        self.vel = new_vel;
        self.pos = pos;

    def is_played(self):
        return np.linalg.norm(self.acc) > acceleration_threshold;

###
### PLAYERS CLASS
### handles players' positions + line information
###

class Players:
    def __init__(self, frame):
        # initialize player yolo
        self.model = PlayerModel(yolo_path, player_model);

        # set player positions
        self.pos = self.model.track_player_yolo(frame);

        # fill pos if less than num_players detected.
        # count players and assign team with less players
        while (len(self.pos) < num_player) :
            team_0 = 0;
            team_1 = 0;
            for i in range(len(self.pos)) :
                if (self.pos[i, 2] == 0) : team_0 += 1;
                else : team_1 += 1;

            new_team = 0 if team_0 < team_1 else 1;
            self.pos = np.append(self.pos, 
                                 np.array([[frame.shape[0] / 2, frame.shape[1] / 2, new_team]]));

        # set vanishing point info and horizontal line
        self.vpo = np.zeros(2);

        self.left  = np.array([frame.shape[0] / 2, 0]);
        self.right = np.array([frame.shape[0] / 2, frame.shape[1]]);

        self.update_line(frame);

        # set dist from goal
        self.dist = np.zeros(num_player);
        self.update_dist();

        # distance, player snapshot for detection
        self.dist_prev = self.dist.copy();
        self.idx_prev = -1;

        return;

    # Update player positions from new tracked coordinates
    # Assumes pos has num_player(20) rows
    # Chooses nearest point
    def update_players(self, frame):
        tracked = self.model.track_player_yolo(frame);

        for i in range(num_player):
            dist_min = dist_max_player;
            for j in range(len(tracked)):
                if (tracked[j, 2] == self.pos[i, 2]):
                    dist_cur = np.linalg.norm(self.pos[i, 0:2] - tracked[j, 0:2]);
                    if (dist_cur < dist_min):
                        self.pos[i, 0:2] = tracked[j, 0:2];
                        tracked[j, 2] = -1; # set team to -1 to prevent duplicates

        return;

    # Update line and vanishing point informations
    # Can only be done once every few frames to save time ?
    # Also set left and right: horizontal lines
    def update_line(self, frame):
        rho_list, theta_list = get_lines(frame);
        self.vpo = get_vanishing_point(rho_list, theta_list);

    # Transform player coordinates to relevant distances
    def update_dist(self):
        
        for i in range(num_player):
            self.dist[i] = self.pos[0] - (self.vpo[0] - self.pos[0]) * self.pos[1] / (self.vpo[1] - self.pos[1]);

        return;

    # return index of the closest player to the ball
    def get_closest(self, ball):
        dist_min = dist_max_ball;
        idx = -1;
        
        for i in range(num_player):
            dist_cur = np.linalg.norm(self.pos[i, 0:1] - ball.pos);
            if (dist_cur < dist_min):
                idx = i;
                dist_min = dist_cur;

        return idx;

    # return if idx_prev and idx point to players of the same team
    def same_team(self, idx):
        return self.pos[self.idx_prev, 2] == self.pos[idx, 2];

    # return if idx points to a player that was offside at the snapshot
    def was_offside(self, idx):

        #####################################
        ##################################### TODO
        #####################################

        return False;

###
### MAIN FUNCTION
###

def main():
    # Open video file
    video_file = cv2.VideoCapture(video_path);

    # Initialize
    read_success, frame = video_file.read();

    ball = Ball(frame);
    players = Players(frame);

    read_success, frame = video_file.read();
    
    frame_count = 0;
    # Detection loop
    while (read_success) :
        frame_count += 1;

        # Update positions
        ball.update();
        players.update_players();

        if (frame_count % line_compute_freq == 0) :
            players.update_line();

        players.update_dist();

        # If somebody touches the ball
        if (ball.is_played()) :
            player_idx = players.get_closest(ball);

            # player idx changed between same team: pass is completed
            if ((player_idx != players.idx_prev) 
                and (players.same_team(player_idx))
                and (players.was_offside(player_idx))) :

                print("Offside detected at frame " + frame_count + "\n");

        read_success, frame = video_file.read();

    return 0;

if __name__ == '__main__' :
    main();
