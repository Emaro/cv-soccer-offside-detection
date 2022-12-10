import cv2
import numpy as np

from detect_yolo import YOLOModel
from detect_field import get_vanishing_point

###
### PARAMETERS
###

video_path              = "./video.mp4";   # File to read
yolo_path               = "ultralytics/yolov5";
player_model            = "models/best_small.pt";

dist_max_player         = 3E+1;                 # Max displacement of player between frames
dist_max_ball           = 1E+1;                 # Max distance between player and the ball (possession)
acceleration_threshold  = 1E+1;                 # Threshold to detect interference

line_compute_freq       = 1;                   # Do line tracking every ~ frames

TEAM_BALL   = 0;
TEAM_FIRST  = 1;
TEAM_SECOND = 2;

###
### Yolo CLASS
### handles players' positions + line information
###

class Yolo:
    def __init__(self, frame):
        # initialize player yolo
        self.model = YOLOModel(yolo_path, player_model);

        # initialize player and ball positions
        tracked = self.model.track_yolo(frame);

        bidx = -1;
        self.bpos = np.zeros(2);
        self.bvel = np.zeros(2);
        self.bacc = np.zeros(2);
        for i in range(len(tracked)):
            if (tracked[i, 2] == TEAM_BALL):
                self.bpos = tracked[i, 0:1];
                bidx = -1;

        if (bidx == -1) : raise ValueError;

        self.pos = tracked[:bidx, :];
        self.pos = np.concatenate((self.pos, tracked[bidx + 1:, :]));

        self.pidx = self.get_closest(self.bpos);
        
        # set dist from goal
        self.dist = np.zeros(len(self.pos));

        # distance, player snapshot for detection
        self.dist_prev = self.dist.copy();
        self.pidx_prev = -1;

        return;

    # Update ball + player positions from new tracked coordinates
    # Assumes pos has num_player(20) rows
    # Chooses nearest point
    def update(self, frame):

        # do track
        tracked = self.model.track_yolo(frame);

        # update players on pos list
        for i in range(len(self.pos)):
            dist_min = dist_max_player;
            for j in range(len(tracked)):
                if (tracked[j, 2] == self.pos[i, 2]):
                    dist_cur = np.linalg.norm(self.pos[i, 0:2] - tracked[j, 0:2]);
                    if (dist_cur < dist_min):
                        self.pos[i, 0:2] = tracked[j, 0:2];
                        tracked[j, 2] = -1; # set team to -1 to prevent duplicates
        
        # add new players and update ball
        found_ball = False;
        bpos = np.zeros(2);
        for i in range(len(tracked)):
            if (tracked[i, 2] == TEAM_BALL):
                bpos = tracked[i, 0:2];
                found_ball = True;

            if (tracked[i, 2] != -1):
                self.pos = np.append(self.pos, tracked[i]);

        if (found_ball):
            pidx_new = self.get_closest(bpos);
            if (self.pidx != pidx_new):
                self.pidx_prev = self.pidx;
                self.pidx = pidx_new;
                self.passed = self.same_team();

        elif (self.pidx != -1):
            bpos = tracked[self.pidx, 0:2];
            
        new_vel   = bpos - self.bpos;
        self.bacc = new_vel - self.bvel;        
        self.bvel = new_vel;
        self.bpos = bpos;
                
        return;

    # Transform player coordinates to relevant distances
    def update_dist(self, vpo):
        
        for i in range(len(self.pos)):
            self.dist[i] = self.pos[0] - (vpo[0] - self.pos[i][0]) * self.pos[i][1] / (vpo[1] - self.pos[i][1]);

        return;

    # return index of the closest player to the ball
    def get_closest(self, bpos):
        dist_min = dist_max_ball;
        idx = -1;
        
        for i in range(len(self.pos)):
            dist_cur = np.linalg.norm(self.pos[i, 0:2] - bpos);
            if (dist_cur < dist_min):
                idx = i;
                dist_min = dist_cur;

        return idx;

    # return if idx_prev and idx point to players of the same team
    def same_team(self):
        return self.pos[self.pidx_prev, 2] == self.pos[self.pidx, 2];

    # return if idx points to a player that was offside at the snapshot
    def was_offside(self, idx):

        #####################################
        ##################################### TODO
        #####################################

        team_idx = TEAM_FIRST if self.pos[idx, 2] == TEAM_SECOND else TEAM_SECOND;
        
        defender_y = 1E+10;

        for i in range(len(self.dist_prev)):
            if (self.pos[i, 2] != team_idx) : continue;
            if (self.dist_prev[i] < defender_y) :
                defender_y = self.dist_prev[i];

        return self.dist_prev[idx] < defender_y;

    def ball_played(self):
        return self.bacc > acceleration_threshold;

    def store_dist(self):
        self.dist_prev = self.dist;
        return;

###
### MAIN FUNCTION
###

def main():
    # Open video file
    video_file = cv2.VideoCapture(video_path);

    # Initialize
    read_success, frame = video_file.read();

    tracker = Yolo(frame);
    vpo = get_vanishing_point(frame);

    read_success, frame = video_file.read();
    
    frame_count = 0;
    # Detection loop
    while (read_success) :
        frame_count += 1;

        # Update vanishing point
        if (frame_count % line_compute_freq == 0) :
            vpo_t = get_vanishing_point(frame);
            if (vpo_t is not None) : vpo = vpo_t;

        #print(vpo);

        # Update positions
        tracker.update(frame);
        tracker.update_dist(vpo);

        # If somebody touches the ball
        if (tracker.ball_played()) :
            # player idx changed between same team: pass is completed
            if (tracker.passed and
                tracker.was_offside(tracker.pidx_prev)):

                print("Offside detected at frame " + frame_count + "\n");

            tracker.store_dist();

        read_success, frame = video_file.read();

    return 0;

if __name__ == '__main__' :
    main();
