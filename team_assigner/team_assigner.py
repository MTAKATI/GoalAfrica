import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None

    def get_jersey_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1
        
        # Crop upper torso: skip head (top 15%), stop at waist (top 50%)
        # Take central 60% horizontally to exclude background
        y_start = y1 + int(height * 0.15)
        y_end = y1 + int(height * 0.50)
        x_start = x1 + int(width * 0.20)
        x_end = x2 - int(width * 0.20)
        
        crop = frame[y_start:y_end, x_start:x_end]
        return crop

    def extract_dominant_color(self, crop):
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.array([0, 0, 0])
            
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Mask out green grass pixels in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv_crop, lower_green, upper_green)
        
        non_grass_pixels = hsv_crop[grass_mask == 0]
        
        if len(non_grass_pixels) == 0:
            non_grass_pixels = hsv_crop.reshape(-1, 3)

        # Return median HSV color profile
        return np.median(non_grass_pixels, axis=0)

    def assign_team_colors(self, video_frames, tracks, sample_frames=100):
        player_colors = []
        
        # 1. Collect color samples across initial frames
        num_frames = min(sample_frames, len(video_frames))
        for frame_idx in range(num_frames):
            frame = video_frames[frame_idx]
            players = tracks['players'][frame_idx]
            
            for player_id, player in players.items():
                crop = self.get_jersey_crop(frame, player['bbox'])
                color = self.extract_dominant_color(crop)
                player_colors.append(color)
                
        if not player_colors:
            return

        # 2. Fit KMeans on HSV colors
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.kmeans.fit(player_colors)
        
        # Store BGR color approximations for rendering
        for i, center in enumerate(self.kmeans.cluster_centers_):
            hsv_pixel = np.uint8([[center]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
            self.team_colors[i + 1] = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

    def assign_teams_by_majority_vote(self, video_frames, tracks):
        player_votes = defaultdict(list)

        # 1. Gather predictions for each player ID across all frames
        for frame_num, player_dict in enumerate(tracks['players']):
            frame = video_frames[frame_num]
            for player_id, player in player_dict.items():
                crop = self.get_jersey_crop(frame, player['bbox'])
                color = self.extract_dominant_color(crop)
                predicted_team = self.kmeans.predict([color])[0] + 1
                player_votes[player_id].append(predicted_team)

        # 2. Assign majority vote team per track ID
        final_assignments = {}
        for player_id, votes in player_votes.items():
            most_common_team = Counter(votes).most_common(1)[0][0]
            final_assignments[player_id] = most_common_team

        # 3. Update main tracks dictionary
        for frame_num, player_dict in enumerate(tracks['players']):
            for player_id in player_dict:
                team_id = final_assignments[player_id]
                tracks['players'][frame_num][player_id]['team'] = team_id
                tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[team_id]