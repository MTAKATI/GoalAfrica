from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np  
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    #Read Video
    video_frames = read_video('input_videos/3.mp4')
    print(f"Loaded {len(video_frames)} frames.")

    if len(video_frames) == 0:
        print("Error: could not read video! Check if 'input_videos/3.mp4' exists")
        return

    #Initialise Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')

    # Get object position 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pk1')

    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    
    # 1. Train cluster on up to 100 frames
    team_assigner.assign_team_colors(video_frames, tracks, sample_frames=100)
    
    # 2. Assign persistent teams using majority voting
    team_assigner.assign_teams_by_majority_vote(video_frames, tracks)

    # Assign Ball to Player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_dict in enumerate(tracks['players']):
        # Reset possession for this frame
        for player_id in player_dict:
            tracks['players'][frame_num][player_id]['has_ball'] = False
        
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_dict, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no one has the ball, use the last person who had it
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                # Initial state if no one has ball in first frame
                team_ball_control.append(None) 

    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    video_frames = None

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    #Save Video frames
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()