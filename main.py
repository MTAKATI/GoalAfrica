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
    # 1. Read Video & extract native FPS dynamically
    video_frames, fps = read_video('input_videos/3.mp4')
    print(f"Loaded {len(video_frames)} frames at {fps:.2f} FPS.")

    if len(video_frames) == 0:
        print("Error: could not read video! Check if 'input_videos/3.mp4' exists")
        return

    # Initialise Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')

    # Get object position 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pk1')
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # 2. Speed and Distance Estimator initialized with dynamic FPS
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=fps, window_size=5)
    speed_and_distance_estimator.calculate_speed_and_distance(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colors(video_frames, tracks, sample_frames=100)
    team_assigner.assign_teams_by_majority_vote(video_frames, tracks)

    # Assign Ball to Player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_dict in enumerate(tracks['players']):
        for player_id in player_dict:
            tracks['players'][frame_num][player_id]['has_ball'] = False
        
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_dict, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None) 

    team_ball_control = np.array(team_ball_control)

    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    video_frames = None

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Draw Tactical 2D Pitch Map
    for frame_num in range(len(output_video_frames)):
        output_video_frames[frame_num] = view_transformer.draw_tactical_map(
            output_video_frames[frame_num], tracks, frame_num
        )

    # 3. Save Video frames using source FPS
    save_video(output_video_frames, 'output_videos/output_video.avi', fps=fps)

if __name__ == '__main__':
    main()