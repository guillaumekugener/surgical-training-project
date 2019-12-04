# surgical-training-project

The videos are broken down in chunks of 1000 frames per chunk.

## TASK CONFIGURATION

The fields below should be copied into CVAT when creating a new task.
```
Name: ConferenceFolder-FolderID-VideoName #omit .mp4 (e.g. "ParkCity2019-170211-ch1_video_03-frames_NNNNNNN_NNNNNNNN") 
Labels: tool ~radio=type:__undefined__,grasper,cottonoid,muscle,suction,drill,string
        frame ~radio=type:start,stop
Bug Tracker:
Dataset Repository: https://github.com/guillaumekugener/surgical-training-project.git
Source: Local 
Z-Order: n/a
Overlap Size: n/a
Segment Size: 5000 #is this default? capped? what does this represent? 
Start Frame: n/a
Stop Frame: n/a
Frame Filter: n/a
```
