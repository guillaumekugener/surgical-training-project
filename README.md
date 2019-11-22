# surgical-training-project

Time: 1 minute chunks (Break them down into chunks we can do in 30 min or 1 hour)

## TASK CONFIGURATION

The fields below should be copied into CVAT when creating a new task.
```
Name: ConferenceFolder-FolderID-VideoName #omit .mp4 (e.g. "ParkCity2019-170211-ch1_video_03") 
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
