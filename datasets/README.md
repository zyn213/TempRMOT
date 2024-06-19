## Prepare Refer-KITTI Data
### Refer-KITTI-V2
Detailed steps are shown as follows.

1. Download the official KITTI images from [official website](https://www.cvlibs.net/datasets/kitti/eval_tracking.php).

2. Download our created expression and labels_with_ids from [Google Drive](https://drive.google.com/drive/folders/1eaxuRK-ewl0cpGshOxylSFZ5PPu3_WUT?usp=sharing).


The directory structure should be as below.
```
.
├── refer-kitti-v2
│   ├── KITTI
│           ├── training
│           ├── labels_with_ids
│   └── expression
```
Note: 
- Our expression (.json) contains corresponding object ids, and the corresponding boxes can be found in 'labels_with_ids' using these ids.
- We have **corrected and regenerated** labels_with_ids, so please download from Google Drive instead of using data from Refer-KITTI.

### Refer-KITTI
Please refer to the [official website](https://github.com/wudongming97/RMOT) to downloading and organization the Refer-KITTI dataset.

## Data Format
Each manually annotated expression file is structured like this :
```json
{"label": {"frame_id_start": ["object_ids"],"frame_id": ["object_ids"],"frame_id_end": ["object_ids"]}, "ignore": [], "video_name": "", "sentence": ""}
```

And each expression extended through **GPT-3.5** is structured like this :
```json
{"label": {"frame_id_start": ["object_ids"],"frame_id": ["object_ids"],"frame_id_end": ["object_ids"]}, "ignore": [], "video_name": "", "sentence": "", "raw_sentence": ""}
```
