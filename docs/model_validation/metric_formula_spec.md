# Metric formula lock (v4)

| metric_id | formula | spacing_required | empty_mask_behavior | official_parity_status | implementation_path |
|---|---|---|---|---|---|
| dice_local | 2TP/(2TP+FP+FN) on label==1 | false | both empty -> 1.0; pred empty target non-empty -> 0 | LOCAL_ONLY | src/wmh2017/evaluation/voxel_metrics.py |
| hd95_local_mm | symmetric HD95 in mm | true | invalid if spacing missing | LOCAL_ONLY | src/wmh2017/evaluation/voxel_metrics.py |
| avd_local_percent | abs(V_pred-V_gt)/V_gt*100 | true | zero-volume policy documented in code | LOCAL_ONLY | src/wmh2017/evaluation/voxel_metrics.py |
| lavd_wmh2017_compat_candidate | candidate only | true | not official lAVD | WMH2017_COMPAT_CANDIDATE | src/wmh2017/evaluation/voxel_metrics.py |
| lesion_recall_local | connected components | false | explicit empty policy | LOCAL_ONLY | src/wmh2017/evaluation/lesion_metrics.py |
| lesion_f1_local | connected components F1 | false | explicit empty policy | LOCAL_ONLY | src/wmh2017/evaluation/lesion_metrics.py |

Metric outputs require: run_id, prediction_manifest_hash, metric_script_hash.
