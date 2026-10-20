import json, tempfile
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu
import supervision as sv

def rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {"size": r["size"], "counts": r["counts"].decode()}

tmp = Path(tempfile.mkdtemp()); (tmp / "images").mkdir()
cv2.imwrite(str(tmp / "images" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
crowd = np.zeros((100, 100)); crowd[40:50, 5:35] = 1
coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
          # one object, two polygons
          {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 200.0, "bbox": [5, 5, 40, 10],
           "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15], [35, 5, 45, 5, 45, 15, 35, 15]]},
          # a crowd region
          {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 1, "area": 300.0, "bbox": [5, 40, 30, 10],
           "segmentation": rle(crowd)}]}
(tmp / "ann.json").write_text(json.dumps(coco))
ds = sv.DetectionDataset.from_coco(str(tmp / "images"), str(tmp / "ann.json"), force_masks=True)
ds.as_yolo(annotations_directory_path=str(tmp / "yolo"), data_yaml_path=str(tmp / "data.yaml"))
print((tmp / "yolo" / "a.txt").read_text())
