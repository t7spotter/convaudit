import json, tempfile
from pathlib import Path
import numpy as np, cv2
import datumaro as dm

tmp = Path(tempfile.mkdtemp())
(tmp / "src" / "annotations").mkdir(parents=True)
(tmp / "src" / "images" / "val").mkdir(parents=True)
cv2.imwrite(str(tmp / "src" / "images" / "val" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
coco = {
    "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
    "categories": [{"id": 1, "name": "thing"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [10.5, 20.25, 30, 40], "area": 1200.0,
         "segmentation": [[10.5, 20.25, 40.5, 20.25, 40.5, 60.25, 10.5, 60.25]]},
        {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 1, "bbox": [50, 50, 40, 40], "area": 1600.0,
         "segmentation": {"size": [100, 100], "counts": [5050] + [40, 60] * 39 + [40, 1010]}},
    ],
}
(tmp / "src" / "annotations" / "instances_val.json").write_text(json.dumps(coco))
ds = dm.Dataset.import_from(str(tmp / "src"), "coco_instances")
ds.export(str(tmp / "voc"), "voc_detection", save_media=False)
print(next((tmp / "voc").rglob("a.xml")).read_text())
ds.export(str(tmp / "yolo"), "yolo", save_media=False)
print(next((tmp / "yolo").rglob("a.txt")).read_text())
