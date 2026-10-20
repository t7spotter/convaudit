import json, tempfile
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from ultralytics.data.converter import convert_coco

tmp = Path(tempfile.mkdtemp()); (tmp / "ann").mkdir()
tri = [[10, 10, 90, 10, 10, 90]]
m = mu.decode(mu.frPyObjects(tri, 100, 100))[:, :, 0]
rle = mu.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode()
coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "thing"}],
        "annotations": [
          {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [10, 10, 80, 80], "area": 3200.0, "segmentation": tri},
          {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [10, 10, 80, 80], "area": 3200.0, "segmentation": rle}]}
(tmp / "ann" / "instances_val.json").write_text(json.dumps(coco))
convert_coco(labels_dir=str(tmp / "ann"), save_dir=str(tmp / "out"), use_segments=True, cls91to80=False)
print(next((tmp / "out").rglob("a.txt")).read_text())
