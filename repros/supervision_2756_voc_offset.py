import json, tempfile
from pathlib import Path
import numpy as np, cv2
import supervision as sv

tmp = Path(tempfile.mkdtemp()); (tmp / "images").mkdir()
cv2.imwrite(str(tmp / "images" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "box"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}]}
(tmp / "ann.json").write_text(json.dumps(coco))
ds = sv.DetectionDataset.from_coco(str(tmp / "images"), str(tmp / "ann.json"))
ds.as_pascal_voc(annotations_directory_path=str(tmp / "voc"))
print((tmp / "voc" / "a.xml").read_text())
