import json, tempfile
from pathlib import Path
import numpy as np, cv2
import fiftyone as fo

tmp = Path(tempfile.mkdtemp()); (tmp / "data").mkdir()
cv2.imwrite(str(tmp / "data" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "thing"}],
        "annotations": [
          {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [29, 57, 29, 14], "area": 406.0},
          {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 1, "bbox": [60, 10, 30, 30], "area": 900.0}]}
(tmp / "ann.json").write_text(json.dumps(coco))
ds = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset, data_path=str(tmp / "data"),
                         labels_path=str(tmp / "ann.json"), label_types="detections", label_field="gt")
ds.export(export_dir=str(tmp / "voc"), dataset_type=fo.types.VOCDetectionDataset,
          label_field="gt", export_media=False)
print(next((tmp / "voc").rglob("a.xml")).read_text())
print(29 / 100 * 100, int(29 / 100 * 100))
ds.delete()
