import json, tempfile
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu
import supervision as sv

W = H = 60
poly = [20.3, 20.6, 40.2, 21.1, 39.7, 40.4, 21.2, 39.8]
rle = mu.merge(mu.frPyObjects([poly], H, W)); rle["counts"] = rle["counts"].decode()
src = mu.decode(mu.frPyObjects([poly], H, W)).max(2)   # COCO's own rasterization

for name, seg in [("polygon", [poly]), ("RLE, same pixels", rle)]:
    d = Path(tempfile.mkdtemp()); (d / "img").mkdir()
    cv2.imwrite(str(d / "img" / "a.jpg"), np.zeros((H, W, 3), np.uint8))
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}],
            "categories": [{"id": 1, "name": "x"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                             "bbox": [20.3, 20.6, 19.9, 19.8], "area": 1.0, "segmentation": seg}]}
    (d / "a.json").write_text(json.dumps(coco))
    ds = sv.DetectionDataset.from_coco(str(d / "img"), str(d / "a.json"), force_masks=True)
    ds.as_yolo(annotations_directory_path=str(d / "y"), data_yaml_path=str(d / "y.yaml"))
    xy = (np.array((d / "y" / "a.txt").read_text().split()[1:], float).reshape(-1, 2) * [W, H]).ravel().tolist()
    out = mu.decode(mu.frPyObjects([xy], H, W)).max(2)
    print(f"{name}: source {src.sum()} px, supervision mask {ds[0][2].mask[0].sum()} px, "
          f"YOLO polygon {out.sum()} px, IoU {(src & out).sum() / (src | out).sum():.3f}")
