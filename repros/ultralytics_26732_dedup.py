import json, tempfile
from pathlib import Path
from ultralytics.data.converter import convert_coco

tmp = Path(tempfile.mkdtemp()); (tmp / "ann").mkdir()
sq = [[20, 20, 30, 20, 30, 30, 20, 30]]
ann = lambda i: {"id": i, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [20, 20, 10, 10], "area": 100.0, "segmentation": sq}
coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "thing"}], "annotations": [ann(1), ann(2)]}
(tmp / "ann" / "instances_val.json").write_text(json.dumps(coco))
convert_coco(labels_dir=str(tmp / "ann"), save_dir=str(tmp / "out"), use_segments=True, cls91to80=False)
print(len(next((tmp / "out").rglob("a.txt")).read_text().splitlines()), "line(s) for 2 annotations")
