# from ultralytics import YOLO
#
# model = YOLO(r"C:\softwares\yolov8_env64\runs\segment\window_segmentation_model8\weights\best.pt")
#
# model.export(
#     format="onnx",
#     imgsz=320,           # keep full input resolution
#     opset=19,             # latest ONNX opset for precision
#     dynamic=False,         # allows flexible image sizes
#     simplify=False,       # keeps full computation graph (better precision)
#     half=False            # ensures FP32, not FP16 , keep it False for CPU processing
# )

from ultralytics import YOLO


model = YOLO(r"C:\softwares\yolov8_env64\runs\segment\window_segmentation_model8\weights\best.pt")

model.export(
    format="onnx",
    imgsz=736,           # keep full input resolution
    opset = 17,
    dynamic= False,         # allows flexible image sizes
    simplify= True,       # keeps full computation graph (better precision)
    half= False,
    task = "segment"

)

