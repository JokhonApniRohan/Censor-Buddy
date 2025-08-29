from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("WORKSPACE_NAME").project("PROJECT_NAME")
dataset = project.version(VERSION_NUMBER).download("yolov8")
