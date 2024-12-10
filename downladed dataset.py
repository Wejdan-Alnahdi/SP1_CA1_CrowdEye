
from roboflow import Roboflow
rf = Roboflow(api_key="877aMmp2fQj8YWvTzMPs")
project = rf.workspace("embien-7xos5").project("peoplecounter-xqtwr")
version = project.version(1)
dataset = version.download("yolov11")