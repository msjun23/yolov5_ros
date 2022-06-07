# yolov5_ros
yolov5 with ROS

# Environment
|||
|---|---|
|**OS**|Ubuntu 18.04|
|**CUDA**|10.2|
|**Pytorch version**|1.8|
|**ROS version**|Melodic|
|**Language**|Python3|

Based on [YOLOv5](https://github.com/ultralytics/yolov5).

---

# How to use
1. Locate weight files on [weights folder](https://github.com/msjun23/yolov5_ros/tree/main/weights), data.yaml files on [data folder](https://github.com/msjun23/yolov5_ros/tree/main/data)(maybe data.yaml file is not necessary).

2. Edit your [detector.launch](https://github.com/msjun23/yolov5_ros/blob/main/launch/detector.launch) file.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<launch>
    <arg name="weights_name"    default="yolov5s_door.pt" />
    <arg name="data_name"       default="door_handle.yaml" />

    <!-- Camera topic and weights, config arguments -->
    <arg name="image_topic"     default="/camera/color/image_raw" />
    <arg name="weights"         default="$(find yolov5_ros)/weights/$(arg weights_name)" />
    <arg name="data"            default="$(find yolov5_ros)/data/$(arg data_name)" />
    <arg name="width"           default="640" />
    <arg name="height"          default="480" />
    <arg name="conf_thres"      default="0.25" />

    <!-- Node -->
    <node name="detector" pkg="yolov5_ros" type="detect.py" output="screen" respawn="true">
        <param name="image_topic"   value="$(arg image_topic)" />
        <param name="weights"       value="$(arg weights)" />
        <param name="data"          value="$(arg data)" />
        <param name="width"         value="$(arg width)" />
        <param name="height"        value="$(arg height)" />
        <param name="conf_thres"    value="$(arg conf_thres)" />
    </node>
</launch>

```

Change arguments, "weights_name", "data_name" to yours. Also change "image_topic" too. Or you can just declare argument when you launch the launch file.

```bash
$ roslaunch yolov5_ros detector.launch weights_name:=${your weight file name} data_name:=${your yaml data file name} image_topic:=${image topic name}
```

# Published topic
- /detected_img
> sensor_msgs/Image

- /bounding_box_array
> yolov5_ros/BoundingBoxes
>> header
>>
>> bounding_boxes[]
>>> string Class
>>>
>>> float64 probability
>>>
>>> int64 xmin
>>>
>>> int64 ymin
>>>
>>> int64 xmax
>>>
>>> int64 ymax

# TensorRT
If you want to use tensorrt model, file name "~~.engine", you have to check tensorrt version. If tensorrt version is mismatched, would face **serialization** error like this.

```bash
Serialization (Serialization assertion safeVersionRead == safeSerializationVersion failed.Version tag does not match.
```

So, it is recommended that make your own tenorrt model at your PC, with following this [doc](https://github.com/ultralytics/yolov5/issues/251).

