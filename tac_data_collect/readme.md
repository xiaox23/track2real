# This is a readme about how to collect tactile data.

## Open roscore
```bash
roscore
```

## Publish the tactile data through ROS
```bash
python catkin_ws/src/gelsight_mini_ros/scripts/gs_node.py
```

## Run the collecting script
```bash
python tac_data_collect/tac_data_collect.py
```

## Same time, Run the visualizing script
```bash
python catkin_ws/src/gelsight_mini_ros/scripts/listener.py
```

## After collecting, Run the plotting script
```bash
python tac_data_collect/tac_data_plot.py
```