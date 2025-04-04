# track2real
## repo clone (include submodule)
```bash
git clone --recurse-submodules https://github.com/xiaox23/track2real.git
```

## make ros packages
delete the packages about `robotiq_3f`
```bash
cd catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
```

add the path to `~/.bashrc`
```bash
sudo vim ~/.bashrc
source /home/tars/workspace/xx/tactile/track2real/catkin_ws/devel/setup.bash
source ~/.bashrc
```

reopen a terminal

## check all peripheral permissions
```bash
ls -l /dev/ttyUSB*
sudo chmod 777 /dev/ttyUSB*
ls -l /dev/video*
sudo chmod 777 /dev/video*
```
make sure that `xyzc` related to `ttyUSB0`, and the `gripper` related to `ttyUSB1`

## cheke the xyzc movement
