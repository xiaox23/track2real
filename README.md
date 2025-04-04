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
- ### absolute_move.py
  测试绝对移动的脚本。

- ### calibrate.py
  标定脚本，暂未标定。

- ### control.py
  xyzc平台的控制函数。

- ### get_info.py
  获取平台xyzc轴的位置信息，速度信息，判断平台有没有移动。

- ### home.py
  根据不同的输入，让xyzc回到零点。

  当输入为`1`时，采用回机械零的方法。

  当输入为`2`时，采用回绝对值零的方法。

- ### absolute_move.py
  测试相对移动的脚本，但是相对移动不能为0，不然会出bug。

- ### move_control.py
  育锋的code。