# 总体介绍
运动系统由3个部件组成，分别是：3轴平移台、单轴旋转台和Hand E夹爪。三者均通过串口与上位机通信，上位机通过串口发送指令控制运动系统的运动。

3轴平移台通过厂商预定义的指令格式控制，我们根据协议用Python进行了实现。

单轴平移台在串口之上还需要ModBus协议，厂商提供了库文件，我们通过Python将其封装成了一个类，方便使用。

Hand E夹爪通过串口发送指令控制，指令格式由Robotiq提供，我们参考GitHub的代码将其封装成了一个类。

本仓库暂时不涉及Hand E的使用，只介绍3轴平移台和单轴旋转台的使用。
# 串口设置
因为3个部件均通过USB转串口与上位机通信，我们推荐使用一个USB HUB将3个部件的USB接口连统一连接到上位机（已经连好了，勿动）。
这样做的好处是，如果USB HUB连接上位机的接口是固定的，就可以将3个部件的串口号设置为固定值，方便上位机程序的编写。

给串口设置别名的方法如下：

1. ubuntu上串口的默认名称是`/dev/ttyUSB*`，`*`是一个数字，表示第几个USB转串口。可以通过udev规则来设置串口别名，方便使用。在`/etc/udev/rules.d/`目录下新建一个文件，文件名随意（如usbnames.rules），文件内容如下：
```
KERNEL=="ttyUSB*", ATTRS{devpath}=="11.3.1", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6015", MODE:="0777", SYMLINK+="hande"
KERNEL=="ttyUSB*", ATTRS{devpath}=="11.3.2", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE:="0777", SYMLINK+="rotation_stage"
KERNEL=="ttyUSB*", ATTRS{devpath}=="11.3.3", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6015", MODE:="0777", SYMLINK+="translation_stage"
```
上面的规则中，`ATTRS{devpath}`的值是通过`udevadm info -a -n /dev/ttyUSB0 | grep devpath`命令得到的，`SYMLINK`的值就是别名，可以自己随意设置。
该规则的意思是，当USB转串口的`devpath`为`11.3.1`时，将其别名设置为`hande`，当`devpath`为`11.3.2`时，将其别名设置为`rotation_stage`，当`devpath`为`11.3.3`时，将其别名设置为`translation_stage`。
2. 重启udev服务：`sudo service udev restart`
3. 拔掉USB转串口再插上
4. 通过`ls -l /dev/translation_stage`命令查看别名是否设置成功，如果成功，会显示`/dev/translation_stage -> ttyUSB*`，其中`/dev/translation_stage`就是别名，`ttyUSB*`是默认名称。


# 旋转台的库文件使用方法
1. 编译生成可以被Python调用的库文件
目录RotationStageLibs下是旋转台的库文件，包括头文件和动态链接库文件。我们在StageController目录下使用C++的Boost库对其进行了封装，使其可以被Python调用。
在StageController目录下运行complie_cpp_to_python_library文件中的指令：

```bash
g++ -I /usr/include/python3.8 -fPIC rotation_stage_py.cpp -lboost_python38 -L ../RotationStageLibs/ -lGlobal -lNiMotionMotorSDK -l ModbusMaster -lQt5Core -shared -o rotation_stage.so
```
即可生成rotation_stage.so文件，该文件可以被Python调用。
2. 使用Python调用库文件
打开rotation_stage_py.py文件，在运行配置中编辑环境变量，将旋转台厂商提供的库文件目录添加到LD_LIBRARY_PATH中：
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../RotationStageLibs
```
然后运行rotation_stage_py.py文件，即可控制旋转台的运动。
其他py文件如果需要调用旋转台，也需要将旋转台厂商提供的库文件目录添加到LD_LIBRARY_PATH中。


# 组合位移台的使用

对demo.py文件的环境变量进行同样的设置，然后运行demo.py文件，即可控制组合位移台的运动。