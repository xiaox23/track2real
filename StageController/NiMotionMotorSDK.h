#ifndef __NIMOTION_MOTOR_SDK_H__
#define __NIMOTION_MOTOR_SDK_H__

#if defined(_WIN32) || defined(__CYGWIN__) || defined(_WIN32_WCE)
#include <winsock2.h>
#include <windows.h>

#ifdef NIMOTIONMOTORSDK_LIBRARY
#define NiMotionMotorAPI __declspec(dllexport)
#else
#define NiMotionMotorAPI __declspec(dllimport)
#endif

#define NIMOTION_CALL __cdecl
#else
#include <unistd.h>
#define NiMotionMotorAPI

#define NIMOTION_CALL

#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *)0)
#endif
#endif

#define BOOL int
#define FALSE 0
#define TRUE 1

#define CHAR char
#define UCHAR unsigned char
#define BYTE unsigned char
#define USHORT unsigned short
#define WORD unsigned short
#define UINT unsigned int
#define DWORD unsigned int
#define PVOID void*

#endif

#pragma pack(push, 4)
//此处为结构体定义

typedef struct _MOTOR_INFO
{
    DWORD nAddr;  //电机地址
    char szSerialNumber[20];    //电机序列号
    char szHardVersion[20];     //硬件版本号
    char szSoftVersion[20];     //软件版本号
} MOTOR_INFO,*P_MOTOR_INFO;

typedef struct _SELFCHECK_RESULT
{
    DWORD nAddr;  //电机地址
    int nResult[4]; //自检结果
} SELFCHECK_RESULT, *P_SELFCHECK_RESULT;

typedef enum _WORK_MODE
{
    POSITION_MODE = 1,
    VELOCITY_MODE = 2,
    GOHOME_MODE = 3
} WORK_MODE;

#pragma pack(pop)


#ifdef __cplusplus
extern "C" {
#endif


/*======================================================单串口函数组=====================================================================*/

/**************************错误码********************************/
/**
 * 0   执行成功
 * 1   不支持的设备类型
 * -1  执行失败
 * -2  电机地址选择错误
 * -3  参数传入错误
 * -4  当前电机运动模式错误
 * -5  电机未在使能状态
 * -7  当前电机不支持的操作
*/

/***********************通信设备操作函数****************************/
/**
 * @brief 打开通信设备
 * @param nType 通信设备类型 (0:RTU 1:TCP，暂不支持)
 * @param strConnectString 连接字符串，描述设备连接参数()
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_openDevice(int nType, const char* strConnectString);

/**
 * @brief 关闭通信设备
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_closeDevice();

/**
 * @brief 指定通信电机类型
 * param nMotorType 通信电机类型 (0:原Modbus电机 1:无刷Modbus电机)
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_specifyMotorType(int nMotorType);

/***********************在线电机管理函数****************************/
/**
 * @brief 扫描电机
 * @param nFromAddr 起始地址
 * @param nToAddr 结束地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_scanMotors(int nFromAddr, int nToAddr);

/**
 * @brief 获取在线电机列表
 * @param pAddrs 电机地址数组指针
 * @param pCount 数量指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getOnlineMotors(int* pAddrs, int* pCount);

/**
 * @brief 判断电机是否在线
 * @param nAddr 电机地址
 * @param pOnline 指针，返回在线状态
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_isMotorOnline(int nAddr, BOOL* pOnline);

/**
 * @brief 获取电机基本信息
 * @param nAddr 电机地址
 * @param pInfo 电机信息结构体指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getMotorInfo(int nAddr, MOTOR_INFO* pInfo);

/**
 * @brief 执行电机自检
 * @param nAddr 电机地址
 * @param pResult 自检结果结构体指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_selfcheck(int nAddr, SELFCHECK_RESULT* pResult);

/**
 * @brief 获取电机最近的报警
 * @param nAddr 电机地址
 * @param pAlarmCode 指针，返回报警值
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getLatestAlarm(int nAddr, int * pAlarmCode);

/**
 * @brief 获取电机故障码
 * @param nAddr 电机地址
 * @param pErrorCode 指针，返回故障码
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getErrorCode(int nAddr, int * pErrorCode);

/**
 * @brief 获取电机历史报警
 * @param nAddr 电机地址
 * @param pAlarmCode 数组指针，返回报警值列表
 * @param pCount 指针，返回报警数量
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getHistoryAlarms(int nAddr, int * pAlarmCode, int * pCount);

/**
 * @brief 清除电机报警
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_clearAlarms(int nAddr);

/**
 * @brief 清除电机故障
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_clearErrorState(int nAddr);

/**
 * @brief 重启电机
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_rebootMotor(int nAddr);

/**********************电机控制函数******************************/

/**
 * @brief 获取电机参数值
 * @param nAddr 电机地址
 * @param nParamID 参数ID
 * @param nBytes 字节数
 * @param pParamValue 指针，返回参数值
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_readParam(int nAddr, int nParamID, int nBytes, int* pParamValue);

/**
 * @brief 设置电机参数值
 * @param nAddr 电机地址
 * @param nParamID 参数ID
 * @param nBytes 字节数
 * @param nParamValue 参数值
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_writeParam(int nAddr, int nParamID, int nBytes, int nParamValue);

/**
 * @brief 保存电机参数
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_saveParams(int nAddr);

/**
 * @brief 恢复电机出厂设置
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_restoreFactorySettings(int nAddr);

/**
 * @brief 修改电机地址
 * @param nCurAddr 电机当前地址
 * @param nNewAddr 修改后的新地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_changeAddr(int nCurAddr, int nNewAddr);

/**
 * @brief 改变DO状态
 * @param nAddr 电机地址
 * @param nDOValue DO配置
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_setDOState(int nAddr, int nDOValue);

/**
 * @brief 读取DI状态
 * @param nAddr 电机地址
 * @param *pDIState 指向存储DI状态的指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_readDIState(int nAddr, int* pDIState);

/**
 * @brief 读取DO状态
 * @param nAddr 电机地址
 * @param *pDOState 指向存储DO状态的指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_readDOState(int nAddr, int* pDOState);

/**
 * @brief 修改电机运行模式
 * @param nAddr 电机地址
 * @param nMode 运行模式
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_changeWorkMode(int nAddr, WORK_MODE nMode);

/**
 * @brief 获取电机状态字
 * @param nAddr 电机地址
 * @param pStatusWord 指针，返回状态字
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getCurrentStatus(int nAddr, int* pStatusWord);

/**
 * @brief 获取电机当前位置
 * @param nAddr 电机地址
 * @param pPosition 指针，返回当前位置
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_getCurrentPosition(int nAddr, int* pPosition);

/**
 * @brief 将电机当前位置保存为原点
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_saveAsHome(int  nAddr);

/**
 * @brief 将电机当前位置保存为零点
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_saveAsZero(int  nAddr);

/**
 * @brief 绝对位置运动
 * @param nAddr 电机地址
 * @param nPosition 目标位置
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_moveAbsolute(int nAddr, int nPosition);

/**
 * @brief 相对位置运动
 * @param nAddr 电机地址
 * @param nDistance 运动距离
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_moveRelative(int nAddr, int nDistance);

/**
 * @brief 速度模式运动
 * @param nAddr 电机地址
 * @param nVelocity 目标速度
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_moveVelocity(int nAddr, int nVelocity);

/**
 * @brief 原点回归
 * @param nAddr 电机地址
 * @param nType 原点回归方式
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_goHome(int nAddr, int nType);

/**
 * @brief 给电机驱动电路上电（抱机）
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_powerOn(int nAddr);

/**
 * @brief 给电机驱动电路断电（脱机）
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_powerOff(int nAddr);

/**
 * @brief 停止当前动作
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_stop(int nAddr);

/**
 * @brief 急停
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_fastStop(int nAddr);

/**
 * @brief SDK调试模式
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_setDebug(BOOL flag);

/*======================================================多串口函数组=====================================================================*/

/**************************错误码********************************/
/**
 * 0   执行成功
 * 1   不支持的设备类型
 * -1  执行失败
 * -2  电机地址选择错误
 * -3  参数传入错误
 * -4  当前电机运动模式错误
 * -5  电机未在使能状态
 * -6  串口被占用或有通信问题
 * -7  当前电机不支持的操作
*/

/***********************通信设备操作函数****************************/
/**
 * @brief 打开通信设备
 * @param nType 通信设备类型 (0:RTU 1:TCP，暂不支持)
 * @param strConnectString 连接字符串，描述设备连接参数()
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPopenDevice(int nType, const char* strConnectString);

/**
 * @brief 关闭通信设备
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPcloseDevice();

/***********************在线电机管理函数****************************/
/**
 * @brief 扫描电机
 * @param strPort 当前操作的端口号
 * @param nFromAddr 起始地址
 * @param nToAddr 结束地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPscanMotors(const char* strPort, int nFromAddr, int nToAddr);

/**
 * @brief 获取在线电机列表
 * @param strPort 当前操作的端口号
 * @param pAddrs 电机地址数组指针
 * @param pCount 数量指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetOnlineMotors(const char* strPort, int* pAddrs, int* pCount);

/**
 * @brief 判断电机是否在线
 * @param strPort 当前操作的端口号
 * @param pOnline 指针，返回在线状态
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPisMotorOnline(const char* strPort, int nAddr, BOOL* pOnline);

/**
 * @brief 获取电机基本信息
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pInfo 电机信息结构体指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetMotorInfo(const char* strPort, int nAddr, MOTOR_INFO* pInfo);

/**
 * @brief 执行电机自检
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pResult 自检结果结构体指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPselfcheck(const char* strPort, int nAddr, SELFCHECK_RESULT* pResult);

/**
 * @brief 获取电机最近的报警
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pAlarmCode 指针，返回报警值
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetLatestAlarm(const char* strPort, int nAddr, int * pAlarmCode);

/**
 * @brief 获取电机故障码
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pErrorCode 指针，返回故障码
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetErrorCode(const char* strPort, int nAddr, int * pErrorCode);

/**
 * @brief 获取电机历史报警
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pAlarmCode 数组指针，返回报警值列表
 * @param pCount 指针，返回报警数量
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetHistoryAlarms(const char* strPort, int nAddr, int * pAlarmCode, int * pCount);

/**
 * @brief 清除电机报警
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPclearAlarms(const char* strPort, int nAddr);

/**
 * @brief 清除电机故障
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPclearErrorState(const char* strPort, int nAddr);

/**
 * @brief 重启电机
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPrebootMotor(const char* strPort, int nAddr);

/**********************电机控制函数******************************/

/**
 * @brief 获取电机参数值
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nParamID 参数ID
 * @param nBytes 字节数
 * @param pParamValue 指针，返回参数值
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPreadParam(const char* strPort, int nAddr, int nParamID, int nBytes, int* pParamValue);

/**
 * @brief 设置电机参数值
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nParamID 参数ID
 * @param nBytes 字节数
 * @param nParamValue 参数值
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPwriteParam(const char* strPort, int nAddr, int nParamID, int nBytes, int nParamValue);

/**
 * @brief 保存电机参数
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPsaveParams(const char* strPort, int nAddr);

/**
 * @brief 恢复电机出厂设置
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPrestoreFactorySettings(const char* strPort, int nAddr);

/**
 * @brief 修改电机地址
 * @param strPort 当前操作的端口号
 * @param nCurAddr 电机当前地址
 * @param nNewAddr 修改后的新地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPchangeAddr(const char* strPort, int nCurAddr, int nNewAddr);

/**
 * @brief 改变DO状态
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nDOValue DO配置
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPsetDOState(const char* strPort, int nAddr, int nDOValue);

/**
 * @brief 读取DI状态
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param *pDIState 指向存储DI状态的指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPreadDIState(const char* strPort, int nAddr, int* pDIState);

/**
 * @brief 读取DO状态
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param *pDOState 指向存储DO状态的指针
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPreadDOState(const char* strPort, int nAddr, int* pDOState);

/**
 * @brief 修改电机运行模式
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nMode 运行模式
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPchangeWorkMode(const char* strPort, int nAddr, WORK_MODE nMode);

/**
 * @brief 获取电机状态字
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pStatusWord 指针，返回状态字
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetCurrentStatus(const char* strPort, int nAddr, int* pStatusWord);

/**
 * @brief 获取电机当前位置
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param pPosition 指针，返回当前位置
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgetCurrentPosition(const char* strPort, int nAddr, int* pPosition);

/**
 * @brief 将电机当前位置保存为原点
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPsaveAsHome(const char* strPort, int  nAddr);

/**
 * @brief 将电机当前位置保存为零点
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPsaveAsZero(const char* strPort, int  nAddr);

/**
 * @brief 绝对位置运动
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nPosition 目标位置
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPmoveAbsolute(const char* strPort, int nAddr, int nPosition);

/**
 * @brief 相对位置运动
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nDistance 运动距离
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPmoveRelative(const char* strPort, int nAddr, int nDistance);

/**
 * @brief 速度模式运动
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nVelocity 目标速度
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPmoveVelocity(const char* strPort, int nAddr, int nVelocity);

/**
 * @brief 原点回归
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @param nType 原点回归方式
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPgoHome(const char* strPort, int nAddr, int nType);

/**
 * @brief 给电机驱动电路上电（抱机）
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPpowerOn(const char* strPort, int nAddr);

/**
 * @brief 给电机驱动电路断电（脱机）
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPpowerOff(const char* strPort, int nAddr);

/**
 * @brief 停止当前动作
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPstop(const char* strPort, int nAddr);

/**
 * @brief 急停
 * @param strPort 当前操作的端口号
 * @param nAddr 电机地址
 * @return 0 成功，其它表示错误码
 */
NiMotionMotorAPI
int NIMOTION_CALL NiM_MPfastStop(const char* strPort, int nAddr);


#ifdef __cplusplus
}
#endif

#endif  //__NIMOTION_MOTOR_SDK_H__
