#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <string>
#include "NiMotionMotorSDK.h"

const int DEGREE_TO_PULSE = 400;
void waitForMotionFinished(int nAddr);
void seekAndSetZero(int nAddr);


class RotationStage
{
private:
    int nAddr;
    int rc;
    double position;
    double velocity;
public:
    RotationStage(const char* serial_port, int _nAddr)
    {
        nAddr = _nAddr;
        std::string prefix = "{\"DeviceName\": \"";
        std::string suffix = "\", \"Baudrate\" : 115200, \"Parity\" : \"None\", \"DataBits\" : 8, \"StopBits\" : 1}";//char数组自动转为string
        std::string port_name = serial_port;
        std::string full_arg = prefix + port_name + suffix;
        rc = NiM_openDevice(0, full_arg.c_str());//string转const char*
        
        rc = NiM_powerOn(nAddr);

    }
    ~RotationStage()
    {
        rc = NiM_powerOff(nAddr);
        usleep(100000);
        NiM_closeDevice();
    }
    
    void seek_and_set_zero()
    {
        int nDIState = 0;
        rc = NiM_readDIState(nAddr, &nDIState);
        printf("Initial Motor DIState is %d\r\n", nDIState);
        int inside_zero_point = 0;

        if (nDIState == 0)
        {	
            rc = NiM_changeWorkMode(nAddr, POSITION_MODE);

            rc = NiM_writeParam(nAddr, 0x5B,4,1000);
            rc = NiM_writeParam(nAddr, 0x5F,4,2000);
            rc = NiM_powerOn(nAddr);
            rc = NiM_moveRelative(nAddr, 360 * DEGREE_TO_PULSE);	//位置模式正转360
            while (nDIState == 0)
            {
                rc = NiM_readDIState(nAddr, &nDIState);
                usleep(50000);
            }
            rc = NiM_stop(nAddr);
            wait_for_motion_finished();
        }
        
        rc = NiM_getCurrentPosition(nAddr, &inside_zero_point);
        printf("Inside Zero Point Region, pos = %lf\r\n", double(inside_zero_point) / DEGREE_TO_PULSE);
        
        // Now the switch should be on
        rc = NiM_changeWorkMode(nAddr, POSITION_MODE);

        rc = NiM_writeParam(nAddr, 0x5B,4,50);
        rc = NiM_writeParam(nAddr, 0x5F,4,1000);    
        
        rc = NiM_moveRelative(nAddr, 10 * DEGREE_TO_PULSE);	//位置模式正转10

        while (nDIState != 0)
        {
            rc = NiM_readDIState(nAddr, &nDIState);
            usleep(50000);
        }
        int zero_point_p = 0;
        rc = NiM_getCurrentPosition(nAddr, &zero_point_p);
        rc = NiM_stop(nAddr);
        wait_for_motion_finished();
        
        //Now at the positive side of zero point

        printf("Zero Point Positive Side, pos = %lf\r\n", double(zero_point_p) / DEGREE_TO_PULSE);
        
        rc = NiM_moveAbsolute(nAddr, inside_zero_point);
        wait_for_motion_finished();
        rc = NiM_readDIState(nAddr, &nDIState);
        printf("Returned, Motor DIState is %d\r\n", nDIState);
        
        
        rc = NiM_moveRelative(nAddr, -10 * DEGREE_TO_PULSE);

        while (nDIState != 0)
        {
            rc = NiM_readDIState(nAddr, &nDIState);
            usleep(50000);
        }
        int zero_point_n = 0;
        rc = NiM_getCurrentPosition(nAddr, &zero_point_n);
        rc = NiM_stop(nAddr);
        wait_for_motion_finished();
        
        //Now at the negative side of zero point

        printf("Zero Point Negative Side, pos = %lf\r\n", double(zero_point_n) / DEGREE_TO_PULSE);
        
        int zero_point = (zero_point_n + zero_point_p) / 2;
        
        rc = NiM_moveAbsolute(nAddr, zero_point);
        wait_for_motion_finished();
        printf("Zero Point is %lf\r\n", double(zero_point) / DEGREE_TO_PULSE);
        
        rc = NiM_powerOff(nAddr);
            /*保存当前位置为原点*/
        rc = NiM_saveAsHome(nAddr);

        /*保存当前位置为零点*/
        rc = NiM_saveAsZero(nAddr);
        
        usleep(500000);
        
        rc = NiM_writeParam(nAddr, 0x5B, 4, 1000);
        rc = NiM_writeParam(nAddr, 0x5F, 4, 2000);
    }
    
    
    int rel_move(double pos, bool wait=false)
    {
        rc = NiM_moveRelative(nAddr, int(pos * DEGREE_TO_PULSE));
        if (wait)
        {
            wait_for_motion_finished();
        }
        return rc;
    }
    
    int abs_move(double pos, bool wait=false)
    {
        rc = NiM_moveAbsolute(nAddr, int(pos * DEGREE_TO_PULSE));
        if (wait)
        {
            wait_for_motion_finished();
        }
        return rc;
    }
    
    int stop()
    {
        rc = NiM_stop(nAddr);
        wait_for_motion_finished();
        return rc;
    }
    
    double get_position()
    {
        int pos;
        rc = NiM_getCurrentPosition(nAddr, &pos);
        position = double(pos) / DEGREE_TO_PULSE;
        return position;
    }
    
    double get_velocity()
    {
        clock_t start, end;
        int pos;
        rc = NiM_getCurrentPosition(nAddr, &pos);
        start = clock();
        position = double(pos) / DEGREE_TO_PULSE;
        
        usleep(100000);
        rc = NiM_getCurrentPosition(nAddr, &pos);
        end = clock();
        double position2 = double(pos) / DEGREE_TO_PULSE;
        
        velocity = (position2 - position) * 1000 / (end - start);
        position = position2;
        return velocity;
    }

    bool is_moving()
    {
        if (abs(get_velocity()) >= 1e-3)
            return true;
        else 
            return false;
    }
    
    void wait_for_motion_finished()
    {
        int pos = 0, last_position = 0;
        int nDIState = 0;
        int zero_count = 0;
        while (1) 
        {
            usleep(50000);
            rc = NiM_getCurrentPosition(nAddr, &pos);
            position = double(pos) / DEGREE_TO_PULSE;
            // printf("pos = %d\r\n", pos);
            int diff = pos - last_position;
            last_position = pos;
            if (diff == 0)
            {
                zero_count ++;
                if (zero_count > 3) 
                    break;
            }
            // rc = NiM_readDIState(nAddr, &nDIState);
            // printf("The Motor DIState is %d\r\n", nDIState);
        }
    }
};


