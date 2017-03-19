#!/usr/bin/env python
# Filename: test_para 
"""
introduction: Test parameters of network in starter.py

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 March, 2017
"""
import time,os,subprocess

logfile = 'test_DL_network.txt'
def outputlogMessage(message):
    """
    output format log message
    Args:
        message: the message string need to be output

    Returns:None

    """
    global logfile
    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
    outstr = timestr +': '+ message
    print(outstr)
    f=open(logfile,'a')
    f.writelines(outstr+'\n')
    f.close()

def output_commandString_from_args_list(args_list):
    commands_str = ''
    if isinstance(args_list,list) and len(args_list)>0:
        for args_str in args_list:
            if ' ' in  args_str:
                commands_str += '\"'+args_str+'\"' + ' ';
            else:
                commands_str += args_str + ' ';
    return commands_str

def exec_command_args_list(args_list):
    """
    execute a command string
    Args:
        args_list: a list contains args

    Returns:

    """
    outputlogMessage(output_commandString_from_args_list(args_list))
    ps = subprocess.Popen(args_list)
    returncode = ps.wait()
    outputlogMessage('return codes: '+ str(returncode))

def main():



    pass


if __name__=='__main__':
    main()
