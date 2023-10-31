#!/usr/bin/env python

# 一个自动化执行多序列的脚本罢了！

import os
import rospy
import subprocess
import datetime
from std_msgs.msg import String

# global roslaunch_process, current_sequence_index,current_weighttype_index,current_curvedfilter_index
current_datetime = datetime.datetime.now()
file_name = "result_output/output_" + str(current_datetime) + ".txt"

showrviz = "true"
# ======================================SEQUENCE==================================================
sequence_list = []
sequence_list += [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
]  # 完整序列
# sequence_list += ["00","02",08"] # 长序列
# sequence_list += ["05","09"] # 中序列
# sequence_list += ["01","03"] # 短序列 1
# sequence_list += ["06","02","04","09"] # 短序列 2
# sequence_list += ["04","10"] # 短序列 3
# sequence_list += ["03","04"]  #序列 - 测试
current_sequence_index = 0


# ======================================WEIGHT==================================================
# weighttype_list = ["0", "1", "2","12"]  # 完整序列
weighttype_list = [ "0","1","2"]  # 权重 - 测试
current_weighttype_index = 0

# ======================================CURVED==================================================
# curvedfilter_list = [ "false","true"]  # 完整序列
curvedfilter_list = ["1"]  # 弯曲体素 - 测试
current_curvedfilter_index = 0

# ======================================Ground==================================================
# groundfilter_list = [ "false","true"]  # 完整序列
groundfilter_list = ["1"]  # 弯曲体素 - 测试
current_groundfilter_index = 0

# ======================================Feature==================================================
# featurefilter_list = [ "false","true"]  # 完整序列
featurefilter_list = ["1"]  # 弯曲体素 - 测试
current_featurefilter_index = 0

# 全局变量，保存roslaunch子进程的引用
roslaunch_process = None


def fileHead():
    with open(file_name, "w") as file:
        # 写入内容
        file.write("======================================\n")
        file.write(str(current_datetime) + "\n")

        variable_strings = []
        variables = [sequence_list, weighttype_list, curvedfilter_list]
        variable_names = ["sequence_list", "weighttype_list", "curvedfilter_list"]

        for name, lst in zip(variable_names, variables):
            variable_string = f"{name} = {repr(lst)}"
            variable_strings.append(variable_string)

        # 打印或使用包含变量名和内容的字符串的列表
        for variable_string in variable_strings:
            print(variable_string)
            file.write(variable_string)
            file.write("\n")


# 启动roslaunch子进程
def start_roslaunch():
    rospy.loginfo("启动launch文件")
    global current_sequence_index, current_weighttype_index, current_curvedfilter_index, current_groundfilter_index,current_featurefilter_index,roslaunch_process
    # 获取当前要传递的sequence值
    sequence_value = sequence_list[current_sequence_index]
    weighttype_value = weighttype_list[current_weighttype_index]
    curvedfilter_value = curvedfilter_list[current_curvedfilter_index]
    groundfilter_value = groundfilter_list[current_groundfilter_index]
    featurePreExtract_value = featurefilter_list[current_featurefilter_index]

    # 启动roslaunch子进程并传递参数
    roslaunch_command = [
        "roslaunch",
        "pfilter",
        "pfilter_kitti.launch",
        "sequence_py:=" + sequence_value,
        "weighttype:=" + weighttype_value,
        "curvedfilter:=" + curvedfilter_value,
        "groundfilter:=" + groundfilter_value,
        "featurePreExtract:=" + featurePreExtract_value,
        "showrviz:=" + showrviz,
    ]
    roslaunch_process = subprocess.Popen(roslaunch_command)


def kitti_eva():
    kitti_eva_root = "/home/r/kitti_eva/KITTI_odometry_evaluation_tool"
    seq_list = ""
    for i, seq_ind in enumerate(sequence_list):
        if i == len(sequence_list) - 1:
            seq_list += str(seq_ind) + "_pred"
        else:
            seq_list += str(seq_ind) + "_pred" + ","
    # print(seq_list)
    eva_command = (
        "python3 "
        + kitti_eva_root
        + "/evaluation.py "
        + "\t"
        + "--result_dir="
        + kitti_eva_root
        + "/data/"
        + "\t"
        "--eva_seqs="
        + seq_list
        + "\t"
        + " --gt_dir="
        + kitti_eva_root
        + "/ground_truth_pose/"
    )
    # print(eva_command)
    # os.system(eva_command)
    try:
        process = subprocess.Popen(
            eva_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # 等待命令执行完毕
        output, _ = process.communicate()
        print(output)
        # 将命令的输出写入文件
        with open(file_name, "a") as file:
            file.write(output)
    except subprocess.CalledProcessError as e:
        # 如果命令返回非零退出状态，则可以在此处处理错误
        print("命令执行错误:", e)
    print("ok")
    return


# 回调函数，处理接收到的消息
def message_callback(msg):
    global current_sequence_index, current_weighttype_index, current_curvedfilter_index, roslaunch_process
    # 在接收到消息后执行需要的操作
    rospy.loginfo("接收到消息：%s，关闭所有节点" % msg.data)

    # 如果已经启动了roslaunch子进程，则关闭它
    if roslaunch_process is not None:
        rospy.loginfo("关闭roslaunch子进程")
        roslaunch_process.terminate()
        roslaunch_process.wait()
        rospy.loginfo("roslaunch子进程已关闭")

        # 更新sequence索引，如果已到达列表末尾，则终止程序
        current_sequence_index = current_sequence_index + 1
        if current_sequence_index == len(sequence_list):
            rospy.loginfo("——————————————————————————————————————————————————————")
            rospy.loginfo("————————————————————— 启动评估程序 —————————————————————")
            rospy.loginfo("——————————————————————————————————————————————————————")

            info_head = (
                "========================================================\n",
                "当前组合：",
                "weighttype\t",
                weighttype_list[current_weighttype_index],
                "curvedfilter\t",
                curvedfilter_list[current_curvedfilter_index],
            )
            info_head_string = " ".join([str(x) for x in info_head])
            print(info_head_string)
            with open(file_name, "a") as file:
                file.write(info_head_string)
            kitti_eva()
            current_weighttype_index = current_weighttype_index + 1
            if current_weighttype_index == len(weighttype_list):
                print("current_weighttype_index == len(weighttype_list)")
                current_curvedfilter_index = current_curvedfilter_index + 1
                if current_curvedfilter_index == len(curvedfilter_list):
                    print("current_curvedfilter_index == len(curvedfilter_list):")
                    rospy.loginfo("已达到列表末尾, 终止PFilter程序")
                    rospy.signal_shutdown("Sequence list finished.")
                else:
                    print("current_curvedfilter_index != len(curvedfilter_list):")
                    current_weighttype_index = 0
                    current_sequence_index = 0
            else:
                print("current_weighttype_index != len(weighttype_list)")
                current_sequence_index = 0

        rospy.loginfo("等待1秒")
        rospy.sleep(1)

        # 重新启动roslaunch子进程
        start_roslaunch()


if __name__ == "__main__":
    try:
        fileHead()
        # 初始化ROS节点
        rospy.init_node("launch_and_shutdown_node")

        # 启动初始的roslaunch子进程
        start_roslaunch()

        # 订阅话题，指定回调函数
        rospy.Subscriber("shutdown", String, message_callback)

        # 等待关闭命令
        rospy.spin()
        # kitti_eva()
    except rospy.ROSInterruptException:
        pass
