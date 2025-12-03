# -*- coding: utf-8 -*-
import tensorrt as trt
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import socket
import time
import threading
import os
import sys
import platform
import logging
import logging.handlers
import json
import traceback
import signal
import select
import psutil
import queue
import struct
from pymavlink import mavutil

print(" Ciallo～(∠・ω< )⌒★ ")   # 程序运行检测提示文字

# --------------------------
# 增强的日志配置
# --------------------------
def setup_logging():
    """设置增强的日志系统 - 支持日志切割和彩色输出"""
    # 创建logs目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 清理过大的日志文件 - 只保留最近10个
    try:
        log_files = sorted([f for f in os.listdir(log_dir) if f.startswith('detection_') and f.endswith('.log')])
        max_log_files = 10
        if len(log_files) > max_log_files:
            for old_log in log_files[:-max_log_files]:
                os.remove(os.path.join(log_dir, old_log))
                print(f"🧹 清理旧日志文件: {old_log}")
    except Exception as e:
        print(f"⚠ 日志清理失败: {e}")
    
    # 按日期生成日志文件名
    current_time = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"detection_{current_time}.log")
    
    # 配置日志（改用RotatingFileHandler切割日志，避免磁盘占满）
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 清除默认handler，避免重复输出
    
    # 控制台Handler（输出INFO及以上级别）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 文件Handler（按大小切割，每个100MB，保留5个备份）
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 统一日志格式（包含行号，方便定位问题）
    formatter = logging.Formatter(
        '%(asctime)s - %(module)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 设置TensorRT日志级别
    trt_logger = trt.Logger(trt.Logger.WARNING)
    
    print(f"📝 日志文件: {log_file}")
    return logger, log_file

# 临时logger，在main函数中会被重新初始化
logger = logging.getLogger(__name__)

# --------------------------
# 混合日志系统
# --------------------------
class HybridLogger:
    """混合日志系统 - 关键信息立即print，详细信息记录到文件"""
    
    def __init__(self):
        # 使用全局logger（在main函数中会被重新初始化）
        self.logger = logger
    
    def critical(self, msg):
        """关键错误 - 立即print并记录到日志"""
        print(f"🔥🔥🔥 {msg} 🔥🔥🔥")
        self.logger.critical(msg)
    
    def error(self, msg, immediate=True):
        """错误信息 - 立即print并记录到日志"""
        if immediate:
            print(f"❌ {msg}")
        self.logger.error(msg)
    
    def warning(self, msg, immediate=True):
        """警告信息 - 立即print"""
        if immediate:
            print(f"⚠ {msg}")
        self.logger.warning(msg)
    
    def info(self, msg, immediate=False):
        """普通信息 - 主要记录到日志，重要信息可print"""
        if immediate:
            print(f"ℹ️ {msg}")
        self.logger.info(msg)
    
    def debug(self, msg):
        """调试信息 - 只记录到日志"""
        self.logger.debug(msg)
    
    def exception(self, msg, immediate=True):
        """异常信息 - 带完整堆栈跟踪"""
        if immediate:
            print(f"💥 {msg}")
            traceback.print_exc()  # 立即显示堆栈
        self.logger.exception(msg)  # 记录到文件

# 创建混合日志实例（在main函数中会重新初始化）
log = HybridLogger()

# --------------------------
# 配置管理
# --------------------------
class Config:
    """配置管理器 - 支持从文件加载配置，新人注意：这里可以统一修改所有参数"""
    def __init__(self):
        self.config_file = "config.json"  # 配置文件路径
        self.default_config = {
            # 🔧 串口配置 - Eport串口链接
            "serial_port": "/dev/ttyUSB0",        # Eport串口设备路径（边缘盒执行dmesg | grep tty获取）
            "serial_baudrate": 921600,            # Eport默认波特率（大疆标准）
            
            # 🔧 流媒体配置 - 视频流处理
            "rtsp_url": "rtsp://192.168.1.20:554/main.264",  # 大疆相机RTSP视频流地址
            
            # 🔧 模型配置 - AI模型相关设置
            "model_path": "best.engine",          # TensorRT引擎文件路径（需ARMv8适配版）
            "classes_file": "classes.txt",        # 类别标签文件路径，定义检测目标
            
            # 🔧 图像处理配置 - 影响检测精度和性能
            "img_size_arm": 640,                  # ARM平台图像尺寸，性能弱用小尺寸
            "img_size_x86": 1080,                 # x86平台图像尺寸，性能强用大尺寸
            "conf_thres_arm": 0.25,               # ARM平台置信度阈值，值越大要求越严格
            "conf_thres_x86": 0.15,               # x86平台置信度阈值，值越小检测越多
            "iou_thres": 0.3,                     # NMS的IOU阈值，值越大重叠框越少
            
            # 🔧 系统性能配置 - 调整运行参数
            "max_targets_per_frame": 5,           # 单帧最大发送目标数，避免数据过载
            "target_fps": 30,                     # 目标帧率，影响CPU/GPU占用
            
            # 🔧 连接和重连配置 - 串口稳定性
            "heartbeat_timeout": 5,               # Mavlink心跳包超时时间（秒）
            "reconnect_delay": 2,                 # 初始重连延迟（秒）
            "max_reconnect_delay": 30,            # 最大重连延迟（秒），指数退避上限
            "connection_timeout": 10,             # 串口连接超时时间（秒）
            
            # 🔧 流媒体配置 - 视频流处理
            "stream_buffer_size": 3,              # RTSP流缓冲区大小，影响流畅度
            
            # 🔧 监控和调试配置 - 系统状态显示
            "health_report_interval": 30,         # 健康状态报告间隔（秒）
            "enable_frame_stats": True,           # 是否启用帧统计信息
            "enable_detection_display": True,     # 是否显示检测结果窗口
            "enable_network_stats": True,         # 是否显示网络统计信息
            
            # 🔧 新增：图像增强配置
            "enable_image_enhancement": True,     # 是否启用图像增强
            "clahe_clip_limit": 2.0,              # CLAHE对比度限制
            "brightness_alpha": 1.2,              # 亮度增强系数
            "brightness_beta": 10,                # 亮度偏移量
            
            # 🔧 新增：性能优化配置
            "warmup_frames": 10,                  # 预热帧数，避免启动时性能波动
            "adaptive_fps": True,                 # 是否启用自适应帧率
            "min_fps": 10,                        # 最低帧率
            "max_fps": 60,                        # 最高帧率
            
            # 🔧 协议可配置化
            "mavlink_command": "MAV_CMD_USER_1",  # Mavlink命令类型，默认USER_1
            "gpu_monitoring": True,               # 是否启用GPU监控（如果可用）
            
            # 🔧 新增：性能阈值配置
            "high_cpu_threshold": 80.0,           # CPU使用率告警阈值（%）
            "high_memory_threshold": 85.0,        # 内存使用率告警阈值（%）
            "slow_inference_threshold": 100.0,    # 推理耗时告警阈值（ms）
            
            # 🔧 新增：跟踪控制配置
            "tracking_enabled": True,             # 是否启用目标跟踪
            "tracking_mode": "gimbal",            # 跟踪模式：'gimbal'(云台) 或 'flight'(飞机)
            "gimbal_tracking": True,              # 是否启用云台跟踪
            "flight_tracking": False,             # 是否启用飞机跟踪
            "gimbal_KP": 0.5,                     # 云台PID比例系数
            "gimbal_KI": 0.0,                     # 云台PID积分系数
            "gimbal_KD": 0.0,                     # 云台PID微分系数
            "flight_KP": 0.8,                     # 飞机PID比例系数
            "flight_KI": 0.0,                     # 飞机PID积分系数
            "flight_KD": 0.0,                     # 飞机PID微分系数
            "dead_zone": 0.05,                    # 死区范围，避免抖动
            "max_gimbal_speed": 30.0,             # 最大云台角速度（度/秒）
            "max_flight_speed": 2.0,              # 最大飞机速度（米/秒）
            "min_tracking_distance": 5.0,         # 最小跟踪距离（米）
            "max_tracking_distance": 50.0,        # 最大跟踪距离（米）
            
            # 🔧 新增：安全配置
            "safe_tracking_altitude": 10.0,       # 安全跟踪高度（米）
            "emergency_stop_distance": 2.0,       # 紧急停止距离（米）
            
            # 🔧 新增：图像预处理配置
            "image_resize_method": "letterbox",   # 图像缩放方法: 'letterbox' 或 'stretch'
            "normalize_mean": [0.485, 0.456, 0.406],  # 归一化均值
            "normalize_std": [0.229, 0.224, 0.225],   # 归一化标准差
            
            # 🔧 新增：后处理配置
            "nms_per_class": True,                # 是否按类别进行NMS
            "merge_overlapping_boxes": False,     # 是否合并重叠框
            "filter_small_boxes": True,           # 是否过滤小框
            "min_box_area_ratio": 0.001,          # 最小框面积比例（相对于图像面积）
            
            # 🔧 新增：调试配置
            "debug_mode": False,                  # 调试模式
            "save_detection_images": False,       # 是否保存检测图像
            "save_detection_path": "detections",  # 检测图像保存路径
        }
        self.config = self._load_config()  # 加载配置，优先使用用户配置
    
    def _load_config(self):
        """从文件加载配置，如果文件不存在则使用默认配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置，用户配置覆盖默认配置
                config = {**self.default_config, **user_config}
                print(f"✅ 从 {self.config_file} 加载配置")
                return config
            except Exception as e:
                log.warning(f"读取配置文件失败: {e}，使用默认配置")
        
        print("ℹ️ 使用默认配置")
        return self.default_config.copy()
    
    def get(self, key, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def save(self):
        """保存当前配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"✅ 配置已保存到 {self.config_file}")
        except Exception as e:
            log.error(f"保存配置失败: {e}")
    
    def validate(self):
        """验证关键配置 - 确保必要参数正确"""
        print("🔍 验证配置参数...")
        required_keys = ['serial_port', 'serial_baudrate', 'rtsp_url', 'model_path']
        for key in required_keys:
            if not self.get(key):
                log.error(f"关键配置缺失: {key}", immediate=True)
                return False
        
        # 验证文件存在性
        if not os.path.exists(self.get("model_path")):
            log.error(f"模型文件不存在: {self.get('model_path')}", immediate=True)
            return False
            
        # 验证数值范围合理性
        if not 0 < self.get("conf_thres_arm", 0.25) <= 1:
            log.error("ARM平台置信度阈值必须在0-1之间", immediate=True)
            return False
            
        if not 0 < self.get("conf_thres_x86", 0.15) <= 1:
            log.error("x86平台置信度阈值必须在0-1之间", immediate=True)
            return False
            
        print("✅ 配置验证通过")
        return True
    
    # 🔧 减少硬编码耦合
    def get_image_enhancement_params(self):
        """获取图像增强参数 - 统一管理，避免硬编码"""
        return {
            'enabled': self.get("enable_image_enhancement", True),
            'clahe_clip_limit': self.get("clahe_clip_limit", 2.0),
            'brightness_alpha': self.get("brightness_alpha", 1.2),
            'brightness_beta': self.get("brightness_beta", 10)
        }
    
    def get_performance_thresholds(self):
        """获取性能阈值配置 - 统一管理"""
        return {
            'high_cpu': self.get("high_cpu_threshold", 80.0),
            'high_memory': self.get("high_memory_threshold", 85.0),
            'slow_inference': self.get("slow_inference_threshold", 100.0)
        }
    
    def get_tracking_params(self):
        """获取跟踪参数配置"""
        return {
            'gimbal_KP': self.get("gimbal_KP", 0.5),
            'gimbal_KI': self.get("gimbal_KI", 0.0),
            'gimbal_KD': self.get("gimbal_KD", 0.0),
            'flight_KP': self.get("flight_KP", 0.8),
            'flight_KI': self.get("flight_KI", 0.0),
            'flight_KD': self.get("flight_KD", 0.0),
            'dead_zone': self.get("dead_zone", 0.05),
            'max_gimbal_speed': self.get("max_gimbal_speed", 30.0),
            'max_flight_speed': self.get("max_flight_speed", 2.0),
            'min_tracking_distance': self.get("min_tracking_distance", 5.0),
            'max_tracking_distance': self.get("max_tracking_distance", 50.0),
            'safe_tracking_altitude': self.get("safe_tracking_altitude", 10.0),
            'emergency_stop_distance': self.get("emergency_stop_distance", 2.0),
        }
    
    def get_mavlink_command(self):
        """获取Mavlink命令配置 - 支持协议可配置化"""
        command_name = self.get("mavlink_command", "MAV_CMD_USER_1")
        # 将字符串命令名映射到实际的Mavlink命令值
        command_map = {
            "MAV_CMD_USER_1": mavutil.mavlink.MAV_CMD_USER_1,
            "MAV_CMD_USER_2": mavutil.mavlink.MAV_CMD_USER_2,
            "MAV_CMD_USER_3": mavutil.mavlink.MAV_CMD_USER_3,
            "MAV_CMD_USER_4": mavutil.mavlink.MAV_CMD_USER_4,
            "MAV_CMD_USER_5": mavutil.mavlink.MAV_CMD_USER_5,
        }
        return command_map.get(command_name, mavutil.mavlink.MAV_CMD_USER_1)

# 全局配置实例
config = Config()

# --------------------------
# 信号处理 - 优雅退出
# --------------------------
def signal_handler(signum, frame):
    """处理系统信号，实现优雅退出"""
    signal_name = "SIGINT(Ctrl+C)" if signum == signal.SIGINT else f"信号{signum}"
    print(f"\n🎯 收到 {signal_name}，正在优雅退出...")
    raise KeyboardInterrupt(f"信号终止: {signal_name}")

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)  # 系统终止信号

# --------------------------
# 全局配置
# --------------------------
class PlatformConfig:
    """平台配置检测 - 自动适配不同硬件，新人注意：这里会根据你的电脑自动调整参数"""
    def __init__(self):
        # 确保配置已加载
        if not hasattr(config, 'config') or not config.config:
            config.config = config._load_config()
            
        # 自动检测系统信息，不用改这里
        self.system = platform.system().lower()  # 获取操作系统：windows/linux
        self.machine = platform.machine().lower()  # 获取CPU架构：x86/arm
        self.is_arm = 'arm' in self.machine or 'aarch' in self.machine  # 判断是不是ARM平台
        self.is_linux = self.system == 'linux'  # 判断是不是Linux系统
        self.is_windows = self.system == 'windows'  # 判断是不是Windows系统
        
        # 检测GUI支持 - 自动判断是否能显示窗口
        self.has_gui = self.system != 'linux' or 'DISPLAY' in os.environ
        
        # 关键启动信息用print立即显示
        print(f"🎯 检测到平台: {self.system.upper()} {self.machine}")
        print(f"🎯 ARM架构: {self.is_arm}")
        print(f"🎯 GUI支持: {self.has_gui}")
        
        # 详细平台信息记录到日志
        log.info(f"平台详细检测: system={self.system}, machine={self.machine}, is_arm={self.is_arm}")
        
        # 根据平台调整参数 - 新人注意：这里可以修改性能参数
        if self.is_arm:
            # ARM平台（比如树莓派、边缘计算盒子、Jetson）性能较弱，用低分辨率
            self.img_size = config.get("img_size_arm", 640)  # 🔧 可以改：ARM平台图像尺寸，640够用且速度快
            self.conf_thres = config.get("conf_thres_arm", 0.25)  # 🔧 可以改：ARM上误检多，置信度阈值调高
            print("🔄 ARM平台: 使用640x640分辨率，置信度阈值0.25")
        else:
            # x86平台（普通电脑）性能强，可以用高分辨率
            self.img_size = config.get("img_size_x86", 1080)  # 🔧 可以改：x86平台图像尺寸，1080更清晰但耗资源
            self.conf_thres = config.get("conf_thres_x86", 0.15)  # 🔧 可以改：x86平台误检少，置信度可以调低

# 初始化平台配置 - 自动适配，新人不用改这里
platform_cfg = PlatformConfig()

# 🔧 串口配置 - 新人注意：这里根据边缘盒实际串口修改
serial_port = config.get("serial_port", "/dev/ttyUSB0")    # 🔧 必改：Eport串口路径（dmesg | grep tty获取）
serial_baudrate = config.get("serial_baudrate", 921600)    # 🔧 通常不用改：Eport默认波特率

# 🔧 流媒体配置 - 新人注意：修改为实际RTSP地址
RTSP_STREAM_URL = config.get("rtsp_url", "rtsp://192.168.1.20:554/main.264")  # 🔧 必改：大疆相机RTSP流地址

# 🔧 性能参数 - 新人注意：这里可以调整检测灵敏度
IMG_SIZE = platform_cfg.img_size  # 图像尺寸，自动根据平台设置
CONF_THRES = platform_cfg.conf_thres  # 🔧 可以改：置信度阈值，值越大要求越严格
IOU_THRES = config.get("iou_thres", 0.3)  # 🔧 可以改：NMS的IOU阈值，值越大框越少

# YOLO模型常量 - 新人注意：这些是模型相关参数，一般不用改
YOLO_OUTPUT_DIM = 85  # YOLOv5输出维度：4坐标 + 1置信度 + 80类别
MAX_TARGETS_PER_FRAME = config.get("max_targets_per_frame", 5)  # 🔧 可以改：单帧最大发送目标数

# 类别配置
def load_classes(class_file=None):
    """从文件加载类别，新人注意：修改classes.txt文件来添加你的检测目标"""
    if class_file is None:
        class_file = config.get("classes_file", "classes.txt")
        
    if os.path.exists(class_file):
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            
            if not classes:  # 检查空文件
                log.warning(f"类别文件 {class_file} 为空，使用默认类别", immediate=True)
                default_classes = ["drone"]
                return default_classes
                
            print(f"✅ 从 {class_file} 加载 {len(classes)} 个类别")
            return classes
        except Exception as e:
            log.error(f"读取类别文件失败: {e}，使用默认类别", immediate=True)
    
    # 默认类别 - 🔧 新人注意：如果没找到classes.txt，就用这个默认类别
    default_classes = ["drone"]  # 🔧 可以改：添加你的检测目标，比如["person", "car", "dog"]
    print(f"ℹ️ 使用默认类别: {default_classes}")
    return default_classes

CLASSES = load_classes()  # 加载类别列表

# --------------------------
# PID控制器
# --------------------------
class PIDController:
    """PID控制器 - 用于跟踪控制"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def update(self, error):
        """更新PID控制器"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.001  # 防止除零错误
            
        # 积分项
        self.integral += error * dt
        
        # 微分项
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # 计算输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 保存当前值供下次使用
        self.last_error = error
        self.last_time = current_time
        
        return output

# --------------------------
# 跟踪管理器
# --------------------------
class TrackingManager:
    """跟踪管理器 - 管理目标跟踪逻辑"""
    def __init__(self):
        self.enabled = config.get("tracking_enabled", True)
        self.gimbal_tracking = config.get("gimbal_tracking", True)
        self.flight_tracking = config.get("flight_tracking", False)
        self.tracking_mode = config.get("tracking_mode", "gimbal")
        
        # 获取PID参数
        tracking_params = config.get_tracking_params()
        self.gimbal_pid_x = PIDController(
            kp=tracking_params['gimbal_KP'],
            ki=tracking_params['gimbal_KI'],
            kd=tracking_params['gimbal_KD']
        )
        self.gimbal_pid_y = PIDController(
            kp=tracking_params['gimbal_KP'],
            ki=tracking_params['gimbal_KI'],
            kd=tracking_params['gimbal_KD']
        )
        self.flight_pid_x = PIDController(
            kp=tracking_params['flight_KP'],
            ki=tracking_params['flight_KI'],
            kd=tracking_params['flight_KD']
        )
        self.flight_pid_y = PIDController(
            kp=tracking_params['flight_KP'],
            ki=tracking_params['flight_KI'],
            kd=tracking_params['flight_KD']
        )
        
        # 跟踪参数
        self.dead_zone = tracking_params['dead_zone']
        self.max_gimbal_speed = tracking_params['max_gimbal_speed']
        self.max_flight_speed = tracking_params['max_flight_speed']
        
        # 跟踪状态
        self.is_tracking = False
        self.last_target_position = None
        self.tracking_start_time = None
        
    def calculate_control_commands(self, frame, boxes):
        """计算跟踪控制命令"""
        if not self.enabled or len(boxes) != 1:
            self.is_tracking = False
            return None, None  # 返回空的控制命令
        
        # 获取单个目标
        x1, y1, x2, y2, conf, cls_id = boxes[0]
        frame_h, frame_w = frame.shape[:2]
        
        # 计算目标中心点（相对于图像中心）
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 图像中心
        image_center_x = frame_w / 2
        image_center_y = frame_h / 2
        
        # 计算归一化误差 [-1, 1]
        error_x = (target_center_x - image_center_x) / image_center_x
        error_y = (target_center_y - image_center_y) / image_center_y
        
        # 检查是否在死区内
        if abs(error_x) <= self.dead_zone and abs(error_y) <= self.dead_zone:
            # 在死区内，停止跟踪
            self.is_tracking = False
            return 0, 0, 0, 0  # 零速度指令
        
        # 开始跟踪
        self.is_tracking = True
        self.last_target_position = (target_center_x, target_center_y)
        
        # 根据跟踪模式计算控制命令
        gimbal_yaw_speed = 0
        gimbal_pitch_speed = 0
        flight_vx = 0
        flight_vy = 0
        
        if self.gimbal_tracking:
            # 云台跟踪 - 使用PID控制器
            gimbal_yaw_speed = self.gimbal_pid_x.update(-error_x)  # 注意正负方向
            gimbal_pitch_speed = self.gimbal_pid_y.update(error_y)
            
            # 限制最大速度
            gimbal_yaw_speed = np.clip(gimbal_yaw_speed, -self.max_gimbal_speed, self.max_gimbal_speed)
            gimbal_pitch_speed = np.clip(gimbal_pitch_speed, -self.max_gimbal_speed, self.max_gimbal_speed)
        
        if self.flight_tracking:
            # 飞机跟踪 - 使用PID控制器
            flight_vy = self.flight_pid_x.update(-error_x)  # 飞机左右移动
            flight_vx = self.flight_pid_y.update(error_y)   # 飞机前后移动
            
            # 限制最大速度
            flight_vx = np.clip(flight_vx, -self.max_flight_speed, self.max_flight_speed)
            flight_vy = np.clip(flight_vy, -self.max_flight_speed, self.max_flight_speed)
        
        return gimbal_pitch_speed, gimbal_yaw_speed, flight_vx, flight_vy

# --------------------------
# 错误处理装饰器（增强版）- 关键错误立即print
# --------------------------
def error_handler(max_retries=3, delay=1, critical=False, retry_exceptions=(Exception,)):
    """增强错误处理装饰器 - 智能重试机制，区分致命错误和临时错误"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    retries += 1
                    last_exception = e
                    error_msg = f"{func.__name__} 第{retries}次失败: {str(e)}"
                    
                    # 关键错误立即显示
                    if retries > max_retries and critical:
                        log.error(error_msg, immediate=True)
                    else:
                        log.error(error_msg, immediate=True)
                    
                    if retries > max_retries:
                        if critical:
                            raise RuntimeError(f"关键操作失败: {func.__name__}") from last_exception
                        else:
                            log.warning(f"非关键操作跳过: {func.__name__}", immediate=True)
                            return None
                    
                    print(f"🔄 {delay}秒后重试...")
                    time.sleep(delay)
                except Exception as e:
                    # 非重试异常直接抛出，立即显示
                    if critical:
                        log.exception(f"关键操作遇到致命错误: {func.__name__}", immediate=True)
                        raise RuntimeError(f"关键操作遇到致命错误: {func.__name__}") from e
                    else:
                        log.exception(f"非关键操作遇到致命错误，跳过: {func.__name__}", immediate=True)
                        return None
            return None
        return wrapper
    return decorator

# --------------------------
# TensorRT引擎加载 - 关键错误立即print
# --------------------------
class TRTInfer:
    """TensorRT推理引擎 - 跨平台优化版本，新人注意：这里负责AI模型推理"""
    
    def __init__(self, engine_path=None):
        if engine_path is None:
            engine_path = config.get("model_path", "best.engine")
            
        self.platform_cfg = platform_cfg
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # TensorRT日志
        self.engine_path = self._find_engine_file(engine_path)  # 智能查找引擎文件
        self.engine = self._load_engine(self.engine_path)  # 加载模型引擎
        self.is_destroyed = False  # 资源释放标记
        
        if not self.engine:
            log.critical("TensorRT引擎初始化失败", immediate=True)
            raise RuntimeError("❌ TensorRT引擎初始化失败")
            
        self.context = self.engine.create_execution_context()  # 创建推理上下文
        # 🔧 修复关键bug：变量名不一致问题
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        print(f"✅ TensorRT引擎加载成功！输入shape: {self.inputs[0]['shape']}")
        log.info(f"TensorRT引擎详细加载信息: {self.inputs[0]['shape']}")
        
        # 预热推理 - 避免首次推理耗时过长
        self._warmup()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 确保资源释放"""
        self.close()

    def _warmup(self):
        """预热推理 - 运行几次空推理来初始化CUDA上下文"""
        warmup_frames = config.get("warmup_frames", 10)
        print(f"🔥 预热推理 ({warmup_frames}帧)...")
        
        # 创建随机测试图像
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(warmup_frames):
            try:
                self.infer(dummy_img)
                if (i + 1) % 5 == 0:
                    print(f"🔥 预热进度: {i+1}/{warmup_frames}")
            except Exception as e:
                log.warning(f"预热推理失败: {e}")
                break
        
        print("✅ 预热完成")

    def _find_engine_file(self, path):
        """智能查找引擎文件 - 跨平台兼容，新人注意：自动匹配平台最优引擎"""
        if os.path.exists(path):
            return path
            
        # 根据平台特征查找合适的引擎文件
        platform_suffix = "arm" if self.platform_cfg.is_arm else "x86"
        available_engines = [f for f in os.listdir('.') if f.endswith('.engine')]
        
        # 优先查找平台匹配的引擎
        for engine_file in available_engines:
            if platform_suffix in engine_file.lower():
                print(f"🎯 找到平台匹配引擎: {engine_file}")
                return engine_file
        
        # 没有平台匹配就用第一个找到的
        if available_engines:
            selected = available_engines[0]
            log.warning(f"使用通用引擎: {selected} (可能不是最优性能)", immediate=True)
            return selected
            
        log.critical("没有找到任何TensorRT引擎文件！", immediate=True)
        raise FileNotFoundError("❌ 没有找到任何TensorRT引擎文件！")

    @error_handler(max_retries=2, critical=True, retry_exceptions=(IOError, RuntimeError))
    def _load_engine(self, path):
        """跨平台引擎加载 - 支持多种格式，新人注意：这里找模型文件"""
        print(f"🔄 加载引擎: {path} (大小: {os.path.getsize(path)//1024//1024}MB)")
        
        try:
            with open(path, "rb") as f:
                engine_data = f.read()  # 读取引擎文件
            
            runtime = trt.Runtime(self.TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_data)  # 反序列化引擎
            
            if not engine:
                raise RuntimeError("引擎反序列化失败 - 文件可能损坏")
                
            return engine
            
        except Exception as e:
            log.exception(f"引擎加载失败: {e}", immediate=True)
            raise

    def _allocate_buffers(self):
        """内存分配 - 平台优化版本，新人不用改这里"""
        inputs = []  # 输入缓冲区
        outputs = []  # 输出缓冲区
        bindings = []  # 绑定列表
        stream = cuda.Stream()  # CUDA流
        try:
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)  # 绑定名称
                shape = self.engine.get_binding_shape(i)  # 张量形状
                size = trt.volume(shape)  # 计算总元素数量
                dtype = trt.nptype(self.engine.get_binding_dtype(i))  # 数据类型
                
                # ARM平台特殊优化 - 内存分配策略不同
                if self.platform_cfg.is_arm:
                    host_mem = cuda.pagelocked_empty(size, dtype, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
                else:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                
                device_mem = cuda.mem_alloc(host_mem.nbytes)  # 分配设备内存
                bindings.append(int(device_mem))  # 添加到绑定列表
                
                if self.engine.binding_is_input(i):
                    inputs.append({
                        "host": host_mem,  # 主机内存
                        "device": device_mem,  # 设备内存
                        "shape": shape,  # 形状
                        "name": binding_name  # 名称
                    })
                else:
                    outputs.append({
                        "host": host_mem, 
                        "device": device_mem, 
                        "shape": shape,
                        "name": binding_name
                    })
            
            return inputs, outputs, bindings, stream
            
        except Exception as e:
            log.error(f"内存分配失败: {e}", immediate=True)
            raise

    def __del__(self):
        """资源清理 - 安全释放CUDA资源，防止内存泄漏和二次释放"""
        # 🔧 修复：检查属性是否存在，避免AttributeError
        if not hasattr(self, 'is_destroyed') or self.is_destroyed:
            return
            
        try:
            # 先释放上下文
            if hasattr(self, 'context') and self.context:
                del self.context
                self.context = None
                
            # 再释放流
            if hasattr(self, 'stream') and self.stream:
                del self.stream
                self.stream = None
                
            # 释放设备内存
            for buf in getattr(self, 'inputs', []):
                if 'device' in buf and buf['device']:
                    try:
                        buf['device'].free()
                    except Exception:
                        pass
            for buf in getattr(self, 'outputs', []):
                if 'device' in buf and buf['device']:
                    try:
                        buf['device'].free()
                    except Exception:
                        pass
                        
            # 最后释放引擎
            if hasattr(self, 'engine') and self.engine:
                del self.engine
                self.engine = None
                
            self.is_destroyed = True
            print("✅ TensorRT资源已释放")
        except Exception as e:
            log.error(f"资源释放过程中出现错误: {e}")

    def close(self):
        """显式关闭资源 - 比__del__更可靠"""
        if not self.is_destroyed:
            self.__del__()

    @error_handler(max_retries=1, retry_exceptions=(RuntimeError,))
    def infer(self, img):
        """执行推理 - 带预处理和后处理，新人注意：这里处理每帧图像"""
        # 预处理 - 把图像转换成模型需要的格式
        input_img, scale_ratio, pad_info = self._preprocess(img)
        if input_img is None:
            return None, None, None
            
        # 把预处理后的图像数据复制到输入缓冲区
        np.copyto(self.inputs[0]["host"], input_img.ravel())
        # 异步推理 - 把数据从主机内存拷贝到设备内存
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], 
            self.inputs[0]["host"], 
            self.stream
        )
        
        # ARM平台使用同步执行更稳定，x86用异步更快
        if self.platform_cfg.is_arm:
            self.context.execute_v2(bindings=self.bindings)  # 同步执行
        else:
            self.context.execute_async_v2(
                bindings=self.bindings, 
                stream_handle=self.stream.handle  # 异步执行
            )
        
        # 取回结果 - 把推理结果从设备内存拷贝回主机内存
        output_data = []
        for out in self.outputs:  # 🔧 修复：这里使用self.outputs而不是self.output
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            output_data.append(out["host"].copy().reshape(out["shape"]))
        
        self.stream.synchronize()  # 等待流完成
        return output_data, scale_ratio, pad_info

    def _preprocess(self, img):
        """图像预处理 - 支持暗光增强和保持宽高比，返回预处理图像、缩放比例和填充信息"""
        if img is None:
            log.warning("输入图像为空", immediate=True)
            return None, None, None
            
        try:
            original_h, original_w = img.shape[:2]
            
            # 🔧 使用配置类方法获取参数
            enhance_params = config.get_image_enhancement_params()
            
            # 图像增强 - 只在启用时执行
            if enhance_params['enabled']:
                if not self.platform_cfg.is_arm:
                    # x86平台使用CLAHE增强
                    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    clahe = cv2.createCLAHE(
                        clipLimit=enhance_params['clahe_clip_limit'],
                        tileGridSize=(8, 8)
                    )
                    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                else:
                    # ARM平台使用轻量级增强
                    img = cv2.convertScaleAbs(img, 
                                            alpha=enhance_params['brightness_alpha'], 
                                            beta=enhance_params['brightness_beta'])
            
            # 保持宽高比的resize - 避免图像变形
            # 计算缩放比例，保持宽高比，短边缩放到IMG_SIZE
            scale = min(IMG_SIZE / original_w, IMG_SIZE / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            
            # 创建目标图像并填充黑边
            resized_img = cv2.resize(img, (new_w, new_h))
            padded_img = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)  # 114是YOLO的填充值
            
            # 计算填充位置（居中）
            dx = (IMG_SIZE - new_w) // 2
            dy = (IMG_SIZE - new_h) // 2
            padded_img[dy:dy+new_h, dx:dx+new_w] = resized_img
            
            # 保存填充信息用于后处理坐标还原
            pad_info = (dx, dy, scale, original_w, original_h)
            
            # 格式转换
            padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)  # BGR转RGB
            padded_img = padded_img.astype(np.float32) / 255.0  # 归一化到0-1
            padded_img = padded_img.transpose(2, 0, 1)  # HWC -> CHW
            padded_img = np.ascontiguousarray(padded_img)  # 确保内存连续
            
            return padded_img, scale, pad_info
            
        except Exception as e:
            log.error(f"图像预处理失败: {e}", immediate=True)
            return None, None, None

# --------------------------
# 大疆Eport通讯
# --------------------------
class EportClient:
    """大疆Eport通讯客户端 - 支持Mavlink和串口协议"""
    
    def __init__(self):
        self.serial_port = config.get("serial_port", "/dev/ttyUSB0")  # Eport串口路径
        self.serial_baudrate = config.get("serial_baudrate", 921600)  # Eport波特率
        self.mavlink_conn = None  # Mavlink连接（串口通信核心）
        self.is_connected = False  # 连接状态
        self.connection_type = f"Serial:{self.serial_port}"  # 连接类型标识
        self.heartbeat_timeout = config.get("heartbeat_timeout", 5)  # 🔧 可以改：Mavlink心跳超时时间
        self.last_heartbeat_time = 0  # 最后心跳时间
        # 🔧 协议可配置化
        self.mavlink_command = config.get_mavlink_command()  # 可配置的Mavlink命令
        self._connect()  # 建立连接

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 确保资源释放"""
        self._cleanup_connection()

    def _cleanup_connection(self):
        """清理连接资源 - 防止串口资源泄漏"""
        if self.mavlink_conn:
            try:
                self.mavlink_conn.close()
            except Exception:
                pass
            finally:
                self.mavlink_conn = None
                
        self.is_connected = False
        print("🔌 Eport串口连接已清理")

    @error_handler(max_retries=3, delay=2, retry_exceptions=(ConnectionError, OSError))
    def _connect(self):
        """增强连接机制 - 串口+Mavlink心跳验证，新人注意：连接失败会自动重试"""
        print(f"🔄 连接Eport串口 {self.serial_port} @ {self.serial_baudrate} ({self.connection_type})...")
        
        try:
            # 清理旧连接
            self._cleanup_connection()
            
            # 1. 给串口添加读写权限（避免每次手动执行sudo chmod）
            if platform.system().lower() == 'linux':
                os.system(f"sudo chmod 666 {self.serial_port}")
            
            # 2. 建立Mavlink串口连接（大疆Eport默认支持Mavlink协议）
            self.mavlink_conn = mavutil.mavlink_connection(
                self.serial_port,
                baud=self.serial_baudrate,
                autoreconnect=True,  # 自动重连
                source_system=255,  # 系统ID
                source_component=0,  # 组件ID
                use_native=False  # 兼容更多设备
            )
            
            # 3. 等待Mavlink心跳包，验证连接有效性
            print("🔄 等待Mavlink心跳包...")
            if self.mavlink_conn.wait_heartbeat(timeout=self.heartbeat_timeout):
                self.last_heartbeat_time = time.time()
                print(f"✅ Mavlink协议验证成功（系统ID:{self.mavlink_conn.target_system} 组件ID:{self.mavlink_conn.target_component}）")
                self.is_connected = True
                print(f"🎉 Eport串口连接成功！协议: {self.connection_type}")
            else:
                raise ConnectionError("Mavlink心跳验证超时 - 串口路径/波特率可能错误")
            
        except Exception as e:
            self._cleanup_connection()
            raise ConnectionError(f"串口连接失败: {e}")

    def check_heartbeat(self):
        """检查心跳连接状态"""
        if not self.is_connected:
            return False
            
        # 检查是否超过心跳超时时间
        current_time = time.time()
        if current_time - self.last_heartbeat_time > self.heartbeat_timeout:
            log.warning("心跳连接超时", immediate=True)
            self.is_connected = False
            return False
            
        return True

    @error_handler(max_retries=2, retry_exceptions=(ConnectionError, OSError))
    def send_detection(self, boxes, frame_size=(1920, 1080)):
        """发送检测结果，把检测框发给飞机"""
        if not self.check_heartbeat():
            log.warning("Eport连接异常，尝试重连...", immediate=True)
            try:
                self._connect()
            except Exception as e:
                log.error(f"重连失败: {e}", immediate=True)
                return
        
        if len(boxes) == 0:  # 没有检测到目标就不发送
            return
            
        try:
            sent_count = 0
            for i, box in enumerate(boxes):
                if i >= MAX_TARGETS_PER_FRAME:  # 限制单帧最大目标数
                    if sent_count == MAX_TARGETS_PER_FRAME:
                        log.warning(f"检测到 {len(boxes)} 个目标，只发送前 {MAX_TARGETS_PER_FRAME} 个", immediate=True)
                    break
                    
                x1, y1, x2, y2, conf, cls_id = box  # 解包检测框信息
                
                # 归一化坐标 - 把像素坐标转换成0-1的相对坐标
                center_x = ((x1 + x2) / 2) / frame_size[0]  # 中心点X坐标
                center_y = ((y1 + y2) / 2) / frame_size[1]  # 中心点Y坐标
                width = (x2 - x1) / frame_size[0]  # 框宽度
                height = (y2 - y1) / frame_size[1]  # 框高度
                
                # 通过Mavlink协议发送（串口通信核心逻辑）
                self._send_mavlink(center_x, center_y, width, height, conf, cls_id)
                
                sent_count += 1
                
            if sent_count > 0:
                print(f"📡 发送 {sent_count} 个检测目标")
            
        except Exception as e:
            log.error(f"发送检测结果失败: {e}", immediate=True)
            self.is_connected = False  # 标记为未连接，下次会重连

    def _send_mavlink(self, x, y, w, h, conf, cls_id):
        """通过Mavlink发送检测结果，新人注意：这是大疆Eport串口专用协议"""
        try:
            # 🔧 使用可配置的Mavlink命令，增强协议兼容性
            msg = self.mavlink_conn.mav.command_long_encode(
                self.mavlink_conn.target_system,  # 目标系统（飞机）
                self.mavlink_conn.target_component,  # 目标组件（Eport模块）
                self.mavlink_command,  # 🔧 可配置的命令类型
                0,  # 确认标志
                x, y, w, h,  # 坐标和尺寸（归一化后）
                conf,  # 置信度
                int(cls_id)  # 类别ID
            )
            self.mavlink_conn.mav.send(msg)  # 发送消息
        except Exception as e:
            log.error(f"Mavlink发送失败: {e}")

    def send_gimbal_speed(self, pitch_speed, yaw_speed):
        """
        发送云台速度控制指令。
        pitch_speed: 俯仰速度（度/秒），正数向上
        yaw_speed: 偏航速度（度/秒），正数向右
        """
        try:
            # 使用 MAV_CMD_DO_GIMBAL_MANAGER_TILTPAN 或类似命令
            # 具体命令号和参数格式需查阅 PSDK/Mavlink 文档
            msg = self.mavlink_conn.mav.command_long_encode(
                self.mavlink_conn.target_system,
                self.mavlink_conn.target_component,
                mavutil.mavlink.MAV_CMD_DO_GIMBAL_MANAGER_TILTPAN,  # 示例命令
                0,  # 确认标志
                0, 0,  # 保留/自定义参数
                pitch_speed,  # 参数3: 俯仰速度
                yaw_speed,    # 参数4: 偏航速度
                0, 0, 0       # 其他参数
            )
            self.mavlink_conn.mav.send(msg)
            # print(f"📤 云台控制: pitch={pitch_speed}, yaw={yaw_speed}")
        except Exception as e:
            log.error(f"云台控制指令发送失败: {e}")

    def send_flight_velocity(self, vx, vy, vz, yaw_rate=0):
        """
        发送飞机速度控制指令（虚拟摇杆模式）。
        vx, vy, vz: 前/右/下的速度（米/秒）
        yaw_rate: 偏航角速度（度/秒）
        """
        try:
            # 使用 MAV_CMD_DO_SET_ROI_LOCATION 或直接通过 SET_POSITION_TARGET_LOCAL_NED 消息
            # 这里是一个简化示例，实际需要构造更复杂的 Mavlink 消息
            # 通常需要设置控制模式（如速度模式）并持续发送
            msg = self.mavlink_conn.mav.set_position_target_local_ned_encode(
                0,  # 时间戳
                self.mavlink_conn.target_system,
                self.mavlink_conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,  # 速度控制掩码 (忽略位置，控制速度)
                0, 0, 0,  # 位置 (忽略)
                vx, vy, vz,  # 速度
                0, 0, 0,     # 加速度 (忽略)
                0, 0         # 偏航，偏航率
            )
            self.mavlink_conn.mav.send(msg)
            # print(f"📤 飞行控制: vx={vx:.2f}, vy={vy:.2f}")
        except Exception as e:
            log.error(f"飞行控制指令发送失败: {e}")

# --------------------------
# RTSP流读取（增强版）- 关键状态立即print
# --------------------------
class RTSPCapture:
    """RTSP视频流捕获 - 带断线重连和最新帧缓冲，新人注意：这里负责获取摄像头视频"""
    
    def __init__(self, url=None, buffer_size=None):
        self.url = url or config.get("rtsp_url", "rtsp://192.168.1.20:554/main.264")  # RTSP流地址
        self.cap = None  # OpenCV视频捕获对象
        self.frame = None  # 当前帧
        self.frame_timestamp = 0  # 帧时间戳
        self.lock = threading.Lock()  # 线程锁，保证帧读取安全
        self.is_running = False  # 运行状态
        self.buffer_size = buffer_size or config.get("stream_buffer_size", 3)  # 🔧 可以改：缓冲区大小，影响流畅度
        self.frame_info = {"width": 0, "height": 0, "fps": 0}  # 帧信息
        self._start_capture()  # 开始捕获

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 确保资源释放"""
        self.stop()

    def _start_capture(self):
        """启动RTSP捕获线程，新人不用改这里"""
        self.is_running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()  # 后台线程
        print(f"🎥 RTSP流捕获启动: {self.url}")

    def _capture_loop(self):
        """RTSP捕获主循环 - 带指数退避重连和帧时间戳，新人注意：这里会自动重连断流的摄像头"""
        reconnect_delay = config.get("reconnect_delay", 2)  # 初始重连延迟
        max_reconnect_delay = config.get("max_reconnect_delay", 30)  # 🔧 可以改：最大重连延迟，单位秒
        frame_counter = 0  # 帧计数器
        consecutive_failures = 0  # 连续失败次数
        last_success_time = time.time()  # 最后成功时间
        
        while self.is_running:
            try:
                # 添加小延迟避免CPU占用过高
                time.sleep(0.01)
                
                if self.cap is None or not self.cap.isOpened():
                    print(f"🔄 连接RTSP流: {self.url}")
                    self.cap = cv2.VideoCapture(self.url)  # 创建视频捕获
                    
                    if not self.cap.isOpened():
                        raise RuntimeError("RTSP流打开失败")
                    
                    # 获取流信息
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.frame_info = {"width": width, "height": height, "fps": fps}
                    
                    # 平台特定的优化 - 不同平台设置不同参数
                    if platform_cfg.is_arm:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ARM平台缓冲区小一点
                        self.cap.set(cv2.CAP_PROP_FPS, 15)  # 🔧 可以改：ARM平台帧率限制
                    else:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)  # x86用大缓冲区
                    
                    reconnect_delay = config.get("reconnect_delay", 2)  # 重置重连延迟
                    consecutive_failures = 0  # 重置连续失败计数
                    print(f"✅ RTSP流连接成功！分辨率: {width}x{height} FPS: {fps:.1f}")
                
                # 原子操作：读取一帧
                ret, new_frame = self.cap.read()
                if ret and new_frame is not None:
                    frame_counter += 1
                    consecutive_failures = 0  # 成功读取，重置失败计数
                    last_success_time = time.time()
                    with self.lock:
                        # 只保留最新帧，避免帧积压导致的延迟
                        self.frame = new_frame.copy()
                        self.frame_timestamp = frame_counter
                else:
                    consecutive_failures += 1
                    # 检查是否长时间无帧
                    if time.time() - last_success_time > 10:  # 10秒无帧认为连接异常
                        raise RuntimeError("RTSP流长时间无数据")
                    elif consecutive_failures >= 5:  # 连续5次读取失败才认为是真正失败
                        raise RuntimeError("RTSP流读取失败")
                    else:
                        time.sleep(0.1)  # 短暂等待后重试
            except Exception as e:
                log.error(f"RTSP流错误: {e}", immediate=True)
                self._cleanup_capture()  # 清理资源
                
                print(f"⏳ {reconnect_delay}秒后重连。。。")
                print("我需要165、JK白丝、声音甜美的双马尾学妹的安慰 (╥╯^╰╥) ")
                time.sleep(reconnect_delay)
                
                # 指数退避：每次重连失败后等待时间加倍，但不超过最大值
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    def _cleanup_capture(self):
        """清理捕获资源，新人不用改这里"""
        if self.cap:
            try:
                self.cap.release()  # 释放摄像头
            except Exception as e:
                log.error(f"释放摄像头资源失败: {e}")
            finally:
                self.cap = None

    def get_frame(self):
        """获取当前帧（线程安全），只返回最新帧避免延迟"""
        with self.lock:
            # 🔧 修复：确保frame不为None再调用copy()
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_frame_info(self):
        """获取帧信息"""
        return self.frame_info.copy()

    def stop(self):
        """停止捕获，新人不用改这里"""
        self.is_running = False
        self._cleanup_capture()
        print("🛑 RTSP流捕获已停止")

# --------------------------
# 自适应帧率控制器
# --------------------------
class AdaptiveFPSController:
    """自适应帧率控制器 - 根据系统负载动态调整帧率"""
    def __init__(self, target_fps=None):
        self.target_fps = target_fps or config.get("target_fps", 30)
        self.min_fps = config.get("min_fps", 10)
        self.max_fps = config.get("max_fps", 60)
        self.adaptive_fps = config.get("adaptive_fps", True)
        
        self.target_delay = 1.0 / self.target_fps
        self.last_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        self.performance_history = []  # 性能历史记录
        self.max_history_size = 30  # 最大历史记录数
        
        # 🔧 添加性能阈值配置
        self.thresholds = config.get_performance_thresholds()
        
    def wait(self):
        """等待以达到目标帧率"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed < self.target_delay:
            time.sleep(self.target_delay - elapsed)
            
        self.last_time = time.time()
        self.frame_count += 1
        
        # 自适应帧率调整 - 结合性能数据动态调整
        if self.adaptive_fps and len(self.performance_history) >= 10:
            avg_processing_time = np.mean(self.performance_history)
            current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else self.target_fps
            
            # 🔧 智能的帧率调整策略
            # 如果处理时间过长，降低目标帧率
            if avg_processing_time > self.target_delay * 1.2 and self.target_fps > self.min_fps:
                self.target_fps = max(self.min_fps, self.target_fps - 1)
                self.target_delay = 1.0 / self.target_fps
                print(f"🔄 检测到性能瓶颈，降低帧率至: {self.target_fps}FPS")
            # 如果处理时间充足，提高目标帧率
            elif avg_processing_time < self.target_delay * 0.8 and self.target_fps < self.max_fps:
                self.target_fps = min(self.max_fps, self.target_fps + 1)
                self.target_delay = 1.0 / self.target_fps
                print(f"🔄 性能充足，提高帧率至: {self.target_fps}FPS")
    
    def update_performance(self, processing_time):
        """更新性能数据"""
        self.performance_history.append(processing_time)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
    def get_actual_fps(self):
        """获取实际帧率"""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
    
    def get_target_fps(self):
        """获取当前目标帧率"""
        return self.target_fps

# --------------------------
# 实时监控统计
# --------------------------
class RealTimeMonitor:
    """实时监控统计 - 实时显示关键指标，新人注意：这里显示系统运行状态"""
    def __init__(self):
        self.start_time = time.time()  # 开始时间
        self.frame_count = 0  # 处理帧数统计
        self.error_count = 0  # 错误次数统计
        self.detection_count = 0  # 检测到目标次数
        self.last_report_time = self.start_time  # 上次报告时间
        self.fps_counter = AdaptiveFPSController()  # 自适应帧率控制器
        self.health_report_interval = config.get("health_report_interval", 30)  # 健康报告间隔
        self.last_fps_display = 0  # 上次FPS显示时间
        self.fps_display_interval = 5  # FPS显示间隔（秒）
        self.process = psutil.Process(os.getpid())  # 进程信息
        self.connection_type = "未连接"  # 初始化连接类型，避免健康报告报错
        self.tracking_status = "未跟踪"  # 跟踪状态
        self.performance_stats = {
            "inference_times": [],
            "preprocess_times": [],
            "postprocess_times": [],
            "network_times": []
        }
        
        # 🔧 性能阈值配置
        self.thresholds = config.get_performance_thresholds()
        self.performance_warnings = {
            'high_cpu': False,
            'high_memory': False,
            'slow_inference': False
        }
        
    def update_frame(self, detected_objects=0, processing_time=0, tracking_status="未跟踪"):
        """更新帧统计"""
        self.frame_count += 1
        if detected_objects > 0:
            self.detection_count += 1
            
        # 更新性能数据
        if processing_time > 0:
            self.fps_counter.update_performance(processing_time)
            
        # 更新跟踪状态
        self.tracking_status = tracking_status
        
        # 帧率控制
        self.fps_counter.wait()
        
        # 每5秒显示一次实时FPS
        current_time = time.time()
        if current_time - self.last_fps_display >= self.fps_display_interval:
            actual_fps = self.fps_counter.get_actual_fps()
            target_fps = self.fps_counter.get_target_fps()
            detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            # 🔧 性能告警检查
            self._check_performance_warnings(cpu_percent, memory_mb, processing_time * 1000)
            
            status_indicator = "✅" if not any(self.performance_warnings.values()) else "⚠️"
            
            print(f"📊 {status_indicator} 实时状态: FPS={actual_fps:.1f}/{target_fps} | 检测率={detection_rate:.1f}% | "
                  f"帧数={self.frame_count} | 内存={memory_mb:.1f}MB | CPU={cpu_percent:.1f}% | 跟踪={self.tracking_status}")
            self.last_fps_display = current_time
        
        # 定期详细报告
        if current_time - self.last_report_time >= self.health_report_interval:
            self.report_health()
            self.last_report_time = current_time
    
    def _check_performance_warnings(self, cpu_percent, memory_mb, inference_time_ms):
        """检查性能告警阈值"""
        # 重置警告状态
        for key in self.performance_warnings:
            self.performance_warnings[key] = False
        
        # 检查CPU使用率
        if cpu_percent > self.thresholds['high_cpu']:
            self.performance_warnings['high_cpu'] = True
            log.warning(f"CPU使用率过高: {cpu_percent:.1f}% > {self.thresholds['high_cpu']}%", immediate=True)
        
        # 检查内存使用率（估算）
        total_memory = psutil.virtual_memory().total / 1024 / 1024
        memory_percent = (memory_mb / total_memory) * 100
        if memory_percent > self.thresholds['high_memory']:
            self.performance_warnings['high_memory'] = True
            log.warning(f"内存使用率过高: {memory_percent:.1f}% > {self.thresholds['high_memory']}%", immediate=True)
        
        # 检查推理耗时
        if inference_time_ms > self.thresholds['slow_inference']:
            self.performance_warnings['slow_inference'] = True
            log.warning(f"推理耗时过长: {inference_time_ms:.1f}ms > {self.thresholds['slow_inference']}ms", immediate=True)
    
    def update_performance_stats(self, inference_time, preprocess_time, postprocess_time, network_time):
        """更新性能统计数据"""
        self.performance_stats["inference_times"].append(inference_time)
        self.performance_stats["preprocess_times"].append(preprocess_time)
        self.performance_stats["postprocess_times"].append(postprocess_time)
        self.performance_stats["network_times"].append(network_time)
        
        # 保持历史数据大小
        for key in self.performance_stats:
            if len(self.performance_stats[key]) > 100:
                self.performance_stats[key] = self.performance_stats[key][-100:]
    
    def report_health(self):
        """报告系统健康状态，新人注意：按's'键可以手动查看这个报告"""
        current_time = time.time()
        elapsed = current_time - self.start_time  # 运行时长
        frames_per_sec = self.frame_count / elapsed if elapsed > 0 else 0  # 计算FPS
        actual_fps = self.fps_counter.get_actual_fps()  # 实际帧率
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        # 计算性能统计
        perf_stats = {}
        for key, times in self.performance_stats.items():
            if times:
                perf_stats[key] = {
                    "avg": np.mean(times) * 1000,
                    "max": np.max(times) * 1000,
                    "min": np.min(times) * 1000
                }
        
        # 🔧 性能告警状态显示
        warning_status = "✅ 正常" if not any(self.performance_warnings.values()) else "⚠️ 告警"
        
        print(f"\n📊 系统健康报告 ({warning_status}):")
        print(f"   运行时间: {elapsed:.1f}秒")
        print(f"   处理帧数: {self.frame_count}")
        print(f"   检测次数: {self.detection_count}")
        print(f"   检测率: {detection_rate:.1f}%")
        print(f"   平均FPS: {frames_per_sec:.1f}")
        print(f"   实际FPS: {actual_fps:.1f}")
        print(f"   目标FPS: {self.fps_counter.get_target_fps()}")
        print(f"   错误次数: {self.error_count}")
        print(f"   内存使用: {memory_mb:.1f}MB")
        print(f"   CPU使用: {cpu_percent:.1f}%")
        print(f"   跟踪状态: {self.tracking_status}")
        print(f"   平台: {platform_cfg.system} {platform_cfg.machine}")
        print(f"   GUI支持: {platform_cfg.has_gui}")
        print(f"   目标类别: {CLASSES}")
        print(f"   Eport连接: {self.connection_type}")
        
        if perf_stats:
            print(f"   性能统计 (ms):")
            for key, stats in perf_stats.items():
                print(f"     {key}: 平均{stats['avg']:.1f}ms, 最大{stats['max']:.1f}ms, 最小{stats['min']:.1f}ms")
        
        # 🔧 显示性能阈值配置
        print(f"   性能阈值:")
        print(f"     CPU告警: >{self.thresholds['high_cpu']}%")
        print(f"     内存告警: >{self.thresholds['high_memory']}%")
        print(f"     推理耗时告警: >{self.thresholds['slow_inference']}ms")
        
    def record_error(self):
        """记录错误"""
        self.error_count += 1

# --------------------------
# 性能分析器
# --------------------------
class PerformanceProfiler:
    """性能分析器 - 分析各模块耗时"""
    def __init__(self):
        self.timers = {}
        self.enabled = config.get("enable_frame_stats", True)
        self.frame_count = 0
        
    def start_timer(self, name):
        """开始计时"""
        if self.enabled:
            self.timers[name] = time.time()
            
    def end_timer(self, name):
        """结束计时并返回耗时"""
        if self.enabled and name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            return elapsed
        return 0
    
    def profile_frame(self, frame_processing_time, inference_time, postprocess_time, network_time):
        """分析单帧性能"""
        if not self.enabled:
            return
            
        total_time = frame_processing_time + inference_time + postprocess_time + network_time
        fps = 1.0 / total_time if total_time > 0 else 0
        
        self.frame_count += 1
            
        if self.frame_count % 30 == 0:
            # 🔧 性能状态指示器
            status = "✅" if inference_time * 1000 < config.get_performance_thresholds()['slow_inference'] else "⚠️"
            
            print(f"⚡ {status} 性能分析: 总耗时={total_time*1000:.1f}ms | "
                  f"推理={inference_time*1000:.1f}ms | "
                  f"后处理={postprocess_time*1000:.1f}ms | "
                  f"串口通信={network_time*1000:.1f}ms | "
                  f"FPS={fps:.1f}")

# --------------------------
# YOLOv5后处理（多尺度输出+按类别NMS+填充坐标还原）
# --------------------------
def postprocess(outputs, img_shape, scale_ratio, pad_info):
    """适配YOLOv5多尺度输出，注意：这里把模型输出转成检测框，支持填充还原"""
    h, w = img_shape  # 原图高度和宽度
    all_boxes = []  # 所有检测框的列表
    
    # 处理每个尺度的输出
    for out in outputs:
        # 解析YOLOv5输出：(batch, anchors*grid*grid, 85)
        # 85 = 4(坐标) + 1(置信度) + 80(类别概率)
        out = out.reshape(-1, YOLO_OUTPUT_DIM)  # reshape成二维
        
        # 过滤低置信度 - 用前面设置的CONF_THRES
        mask = out[:, 4] > CONF_THRES  # 第5列是物体置信度
        out = out[mask]  # 只保留高置信度的检测
        
        if len(out) == 0:  # 这个尺度没有检测到目标
            continue
        
        # 坐标还原（xywh -> xyxy，考虑填充）
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]  # 中心点坐标和宽高
        
        if pad_info:
            # 有填充信息：需要从填充图像坐标还原到原图坐标
            dx, dy, scale, orig_w, orig_h = pad_info
            
            # 从填充图像坐标转换到resize后图像坐标
            x1_resized = (cx - bw/2) * IMG_SIZE - dx
            y1_resized = (cy - bh/2) * IMG_SIZE - dy
            x2_resized = (cx + bw/2) * IMG_SIZE - dx
            y2_resized = (cy + bh/2) * IMG_SIZE - dy
            
            # 从resize后图像坐标转换到原图坐标
            x1 = x1_resized / scale
            y1 = y1_resized / scale
            x2 = x2_resized / scale
            y2 = y2_resized / scale
        else:
            # 无填充信息：直接缩放（兼容旧版本）
            x1 = (cx - bw/2) * IMG_SIZE / scale_ratio
            y1 = (cy - bh/2) * IMG_SIZE / scale_ratio
            x2 = (cx + bw/2) * IMG_SIZE / scale_ratio
            y2 = (cy + bh/2) * IMG_SIZE / scale_ratio
        
        # 限制坐标在图像范围内
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        conf = out[:, 4] * out[:, 5:].max(axis=1)  # 置信度 = 物体置信度 * 最大类别概率
        cls_ids = out[:, 5:].argmax(axis=1)  # 类别ID = 概率最大的类别索引
        
        # 过滤无效类别ID
        valid_mask = cls_ids < len(CLASSES)
        if np.any(valid_mask):
            valid_boxes = np.column_stack([x1[valid_mask], y1[valid_mask], x2[valid_mask], 
                                         y2[valid_mask], conf[valid_mask], cls_ids[valid_mask]])
            all_boxes.append(valid_boxes)
    
    if len(all_boxes) == 0:  # 所有尺度都没有检测到目标
        return []
    
    # 合并所有尺度的框
    all_boxes = np.concatenate(all_boxes, axis=0)
    
    # 按类别NMS（避免不同类别互相过滤）- 🔧 可以改：NMS参数在全局配置里
    unique_cls = np.unique(all_boxes[:, 5])  # 所有不重复的类别
    final_boxes = []  # 最终框列表
    
    for cls in unique_cls:
        cls_boxes = all_boxes[all_boxes[:, 5] == cls]  # 当前类别的所有框
        if len(cls_boxes) == 0:
            continue
            
        # 将框坐标转换为(x, y, w, h)格式供NMS使用
        boxes_xywh = []
        for box in cls_boxes:
            x1, y1, x2, y2 = box[:4]
            boxes_xywh.append([x1, y1, x2-x1, y2-y1])
            
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,  # 框坐标 (x,y,w,h)
            cls_boxes[:, 4].tolist(),  # 置信度
            CONF_THRES,  # 置信度阈值
            IOU_THRES  # IOU阈值
        )
        if len(indices) > 0:
            # 处理不同版本的OpenCV返回格式
            if hasattr(indices, 'shape') and len(indices.shape) > 1:
                indices = indices.flatten()
            final_boxes.append(cls_boxes[indices])  # 添加NMS后的框
    
    return np.concatenate(final_boxes, axis=0) if final_boxes else []  # 返回最终检测结果

# --------------------------
# 图像保存工具
# --------------------------
class ImageSaver:
    """图像保存工具 - 保存检测结果"""
    def __init__(self):
        self.save_path = config.get("save_detection_path", "detections")
        self.save_enabled = config.get("save_detection_images", False)
        
        if self.save_enabled:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            print(f"📁 检测图像保存路径: {self.save_path}")
    
    def save_detection_image(self, frame, boxes, frame_count):
        """保存检测图像"""
        if not self.save_enabled:
            return
            
        try:
            # 在图像上绘制检测框
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                
                if cls_id >= len(CLASSES):
                    continue
                    
                label = f"{CLASSES[cls_id]} {conf:.2f}"
                
                # 画检测框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                
                # 画标签背景
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, 
                             (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), 
                             (0, 255, 255), -1)
                
                # 写标签文字
                cv2.putText(frame, label, 
                           (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}_{frame_count:06d}.jpg"
            filepath = os.path.join(self.save_path, filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            print(f"📸 检测图像已保存: {filename}")
            
        except Exception as e:
            log.error(f"保存检测图像失败: {e}")

# --------------------------
# 主程序
# --------------------------
@error_handler(max_retries=float('inf'), retry_exceptions=(RuntimeError, ConnectionError))  # 无限重试，直到用户退出
def main_loop(trt_infer, eport_client, rtsp_cap, monitor, profiler, tracking_manager, image_saver):
    """主推理循环 - 带完整错误处理，这是程序的核心循环"""
    frame_start_time = time.time()
    
    # 获取当前帧
    frame = rtsp_cap.get_frame()
    if frame is None:
        log.warning("获取帧失败，跳过...", immediate=True)
        time.sleep(0.1)  # 短暂等待后继续
        return
    
    profiler.start_timer("preprocess")
    # 预处理计时在TRTInfer.infer内部完成，这里只记录总时间
    preprocess_time = 0
    
    profiler.start_timer("inference")
    # 推理 - 调用AI模型检测目标
    outputs, scale_ratio, pad_info = trt_infer.infer(frame)
    inference_time = profiler.end_timer("inference")
    
    if outputs is None:  # 推理失败
        monitor.record_error()
        return
    
    profiler.start_timer("postprocess")
    # 后处理 - 把模型输出转换成检测框
    boxes = postprocess(outputs, frame.shape[:2], scale_ratio, pad_info)
    postprocess_time = profiler.end_timer("postprocess")
    
    detected_count = len(boxes)
    
    # 绘制检测结果 - 在图像上画框和标签
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box  # 解包框信息
        cls_id = int(cls_id)
        
        # 验证类别ID有效性
        if cls_id >= len(CLASSES):
            continue
            
        label = f"{CLASSES[cls_id]} {conf:.2f}"  # 标签 = 类别名 + 置信度
        
        # 画检测框 - 🔧 可以改：(0, 255, 255)是黄色，2是线宽
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        
        # 画标签背景 - 让文字更清晰
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - label_size[1] - 10),  # 背景矩形左上角
                     (int(x1) + label_size[0], int(y1)),  # 背景矩形右下角
                     (0, 255, 255), -1)  # -1表示填充
        
        # 写标签文字 - 🔧 可以改：0.6是字体大小，2是线宽
        cv2.putText(frame, label, 
                   (int(x1), int(y1) - 5),  # 文字位置
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # 黑色文字
    
    # 计算跟踪控制命令
    gimbal_pitch_speed, gimbal_yaw_speed, flight_vx, flight_vy = 0, 0, 0, 0
    tracking_status = "未跟踪"
    
    if tracking_manager.enabled:
        gimbal_pitch_speed, gimbal_yaw_speed, flight_vx, flight_vy = tracking_manager.calculate_control_commands(frame, boxes)
        if tracking_manager.is_tracking:
            tracking_status = "正在跟踪"
        else:
            tracking_status = "目标丢失"
    
    # 发送跟踪控制命令
    if tracking_manager.gimbal_tracking and gimbal_pitch_speed != 0 and gimbal_yaw_speed != 0:
        eport_client.send_gimbal_speed(gimbal_pitch_speed, gimbal_yaw_speed)
    
    if tracking_manager.flight_tracking and flight_vx != 0 and flight_vy != 0:
        eport_client.send_flight_velocity(flight_vx, flight_vy, 0)
    
    profiler.start_timer("network")
    # 发送检测结果到无人机
    if detected_count > 0:
        eport_client.send_detection(boxes, frame_size=(frame.shape[1], frame.shape[0]))
    network_time = profiler.end_timer("network")
    
    frame_processing_time = time.time() - frame_start_time
    
    # 性能分析
    profiler.profile_frame(frame_processing_time, inference_time, postprocess_time, network_time)
    
    # 更新性能统计数据
    monitor.update_performance_stats(inference_time, preprocess_time, postprocess_time, network_time)
    
    # 更新监控统计
    monitor.update_frame(detected_count, frame_processing_time, tracking_status)
    
    # 保存检测图像
    if detected_count > 0 and config.get("save_detection_images", False):
        image_saver.save_detection_image(frame.copy(), boxes, monitor.frame_count)
    
    # 只在支持GUI时显示窗口
    if platform_cfg.has_gui and config.get("enable_detection_display", True):
        # 在画面上显示统计信息
        stats_text = f"FPS: {monitor.fps_counter.get_actual_fps():.1f}/{monitor.fps_counter.get_target_fps()} | Targets: {detected_count} | Track: {tracking_status}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示连接状态
        conn_status = "Eport: Connected" if eport_client.is_connected else "Eport: Disconnected"
        cv2.putText(frame, conn_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if eport_client.is_connected else (0, 0, 255), 2)
        
        # 🔧 性能状态指示
        status_color = (0, 255, 0)  # 默认绿色
        if any(monitor.performance_warnings.values()):
            status_color = (0, 165, 255)  # 橙色警告
        if monitor.performance_warnings['high_cpu'] or monitor.performance_warnings['slow_inference']:
            status_color = (0, 0, 255)  # 红色严重
        
        cv2.putText(frame, "●", (frame.shape[1] - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        cv2.imshow("DJI M350 RTK + TensorRT 目标检测", frame)
        
        # 检查窗口是否被关闭 - 如果用户关了窗口就退出
        if cv2.getWindowProperty("DJI M350 RTK + TensorRT 目标检测", cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt("用户关闭了显示窗口")

if __name__ == "__main__":
    # 设置增强日志系统
    logger, log_file_path = setup_logging()
    
    # 重新初始化log实例，使用新的logger
    global log
    log = HybridLogger()
    
    print("🚀 初始化大疆M350 RTK目标检测系统...")
    print(f"📝 详细日志请查看: {log_file_path}")
    print("=" * 50)
    
    # 验证配置
    if not config.validate():
        log.critical("配置验证失败，请检查config.json", immediate=True)
        sys.exit(1)
    
    monitor = RealTimeMonitor()  # 创建实时监控器
    profiler = PerformanceProfiler()  # 创建性能分析器
    tracking_manager = TrackingManager()  # 创建跟踪管理器
    image_saver = ImageSaver()  # 创建图像保存器
    
    try:
        # 使用上下文管理器确保资源正确释放
        with TRTInfer() as trt_infer, \
             EportClient() as eport_client, \
             RTSPCapture() as rtsp_cap:
            
            # 给监控器添加连接类型属性，用于健康报告
            monitor.connection_type = eport_client.connection_type
            
            print("✅ 所有组件初始化完成！")
            
            # 等待第一帧 - 确保视频流正常
            print("🔄 等待视频流就绪...")
            frame_info = rtsp_cap.get_frame_info()
            print(f"📺 视频流信息: {frame_info['width']}x{frame_info['height']} @ {frame_info['fps']:.1f}FPS")
            
            for i in range(30):  # 🔧 可以改：30是超时时间，单位秒
                frame = rtsp_cap.get_frame()
                if frame is not None:
                    print(f"✅ 视频流就绪！分辨率: {frame.shape[1]}x{frame.shape[0]}")
                    log.info(f"视频流详细信息: 分辨率={frame.shape[1]}x{frame.shape[0]}")
                    break
                time.sleep(1)  # 每秒检查一次
            else:
                log.critical("视频流初始化超时", immediate=True)
                raise RuntimeError("视频流初始化超时")
            
            print("🎉 系统初始化完成！开始实时检测...")
            if platform_cfg.has_gui:
                print("💡 控制命令:")
                print("   q : 退出程序")
                print("   r : 重新连接Eport") 
                print("   s : 显示系统状态")
                print("   c : 校准摄像头（待实现）")
                print("   p : 保存当前配置")
                print("   d : 切换检测显示")
                print("   f : 切换自适应帧率")
                print("   e : 切换图像增强")
                print("   t : 切换跟踪模式")
                print("   g : 切换云台跟踪")
                print("   l : 切换飞机跟踪")
                print("   i : 切换图像保存")
            else:
                print("💡 无GUI模式运行，使用 Ctrl+C 退出程序")
            print("=" * 50)
            
            detection_display_enabled = True
            adaptive_fps_enabled = config.get("adaptive_fps", True)
            image_enhancement_enabled = config.get("enable_image_enhancement", True)
            save_images_enabled = config.get("save_detection_images", False)
            
            # 主循环 - 不断处理视频帧
            while True:
                main_loop(trt_infer, eport_client, rtsp_cap, monitor, profiler, tracking_manager, image_saver)
                
                # 只在有GUI时处理键盘输入
                if platform_cfg.has_gui:
                    key = cv2.waitKey(1) & 0xFF  # 等待按键，1ms超时
                    if key == ord('q'):  # 按q退出
                        print("👋 用户请求退出")
                        break
                    elif key == ord('r'):  # 按r重连
                        print("🔄 手动重新连接Eport串口...")
                        eport_client._connect()
                        monitor.connection_type = eport_client.connection_type
                    elif key == ord('s'):  # 按s显示状态
                        monitor.report_health()
                    elif key == ord('c'):  # 按c校准（功能待实现)
                        print("🔧 摄像头校准功能待实现...")
                        print("我就装个b…")
                    elif key == ord('p'):  # 按p保存配置
                        config.save()
                    elif key == ord('d'):  # 按d切换检测显示
                        detection_display_enabled = not detection_display_enabled
                        config.config["enable_detection_display"] = detection_display_enabled
                        status = "开启" if detection_display_enabled else "关闭"
                        print(f"🔄 检测显示: {status}")
                    elif key == ord('f'):  # 按f切换自适应帧率
                        adaptive_fps_enabled = not adaptive_fps_enabled
                        config.config["adaptive_fps"] = adaptive_fps_enabled
                        monitor.fps_counter.adaptive_fps = adaptive_fps_enabled
                        status = "开启" if adaptive_fps_enabled else "关闭"
                        print(f"🔄 自适应帧率: {status}")
                    elif key == ord('e'):  # 按e切换图像增强
                        image_enhancement_enabled = not image_enhancement_enabled
                        config.config["enable_image_enhancement"] = image_enhancement_enabled
                        status = "开启" if image_enhancement_enabled else "关闭"
                        print(f"🔄 图像增强: {status}")
                    elif key == ord('t'):  # 按t切换跟踪模式
                        tracking_manager.enabled = not tracking_manager.enabled
                        status = "开启" if tracking_manager.enabled else "关闭"
                        print(f"🔄 目标跟踪: {status}")
                    elif key == ord('g'):  # 按g切换云台跟踪
                        tracking_manager.gimbal_tracking = not tracking_manager.gimbal_tracking
                        status = "开启" if tracking_manager.gimbal_tracking else "关闭"
                        print(f"🔄 云台跟踪: {status}")
                    elif key == ord('l'):  # 按l切换飞机跟踪
                        tracking_manager.flight_tracking = not tracking_manager.flight_tracking
                        status = "开启" if tracking_manager.flight_tracking else "关闭"
                        print(f"🔄 飞机跟踪: {status}")
                    elif key == ord('i'):  # 按i切换图像保存
                        save_images_enabled = not save_images_enabled
                        config.config["save_detection_images"] = save_images_enabled
                        image_saver.save_enabled = save_images_enabled
                        status = "开启" if save_images_enabled else "关闭"
                        print(f"🔄 图像保存: {status}")
                else:
                    # 无GUI模式下的退出检查 - 使用非阻塞输入
                    try:
                        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1)
                            if key == 'q':
                                print("👋 用户请求退出")
                                break
                    except Exception:
                        pass  # 忽略select异常
                    
    except KeyboardInterrupt:  # 按Ctrl+C
        print("\n👋 中断程序…")
    except Exception as e:  # 其他异常
        print(f"\n❌ 系统错误: {e}")
        print("💡 故障排除建议:")
        print("   1 : 检查Eport串口路径和波特率")
        print("   2 : 验证RTSP流地址有效性") 
        print("   3 : 确认TensorRT引擎文件（ARMv8适配）")
        print("   4 : 查看详细错误日志")
        # 同时记录完整错误到日志文件
        log.exception("系统主循环异常", immediate=False)
    finally:
        # 资源清理 - 确保程序退出前释放所有资源
        print("🔄 清理系统资源...")
        if platform_cfg.has_gui:
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
        monitor.report_health()  # 最终状态报告
        print("🎯 系统已安全退出")

# 开发者：纸张神探
# 开发者：韩言悦欣
# 哈哈哈哈哈哈哈哈哈哈哈哈哈哈老子写完了
# 都得给我跪下
#  ヾ(｡｀Д´｡)ﾉ彡

