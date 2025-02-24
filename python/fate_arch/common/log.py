#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import inspect
import traceback
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from threading import RLock

from fate_arch.common import file_utils

# Logger 基础方法，提供必要的基础能力

class LoggerFactory(object):
    TYPE = "FILE"
    LOG_FORMAT = "[%(levelname)s] [%(asctime)s] [jobId] [%(process)s:%(thread)s] - [%(module)s.%(funcName)s] [line:%(lineno)d]: %(message)s"
    LEVEL = logging.DEBUG
    logger_dict = {}
    global_handler_dict = {}

    LOG_DIR = None
    PARENT_LOG_DIR = None
    log_share = True

    append_to_parent_log = None

    lock = RLock()
    # CRITICAL = 50
    # FATAL = CRITICAL
    # ERROR = 40
    # WARNING = 30
    # WARN = WARNING
    # INFO = 20
    # DEBUG = 10
    # NOTSET = 0
    levels = (10, 20, 30, 40)
    schedule_logger_dict = {}

    @staticmethod
    def set_directory(directory=None, parent_log_dir=None, append_to_parent_log=None, force=False):
        if parent_log_dir:
            LoggerFactory.PARENT_LOG_DIR = parent_log_dir
        if append_to_parent_log:
            LoggerFactory.append_to_parent_log = append_to_parent_log
        with LoggerFactory.lock:
            if not directory:
                directory = file_utils.get_project_base_directory("logs")
            if not LoggerFactory.LOG_DIR or force:
                LoggerFactory.LOG_DIR = directory
            if LoggerFactory.log_share:
                oldmask = os.umask(000)
                os.makedirs(LoggerFactory.LOG_DIR, exist_ok=True)
                os.umask(oldmask)
            else:
                os.makedirs(LoggerFactory.LOG_DIR, exist_ok=True)
            for loggerName, ghandler in LoggerFactory.global_handler_dict.items():
                for className, (logger, handler) in LoggerFactory.logger_dict.items():
                    logger.removeHandler(ghandler)
                ghandler.close()
            LoggerFactory.global_handler_dict = {}
            for className, (logger, handler) in LoggerFactory.logger_dict.items():
                logger.removeHandler(handler)
                _handler = None
                if handler:
                    handler.close()
                if className != "default":
                    _handler = LoggerFactory.get_handler(className)
                    logger.addHandler(_handler)
                LoggerFactory.assemble_global_handler(logger)
                LoggerFactory.logger_dict[className] = logger, _handler

    # 创建 logger
    @staticmethod
    def new_logger(name):
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(LoggerFactory.LEVEL)
        return logger

    @staticmethod
    def get_logger(class_name=None):
        with LoggerFactory.lock:
            if class_name in LoggerFactory.logger_dict.keys():
                logger, handler = LoggerFactory.logger_dict[class_name]
                if not logger:
                    logger, handler = LoggerFactory.init_logger(class_name)
            else:
                logger, handler = LoggerFactory.init_logger(class_name)
            return logger

    @staticmethod
    def get_global_handler(logger_name, level=None, log_dir=None):
        if not LoggerFactory.LOG_DIR:
            return logging.StreamHandler()
        if log_dir:
            logger_name_key = logger_name + "_" + log_dir
        else:
            logger_name_key = logger_name + "_" + LoggerFactory.LOG_DIR
        # if loggerName not in LoggerFactory.globalHandlerDict:
        if logger_name_key not in LoggerFactory.global_handler_dict:
            with LoggerFactory.lock:
                if logger_name_key not in LoggerFactory.global_handler_dict:
                    handler = LoggerFactory.get_handler(logger_name, level, log_dir)
                    LoggerFactory.global_handler_dict[logger_name_key] = handler
        return LoggerFactory.global_handler_dict[logger_name_key]

    # 生成输出至文件的 handler，方便外层将日志输出至不同文件
    # 优先使用 log_type 进行文件命名，没有的情况下根据 class_name 进行文件命名，无法确定的情况下将日志输出至终端
    @staticmethod
    def get_handler(class_name, level=None, log_dir=None, log_type=None, job_id=None):
        #  优先根据 log_type 进行日志命名
        if not log_type:
            # 没办法确定输出目录，使用 StreamHandler 输出至终端
            if not LoggerFactory.LOG_DIR or not class_name:
                return logging.StreamHandler()

            # 没有 log_type 时，根据 class_name 进行文件命名
            if not log_dir:
                log_file = os.path.join(LoggerFactory.LOG_DIR, "{}.log".format(class_name))
            else:
                log_file = os.path.join(log_dir, "{}.log".format(class_name))
        else:
            log_file = os.path.join(log_dir, "fate_flow_{}.log".format(
                log_type) if level == LoggerFactory.LEVEL else 'fate_flow_{}_error.log'.format(log_type))

        # 包含 job_id 的情况下将 job_id 加入日志
        job_id = job_id or os.getenv("FATE_JOB_ID")
        if job_id:
            formatter = logging.Formatter(LoggerFactory.LOG_FORMAT.replace("jobId", job_id))
        else:
            formatter = logging.Formatter(LoggerFactory.LOG_FORMAT.replace("jobId", "Server"))

        # 最终使用 TimedRotatingFileHandler 实现按天的日志搜集
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if LoggerFactory.log_share:
            handler = ROpenHandler(log_file,
                                   when='D',
                                   interval=1,
                                   backupCount=14,
                                   delay=True)
        else:
            handler = TimedRotatingFileHandler(log_file,
                                               when='D',
                                               interval=1,
                                               backupCount=14,
                                               delay=True)

        if level:
            handler.level = level

        handler.setFormatter(formatter)
        return handler

    # 初始化 logger，创建 logger，并增加对应的 handler，输出日志至文件中
    @staticmethod
    def init_logger(class_name):
        with LoggerFactory.lock:
            logger = LoggerFactory.new_logger(class_name)
            handler = None
            if class_name:
                handler = LoggerFactory.get_handler(class_name)
                logger.addHandler(handler)
                LoggerFactory.logger_dict[class_name] = logger, handler

            else:
                LoggerFactory.logger_dict["default"] = logger, handler

            LoggerFactory.assemble_global_handler(logger)
            return logger, handler

    # 根据日志等级输出输出至特定文件，目前主要是输出至 DEBUG.log INFO.log WARNING.log ERROR.log
    @staticmethod
    def assemble_global_handler(logger):
        if LoggerFactory.LOG_DIR:
            for level in LoggerFactory.levels:
                if level >= LoggerFactory.LEVEL:
                    level_logger_name = logging._levelToName[level]
                    logger.addHandler(LoggerFactory.get_global_handler(level_logger_name, level))
        if LoggerFactory.append_to_parent_log and LoggerFactory.PARENT_LOG_DIR:
            for level in LoggerFactory.levels:
                if level >= LoggerFactory.LEVEL:
                    level_logger_name = logging._levelToName[level]
                    logger.addHandler(
                        LoggerFactory.get_global_handler(level_logger_name, level, LoggerFactory.PARENT_LOG_DIR))


def setDirectory(directory=None):
    LoggerFactory.set_directory(directory)


def setLevel(level):
    LoggerFactory.LEVEL = level


# 根据 className 初始化 logger
def getLogger(className=None, useLevelFile=False):
    if className is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        className = 'stat'
    return LoggerFactory.get_logger(className)


def exception_to_trace_string(ex):
    return "".join(traceback.TracebackException.from_exception(ex).format())


class ROpenHandler(TimedRotatingFileHandler):
    def _open(self):
        prevumask = os.umask(000)
        rtv = TimedRotatingFileHandler._open(self)
        os.umask(prevumask)
        return rtv
