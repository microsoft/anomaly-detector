import os
from fluent import asyncsender as sender

DEFAULT_FLUENT_TAG = 'microsoft.cloudai.anomalydetector'
fluent_tag = os.environ.get('KENSHO2_LOG_FLUENTD_PREFIX', DEFAULT_FLUENT_TAG) 
log_app = os.environ.get("KENSHO2_LOG_APP", "AnomalyDetectorApi")

class FluentdWrapper(object):
    __logger = None

    def __init__(self, host, port):
        self.__logger = sender.FluentSender(
            fluent_tag, host=host, port=port, queue_maxsize=100000, queue_circular=True)
        self.addition_tag = {} if fluent_tag == DEFAULT_FLUENT_TAG else {"app": log_app}

    def info(self, msg, *args, **kwargs):
        base_log = {'level': 'INFO', 'message': msg}
        log_content = {**base_log, **kwargs, **self.addition_tag}
        self.__logger.emit('log', log_content)

    def warning(self, msg, *args, **kwargs):
        base_log = {'level': 'WARNING', 'message': msg}
        log_content = {**base_log, **kwargs, **self.addition_tag}
        self.__logger.emit('log', log_content)

    def error(self, msg, *args, **kwargs):
        base_log = {'level': 'ERROR', 'message': msg}
        log_content = {**base_log, **kwargs, **self.addition_tag}
        self.__logger.emit('log', log_content)

    def track_exception(self,  traceback, properties):
        base_log = {'level': 'EXCEPTION', 'traceback': traceback}
        log_content = {**base_log, **properties, **self.addition_tag}
        self.__logger.emit('log', log_content)
