from telemetry.telemetry_base import BaseTelemetry
import logging
import sys


class Console(BaseTelemetry):
    def __init__(self):
        super().__init__()
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        info_handler = logging.StreamHandler(stream=sys.stdout)
        info_handler.setFormatter(formatter)
        self.logger = logging.getLogger('anomaly-detector-info')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(info_handler)

        err_handler = logging.StreamHandler(stream=sys.stderr)
        err_handler.setFormatter(formatter)
        self.err_logger = logging.getLogger('anomaly-detector-err')
        self.err_logger.setLevel(logging.ERROR)
        self.err_logger.addHandler(err_handler)

    def count(self, name, int_val, **tags):
        self.logger.debug('count: %s : %d ' % (name + (str(tags) if tags else ''), int_val))

    def gauge(self, name, float_val, **tags):
        self.logger.debug('gauge: %s : %d ' % (name + (str(tags) if tags else ''), float_val))

    def duration(self, name, time_in_seconds, **tags):
        self.logger.debug('duration: %s : %d ms' % (name + (str(tags) if tags else ''), time_in_seconds * 1000))

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg + (str(args) if args else '') + (str(kwargs) if kwargs else ''))

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg + (str(args) if args else '') + (str(kwargs) if kwargs else ''))

    def error(self, msg, *args, **kwargs):
        self.err_logger.error(msg + (str(args) if args else '') + (str(kwargs) if kwargs else ''))

    def track_exception(self, traceback, properties):
        self.err_logger.error(traceback + (str(properties) if properties else ''))
