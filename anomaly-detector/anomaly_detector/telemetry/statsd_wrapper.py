import os
import statsd
import socket
from datetime import datetime
from requests import get
import traceback

DEFAULT_TIMEOUT = 0.01



def __dims__(tags, default_tags = None, logger = None):
    if default_tags is not None:
        for k, v in default_tags.items():
            if k not in tags:
                tags[k] = v
            elif logger is not None:
                logger.info(f"{k} is both in default_tags: {k}:{v} and tags: {k}:{tags[k]}")
    tmp_array = ['"' + x + '":"' + str(tags[x]) + '"' for x in tags.keys()]
    return "{" + ",".join(tmp_array) + "}"


class StatsdWrapper(object):

    def __init__(self, logger, host, port, mode):
        self.__logger = logger
        self.__stats_host = host
        self.__stats_port = port
        self.__stats_mode = mode
        self.__init_log_params()
        self.__connect__()

    def __init_log_params(self):
        DEFAULT_VALUE = "Unknown"
        self.log_namespace = os.environ.get("METRIC_NAMESPACE", "Api.AnomalyDetector")
        self.log_app = os.environ.get("KENSHO2_LOG_APP", "AnomalyDetectorApi")
        self.log_host = DEFAULT_VALUE
        self.log_service = DEFAULT_VALUE
        self.log_resourceGroup = DEFAULT_VALUE
        self.log_region = DEFAULT_VALUE
        self.log_deployment = DEFAULT_VALUE
        self.log_is_dr = os.environ.get("IS_DR_ENV", False)
        self.log_node = os.environ.get("KUBERNETES_NODE_NAME", DEFAULT_VALUE)
        self.log_profile = os.environ.get("KENSHO2_PROFILE", DEFAULT_VALUE)
        try:
            self.log_host = socket.gethostname()
            self.log_service = self.log_host
        except Exception as e:
            self.__logger.info(f"Get host name error: {traceback.format_exc()}")

        try:
            cluster_info = get("http://169.254.169.254/metadata/instance/compute?api-version=2020-12-01", headers = {"Metadata": "true"}).json()
            self.log_resourceGroup = cluster_info["resourceGroupName"]
            self.log_region = cluster_info["location"]
            self.log_deployment = "STAGING" if self.log_region in ["centraluseuap", "eastus2euap"] else "PROD"
        except Exception as e:
            self.__logger.info(f"Get resourceGroup, region or deployment error: {traceback.format_exc()}")
        ##get default tag for log
        self.default_tags = {
            "region": self.log_region,
            "resourceGroup": self.log_resourceGroup,
            "deployment": self.log_deployment,
            "is_dr": str(self.log_is_dr),
            "app": self.log_app,
            "host": self.log_host,
            "node": self.log_node,
            "profile": self.log_profile,
            "service": self.log_service
        }


    def __connect__(self):
        self.__stats = None
        if self.__stats_mode == 'tcp':
            # noinspection PyBroadException
            try:
                self.__stats = statsd.TCPStatsClient(
                    self.__stats_host, self.__stats_port, timeout=DEFAULT_TIMEOUT)
                self.__stats.connect()
            except Exception:
                self.__logger.track_exception(traceback=traceback.format_exc(), properties={})
        else:
            self.__stats = statsd.StatsClient(
                self.__stats_host, self.__stats_port)

    def timing(self, stat, delta, **tags):
        ts = datetime.now()
        str_datetime = ts.strftime('%Y-%m-%dT%H:%M:%S.%f0Z')
        metric_json = '{"Metric":"' + str(
            stat) + '","TS":"' + str_datetime + '","Namespace":"' + self.log_namespace + '","Dims":' + __dims__(tags, self.default_tags, self.__logger) + '}'

        # noinspection PyBroadException
        try:
            self.__stats.gauge(
                metric_json,
                value=delta,
                rate=1,
                delta=delta
            )
        except Exception:
            self.__logger.track_exception(traceback=traceback.format_exc(), properties={})
            self.__connect__()

    def gauge(self, stat, value, delta=True, **tags):
        ts = datetime.now()
        str_datetime = ts.strftime('%Y-%m-%dT%H:%M:%S.%f0Z')
        metric_json = '{"Metric":"' + stat + '","TS":"' + str_datetime + '","Namespace":"' + self.log_namespace + '","Dims":' \
                      + __dims__(tags, self.default_tags, self.__logger) + '}'
        # noinspection PyBroadException
        try:
            self.__stats.gauge(
                metric_json,
                value=value,
                rate=1,
                delta=delta
            )
        except Exception:
            self.__logger.track_exception(traceback=traceback.format_exc(), properties={})
            self.__connect__()
