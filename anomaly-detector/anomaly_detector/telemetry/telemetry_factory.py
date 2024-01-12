import os


class TelemetryFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_telemetry(telemetry_type):
        if type(telemetry_type) != str:
            return None
        if telemetry_type.lower() == "appinsights":
            from telemetry.appinsights import ApplicationInsights
            log_config = {
                'APP_INSIGHTS_APK': os.environ['APP_INSIGHTS_APK'] if "APP_INSIGHTS_APK" in os.environ else "",
            }
            return ApplicationInsights(app_insights_apk=log_config['APP_INSIGHTS_APK'])
        if telemetry_type.lower() == "geneva":
            from telemetry.geneva import Geneva
            log_config = {
                'STATSD_HOST': os.environ['STATSD_HOST'] if "STATSD_HOST" in os.environ else "localhost",
                'STATSD_PORT': int(os.environ['STATSD_PORT']) if "STATSD_PORT" in os.environ else 8125,
                'STATSD_MODE': os.environ['STATSD_MODE'] if "STATSD_MODE" in os.environ else "tcp",
                'FLUENTD_HOST': os.environ['FLUENTD_HOST'] if "FLUENTD_HOST" in os.environ else "localhost",
                'FLUENTD_PORT': int(os.environ['FLUENTD_PORT']) if "FLUENTD_PORT" in os.environ else 24224
            }
            return Geneva(statsd_host=log_config['STATSD_HOST'],
                          statsd_port=log_config['STATSD_PORT'],
                          statsd_mode=log_config['STATSD_MODE'],
                          fluentd_host=log_config['FLUENTD_HOST'],
                          fluentd_port=log_config['FLUENTD_PORT'])
        if telemetry_type.lower() == "mon3":
            from telemetry.mon3 import Mon3
            if os.environ.get('USE_GENEVA', '').lower() == 'true':
                # Logging to geneva via mon-collectd in MetricsAdvisor system.
                mon3_server = 'plain:%s:%s' % (os.environ['NODE_IP'], os.environ['KENSHO2_LOG_PORT'])
            elif 'KENSHO2_CONFIG_DIR' in os.environ:
                # Logging to dev by specified endpoint in MetricsAdvisor system.
                # Load endpoint from config server.
                confpath = os.environ['KENSHO2_CONFIG_DIR'] + "/endpoints.ini"
                import requests
                # Read endpoints.ini file from config endpoint
                if confpath.startswith('file://'):
                    with open(confpath[7:], 'r') as f:
                        inistr = f.read()
                elif confpath.startswith('http://') or confpath.startswith('https://'):
                    r = requests.get(confpath)
                    if r.status_code != requests.codes.ok:
                        raise Exception('Configure file "{}" fetch failed, statuscode: {}, message: {}'.format(confpath, r.status_code, r.content))
                    inistr = r.text
                else:
                    raise Exception('Configure file protocal not supported: {}'.format(confpath))

                import configparser
                iniconfig = configparser.RawConfigParser()
                iniconfig.read_string(inistr)
                mon3_server = iniconfig['logging']['endpoint-secure']
            else:
                # Otherwise use MON3_SERVER
                mon3_server = os.environ.get('MON3_SERVER', 'localhost')

            log_config = {
                'MON3_SERVER': mon3_server,
                'KENSHO2_PROFILE': os.environ.get('KENSHO2_PROFILE', 'INT'),
                'MON3_APP': os.environ.get('MON3_APP', os.environ.get('KENSHO2_LOG_APP', 'telemetry')),
                'MON3_SERVICE': os.environ.get('MON3_SERVICE', os.environ.get('KENSHO2_LOG_SERVICE', 'telemetry'))
            }
            return Mon3(mon3_server=log_config['MON3_SERVER'],
                        kensho2_profile=log_config['KENSHO2_PROFILE'],
                        app=log_config['MON3_APP'],
                        service=log_config['MON3_SERVICE']
                        )
        if telemetry_type.lower() == 'console':
            from telemetry.console import Console
            return Console()
        else:
            return None
