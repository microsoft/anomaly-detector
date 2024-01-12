import os
from telemetry.telemetry_factory import TelemetryFactory

DEFAULT_TELEMETRY_TYPE = 'console'
telemetry_type = DEFAULT_TELEMETRY_TYPE


def validate_env(t_type, env_list):
    for env_item in env_list:
        if env_item not in os.environ:
            print("Missing mandatory environment variables %s for TelemetryType %s e.g: export %s=xxxxx "
                  % (env_item, t_type, env_item))
            os.abort()


def validate_telemetry_type():
    ###
    # verify env variables required by appinsights
    ###
    if telemetry_type == "appinsights":
        appinsights_env_list = ["APP_INSIGHTS_APK"]
        validate_env('appinsights', appinsights_env_list)

    ###
    # verify env variables required by Geneva
    ###
    if telemetry_type == "geneva":
        geneva_env_list = ["STATSD_HOST", "STATSD_PORT", "STATSD_MODE",
                           "FLUENTD_HOST", "FLUENTD_PORT"]
        validate_env('geneva', geneva_env_list)

    ###
    # verify env variables required by Geneva
    ###
    if os.environ['TELEMETRY_TYPE'] == "mon3":
        mon3_env_list = ["KENSHO2_PROFILE"]
        validate_env('mon3', mon3_env_list)


if 'TELEMETRY_TYPE' not in os.environ:
    os.environ.setdefault('TELEMETRY_TYPE', DEFAULT_TELEMETRY_TYPE)
    print("Telemetry type not provided,"
          "will use console logger."
          "You can set Telemetry Type to appinsights|geneva|mon3 "
          "e.g: export TELEMETRY_TYPE=geneva")
else:
    telemetry_type = str(os.environ['TELEMETRY_TYPE']).strip()

validate_telemetry_type()
log = TelemetryFactory.get_telemetry(telemetry_type)
