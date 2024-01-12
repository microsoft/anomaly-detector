import os
import sys
import socket


TAG_NODE_VAL = os.environ.get("KUBERNETES_NODE_NAME")
tag_trans_table = str.maketrans({
    '\\': '\\ ',
    ' ': '\\ ',
    ',': '\\,',
    '=': '\\='
})


def escape_tag(val):
    if isinstance(val, str):
        s = val
    else:
        s = str(val)

    return s.translate(tag_trans_table)


class Mon3Meter(object):
    required_tag_line = ""

    def __init__(self, sender, app, service, profile):
        self.sender = sender
        self.service = service
        self.app = app
        self.profile = profile
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        if TAG_NODE_VAL:
            self.required_tag_line = "host=%s,service=%s,app=%s,profile=%s,node=%s" % (self.hostname, self.service, self.app, self.profile, TAG_NODE_VAL)
        else:
            self.required_tag_line = "host=%s,service=%s,app=%s,profile=%s" % (self.hostname, self.service, self.app, self.profile)

    def _send_perf(self, prefix, measurement, name, field_key, val, **tags):
        line = "+%s %s %s,metric=%s" % (
            prefix, measurement, self.required_tag_line, escape_tag(name))
        for k, v in tags.items():
            qk = escape_tag(k)
            qv = escape_tag(v)

            if qk is not None and qv is not None:
                line += ",%s=%s" % (qk, qv)

        line += " %s=%s" % (field_key, str(val))
        try:
            self.sender.send(line.encode(encoding='utf-8'))
        except Exception as ex:
            repr(ex)

    def count(self, name, int_val, **tags):
        # +counter name [tag1=val1[,tag2=val2]...] val
        try:
            val = int(int_val)
            if val < 0 or val >= 2 ** 63:
                print("The value is out of range [9223372036854775807, -9223372036854775808]", file=sys.stderr)
                return
        except Exception as ex:
            print(ex, file=sys.stderr)
            return

        self._send_perf("counter", "app_counter", name, "value", val, **tags)

    def gauge(self, name, float_val, **tags):
        # +gauge name [tag1=val1[,tag2=val2]...] val
        try:
            val = float(float_val)
        except Exception as ex:
            print(ex, file=sys.stderr)
            return

        self._send_perf("gauge", "app_gauge", name, "value", val, **tags)

    def gauge_str(self, name, str_val, **tags):
        # +gauge_s name [tag1=val1[,tag2=val2]...] val
        try:
            val = str(str_val)
        except Exception as ex:
            print(ex, file=sys.stderr)
            return

        self._send_perf("gauge-s", "app_gauge_str", name, "value", val, **tags)

    def duration(self, name, time_in_sec, **tags):
        # +timer name [tag1=val1[,tag2=val2]...] val
        try:
            val = int(time_in_sec * 1000 * 1000)
        except Exception as ex:
            print(ex, file=sys.stderr)
            return

        self._send_perf("timer", "app_duration", name, "duration", val, **tags)
