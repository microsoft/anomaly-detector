import os
import sys
import json
import time
import socket

TAG_NODE_VAL = os.environ.get("KUBERNETES_NODE_NAME")
internal_seq = 0

class Mon3Logger(object):
    def __init__(self, sender, app, service, profile):
        self.sender = sender
        self.service = service
        self.app = app
        self.profile = profile
        self.hostname = socket.gethostname()
        self.pid = os.getpid()

    def log_message(self, ts, level, msg, is_error):
        global internal_seq
        internal_seq = internal_seq + 1
        obj = {
            "ts": ts,
            "host": self.hostname,
            "service": self.service,
            "app": self.app,
            "profile": self.profile,
            "level": str(level),
            "msg": str(msg),
            "stack": '',
            "category": "none",
            "pid": self.pid,
            "seq": internal_seq
        }

        if TAG_NODE_VAL:
            obj["node"] = TAG_NODE_VAL

        prefix = "!" if is_error else "#"
        line = prefix + json.dumps(obj)

        try:
            self.sender.send(line.encode(encoding='utf-8'))
        except BaseException as ex:
            print(ex, file=sys.stderr)

    def log_info(self, msg):
        self.log_message(int(time.time() * 1000), "INFO", msg, False)

    def log_error(self, msg):
        self.log_message(int(time.time() * 1000), "ERROR", msg, True)

    def log_warn(self, msg):
        self.log_message(int(time.time() * 1000), "WARN", msg, False)
