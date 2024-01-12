import threading
from queue import Queue, Full, Empty
import socket
import ssl

DEFAULT_QUEUE_MAXSIZE = 100
DEFAULT_QUEUE_CIRCULAR = False
MAGIC_FLAG = 0x4B532D32.to_bytes(4, byteorder='big')
_TOMBSTONE = object()


class Mon3Sender(object):

    def __init__(self,
                 server_address,
                 queue_maxsize=DEFAULT_QUEUE_MAXSIZE,
                 queue_circular=DEFAULT_QUEUE_CIRCULAR):
        self._queue_maxsize = queue_maxsize
        self._queue_circular = queue_circular
        self._closed = False

        tokens = server_address.split(':')
        if len(tokens) == 3:
            protocol = tokens[0]
            if protocol == 'plain':
                self._secure = False
            elif protocol == 'ssl' or protocol == 'tls':
                self._secure = True
            else:
                raise Exception('Unknown protocol %s in logging address.' % protocol)
            self._addr = tokens[1]
            self._port = int(tokens[2])

        elif len(tokens) == 2:
            # For backwards compatibility
            self._addr = tokens[0]
            self._port = int(tokens[1])
            if self._port == 5201:
                self._port = 5202
                self._secure = True
            elif self._port == 5202:
                self._secure = True
            else:
                raise Exception('Do not support port %s if not specified the prefix "plain:" or "ssl:"' % self._port)
        else:
            raise Exception('Format error. too many colon in logging address.')

        self.socket = None
        self._queue = Queue(maxsize=queue_maxsize)
        # try to connect to mon3lib server
        self._connect()

        # send task
        self._send_thread = threading.Thread(target=self._send_loop,
                                             name="AsyncMon3Sender %d" % id(self))
        self.lock = threading.Lock()
        self._send_thread.daemon = True
        self._send_thread.start()

    def _connect(self):
        try:
            if self.socket is None:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                self.socket.connect((self._addr, self._port))
                if self._secure:
                    self.socket = ssl.wrap_socket(self.socket, ciphers="HIGH:-aNULL:-eNULL:-PSK:RC4-SHA:RC4-MD5",
                                                  ssl_version=ssl.PROTOCOL_TLSv1_2, cert_reqs=ssl.CERT_NONE)

                try:
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
                except OSError as ex:
                    print('Set socket option failed. %s', ex)

                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)
        except ConnectionError as ex:
            self._close()
            print("Socket connection error. %s" % ex)

    def send(self, content):
        with self.lock:
            if self._closed:
                return False
            if self._queue_circular and self._queue.full():
                # discard oldest
                try:
                    self._queue.get(block=False)
                except Empty:  # pragma: no cover
                    pass
            try:
                self._queue.put_nowait(content)
            except Full:  # pragma: no cover
                return False

            return True

    def close(self, flush=True):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            if not flush:
                while True:
                    try:
                        self._queue.get(block=False)
                    except Empty:
                        break
            self._queue.put(_TOMBSTONE)
            self._send_thread.join()

    def _send_data(self, data):
        self._connect()
        full_data = MAGIC_FLAG + (len(data)).to_bytes(4, byteorder='big') + data
        try:
            self.socket.sendall(full_data)
        except socket.error as ex:
            self._close()
            print("socket error encountered, trying to reconnect to the server. %s" % ex)
        except Exception as ex:
            self._close()
            print("Unexcepted exception raised. %s" % ex)

    def _send_loop(self):
        try:
            while True:
                try:
                    data = self._queue.get(block=True, timeout=10)
                except Empty:
                    data = b'' # Empty data treat as heartbeat.

                # close signal
                if data is _TOMBSTONE:
                    break
                self._send_data(data)

        finally:
            self._close()

    def _close(self):
        try:
            sock = self.socket
            if sock:
                try:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except socket.error:  # pragma: no cover
                        pass
                finally:
                    try:
                        sock.close()
                    except socket.error:  # pragma: no cover
                        pass
        finally:
            self.socket = None
