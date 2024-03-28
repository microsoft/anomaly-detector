# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

class AnomalyDetectorException(Exception):
    def __init__(self, code="", message=""):
        if code == "":
            self.code = self.__class__.__name__
        else:
            self.code = code
        self.message = message.replace("'", '"')

    def __str__(self):
        return f"error_code={self.code}, error_message={self.message}"

    def __repr__(self):
        return f"{self.__class__.__name__}('code={self.code}', message={self.message})"

    def to_dict(self):
        return {"code": self.code, "message": self.message}

class AnomalyDetectionRequestError(Exception):
    """Raised when an error occurs in the request."""
    def __init__(self, error_msg, error_code=None):
        if isinstance(error_msg, type(b'')):
            error_msg = error_msg.decode('UTF-8', 'replace')

        super(AnomalyDetectionRequestError, self).__init__(
            error_msg
        )
        self.message = error_msg
        self.code = error_code

class DataFormatError(AnomalyDetectorException):
    pass


class InvalidParameterError(AnomalyDetectorException):
    pass
