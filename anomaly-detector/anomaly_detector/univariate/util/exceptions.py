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
