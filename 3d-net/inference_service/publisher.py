import logging

import zmq

from .models import FrameResult

logger = logging.getLogger(__name__)


class DetectionPublisher:
    """
    Publishes FrameResult objects on a ZeroMQ PUB socket.

    Message format (multipart, mirrors MetadataPublisher):
        frame 0: b"detections"          topic bytes
        frame 1: FrameResult JSON UTF-8
    """

    TOPIC = b"detections"

    def __init__(self, endpoint: str = "tcp://*:5556"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(endpoint)
        logger.info("ZMQ PUB bound to %s", endpoint)

    def publish(self, result: FrameResult) -> None:
        payload = result.model_dump_json().encode("utf-8")
        self._socket.send_multipart([self.TOPIC, payload])

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
