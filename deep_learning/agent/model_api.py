from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np
import tensorflow as tf 
from deep_learning.config import Config

class ChessAPI:
    """
    Allows one instance of the model to be waiting for game observations

    """
    def __init__(self, model):
        self.model = model
        self.pipes = []

    def start(self):
        prediction_worker = Thread(target=self._predict_batchdata_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def create_pipe(self):
        this, other = Pipe()
        self.pipes.append(this)
        return other

    def _predict_batchdata_worker(self):
        """
        Thread worker that waits for game observations and then outputs the predictions for p and v
        networks
        """
        while True:
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [],[]
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            policy_array, value_array = self.model.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_array, value_array):
                pipe.send((p, float(v)))
