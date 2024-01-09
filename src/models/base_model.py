import pandas as pd

from constant import ONSET_DURATION, RECORD_DELAY, SAMPLE_RATE


class BaseModel:
    def __init__(
        self,
        data_file,
        training_epochs,
        opt_learning_rate,
        batch_size,
    ):
        self.df = pd.read_csv(data_file)  # load data
        self.training_epochs = training_epochs
        self.opt_learning_rate = opt_learning_rate
        self.batch_size = batch_size
        self.sample_rate = SAMPLE_RATE
        self.onset_duration = ONSET_DURATION
        self.record_delay = RECORD_DELAY
        self.x_train = None
        self.y_tarin = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def input_reshape(self, data):
        # Implement input reshaping logic
        pass

    def create_dataset(self):
        # Implement dataset split feature & label logic
        pass

    def print_dataset_shape(self):
        print("x_train : ", self.x_train.shape)
        print("y_train : ", self.y_train.shape)
        print("x_val : ", self.x_val.shape)
        print("y_val : ", self.y_val.shape)
        print("x_test : ", self.x_test.shape)
        print("y_test : ", self.y_test.shape)

    def create(self):
        # Implement model creation logic
        pass

    def train(self):
        # Implement model train logic
        pass

    def evaluate(self, x_test, y_test):
        # Implement model evaluation logic
        pass

    def predict(self, data):
        # Implement model predict logic
        pass
