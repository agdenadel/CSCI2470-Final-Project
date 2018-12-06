import tensorflow as tf

LEARNING_RATE = 0.001
BATCH_SIZE = 100

class Model:
    def __init__(self):
        

        # computation graph
        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self. accuracy = self.accuracy_function()


    def forward_pass(self):
        pass


    def loss_function(self):
        pass


    def optimizer(self):
        return tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)


    def accuracy_function(self):
        pass

