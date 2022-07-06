import numpy as np
import tensorflow as tf


def customLoss(ytrue, ypred):
    return tf.math.reduce_mean((ypred[0] - ytrue["res0"]) ** 2) + tf.math.reduce_mean(
        (ypred[1] - ytrue["res1"]) ** 2
    )

def custom_metric(ytrue, ypred):
    return tf.math.count_nonzero(tf.math.abs(ytrue['res1'] - ypred[1]))

class CustomFitModel(tf.keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean("loss", dtype=tf.float32)
        self.metric_tracker = tf.keras.metrics.Sum("metric", dtype=tf.float32)

    # def compute_loss(self, x, y, y_pred, sample_weight):
        # loss = customLoss(y, y_pred)
        # print(self.metrics_names)
        # loss = self.losses(y, y_pred)
        # loss += tf.add_n(self.losses)
        # self.loss_tracker.update_state(loss)
        # return loss

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.metric_tracker.reset_states()

    def train_step(self, data):
        X, Y = data
        with tf.GradientTape() as tape:
            outputs = self(X, training=True)
            pred_loss = customLoss(Y, outputs)
            metric = custom_metric(Y, outputs)
        gradients = tape.gradient(pred_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(pred_loss)
        self.metric_tracker.update_state(metric)
        return {"loss": self.loss_tracker.result(), "metric" : self.metric_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metric_tracker]


inp = tf.keras.layers.Input((7, 11))
a = tf.keras.layers.Dense(3, name="res0")(inp)
b = tf.keras.layers.Dense(5, name="res1")(inp)

model = CustomFitModel([inp], [a, b])

model.summary(150)


# data = np.random.uniform(0, 1, (2, 7, 11))

# pred = model(data)

# print(pred)


dataX = np.random.uniform(0, 1, (2000, 7, 11))
dataY = {
    "res0": np.random.uniform(0, 1, (2000, 7, 3)),
    "res1": np.random.uniform(0, 1, (2000, 7, 5)),
}

model.compile(tf.keras.optimizers.SGD())

print(model.metrics_names)

model.fit(dataX, dataY, epochs=10)


# print(customLoss(dataY, pred))


#     "res0" : np.random.uniform(0, 1, (2, 7, 3)),
#     "res1" : np.random.uniform(0, 1, (2, 7, 5)),
# }


# dataY = [
#     #[
#         np.random.uniform(0, 1, (1, 7, 3)),
#         np.random.uniform(0, 1, (1, 7, 5)),
#     #],
#     #np.random.uniform(0, 1, (1,)),
# ]

# print(dataY)

# print(customLoss(dataY, pred))
