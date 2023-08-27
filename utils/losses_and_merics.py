import tensorflow as tf

class EndPointError(tf.keras.metrics.Metric):
    """
    End point error metric.
    Calculates the average absolute difference
    between pixels in predicted disparity
    and groundtruth.

    """

    def __init__(self, name="EPE", **kwargs):
        super(EndPointError, self).__init__(name=name, **kwargs)
        self.end_point_error = self.add_weight(name='EPE', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        abs_errors = tf.abs(y_pred - y_true)
        # Valid map has all non-zero pixels set to 1 and 0 pixels remain 0
        valid_map = tf.where(
            tf.equal(y_true, 0),
            tf.zeros_like(y_true, dtype=tf.float32),
            tf.ones_like(y_true, dtype=tf.float32)
        )
        # Remove the errors with 0 groundtruth disparity
        filtered_error = abs_errors * valid_map
        # Get the mean error (non-zero groundtruth pixels)
        self.end_point_error.assign_add(
            tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
        )

    def result(self):
        return self.end_point_error

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.end_point_error.assign(0.0)

class Bad3(tf.keras.metrics.Metric):
    """
    Bad3 also called D1-all is the percentage
    of pixels with disparity difference >= 3
    between predicted disparity and groundtruth.

    """

    def __init__(self, name="Bad3(%)", **kwargs):
        super(Bad3, self).__init__(name=name, **kwargs)
        self.pixel_threshold = 3
        self.bad3 = self.add_weight(name='bad3_percent', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        abs_errors = tf.abs(y_pred - y_true)
        # Valid map has all non-zero pixels set to 1 and 0 pixels remain 0
        valid_map = tf.where(
            tf.equal(y_true, 0),
            tf.zeros_like(y_true, dtype=tf.float32),
            tf.ones_like(y_true, dtype=tf.float32)
        )
        # Remove the errors with 0 groundtruth disparity
        filtered_error = abs_errors * valid_map
        # 1 assigned to all errors greater than threshold, 0 to the rest
        bad_pixel_abs = tf.where(
            tf.greater(filtered_error, self.pixel_threshold),
            tf.ones_like(filtered_error, dtype=tf.float32),
            tf.zeros_like(filtered_error, dtype=tf.float32)
        )
        # (number of errors greater than threshold) / (number of errors)
        self.bad3.assign_add(tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map) * 100)

    def result(self):
        return self.bad3

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bad3.assign(0.0)
