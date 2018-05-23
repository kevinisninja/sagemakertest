def model_fn(features, labels, mode, hyperparameters):
	# Connect the first hidden layer to input layer
	# (features["x"]) with relu activation
	first_hidden_layer = Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])

	# Connect the second hidden layer to first hidden layer with relu
	second_hidden_layer = Dense(20, activation='relu')(first_hidden_layer)

	# Connect the output layer to second hidden layer (no activation fn)
	output_layer = Dense(1, activation='linear')(second_hidden_layer)

	# Reshape output layer to 1-dim Tensor to return predictions
	predictions = tf.reshape(output_layer, [-1])

	# Provide an estimator spec for `ModeKeys.PREDICT`.
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions={"ages": predictions})

	# Calculate loss using mean squared error
	loss = tf.losses.mean_squared_error(labels, predictions)

	# Calculate root mean squared error as additional eval metric
	eval_metric_ops = {
		"rmse": tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float64), predictions)
	}

	optimizer = tf.train.GradientDescentOptimizer(
		learning_rate=hyperparameters["learning_rate"])
	train_op = optimizer.minimize(
		loss=loss, global_step=tf.train.get_global_step())

	# Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
	return tf.estimator.EstimatorSpec(
		mode=mode,
		loss=loss,
		train_op=train_op,
		eval_metric_ops=eval_metric_ops)

def train_input_fn(training_dir, hyperparameters):
	print('w')

def eval_input_fn(training_dir, hyperparameters):
	print('l')
