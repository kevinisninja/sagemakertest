import sagemaker
from sagemaker.tensorflow import TensorFlow


tf_estimator = TensorFlow(entry_point='tf-train.py', role='SageMakerRole',
                          training_steps=10000, evaluation_steps=100,
                          train_instance_count=1, train_instance_type='ml.p2.xlarge')
#tf_estimator.fit('s3://bucket/path/to/training/data')
