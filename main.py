from trainer import SparkConfig, Trainer
from logistic_regression import LogisticRegressionModel
from transforms import Transforms, Normalize

# Computed mean/std for Cancer Data
transforms = Transforms([
    Normalize(
        mean=[0]*30,
        std=[1]*30
    )
])

if __name__ == "__main__":
    spark_config = SparkConfig()
    spark_config.receivers = 4
    spark_config.batch_interval = 10

    rf = LogisticRegressionModel(maxIter=100, regParam=0.0)
    trainer = Trainer(rf, "train", spark_config, transforms)
    trainer.train()
    trainer.predict()