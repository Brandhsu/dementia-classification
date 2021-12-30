from tfcaidm import Jobs

YML_CONFIG = "/home/brandon/projects/dementia/config/pipeline.yml"
TRAIN_ROUTINE_PATH = "main.py"

Jobs(path=YML_CONFIG).setup(
    producer=__file__,
    consumer=TRAIN_ROUTINE_PATH,
).train_cluster(num_gpus=-1)
