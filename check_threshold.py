import mlflow
from mlflow.tracking import MlflowClient

THRESHOLD = 0.85

# mlflow.set_tracking_uri("file:./mlruns")  # Removed to respect environment variable

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

print(f"Accuracy: {accuracy}")

if accuracy is None or accuracy < THRESHOLD:
    raise Exception("Model failed threshold ")

print("Model passed ")