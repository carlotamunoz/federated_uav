import flwr as fl
import numpy as np

# =====================================================
# METRICS (from the original code)
# =====================================================
def aggregate_metrics(metrics):
    accuracy_values = [m[1]["accuracy"] for m in metrics]
    avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
    return {"accuracy": avg_accuracy}


# =====================================================
# FLOWER SERVER STRATEGY
# =====================================================
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=aggregate_metrics
)

# =====================================================
# SERVER STARTUP
# =====================================================
if __name__ == "__main__":
    print("[SERVER] Starting Flower federated server on 0.0.0.0:8080")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024  # 512 MB
    )
