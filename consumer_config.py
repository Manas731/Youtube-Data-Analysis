
import os
import yaml
from typing import Dict, Any
from confluent_kafka import Consumer


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_kafka_consumer(
        config: Dict[str, Any],
        group_id: str = "youtube-sentiment-dashboard-group",
        offset_reset: str = "latest",
        skip_hostname_verify: bool = False
) -> Consumer:

    auth = config.get("auth", {})
    method = auth.get("method", "sasl").lower()

    # env-var overrides
    auth["username"] = os.getenv("KAFKA_USERNAME", auth.get("username"))
    auth["password"] = os.getenv("KAFKA_PASSWORD", auth.get("password"))
    auth["ca_path"]  = os.getenv("KAFKA_CA_PATH",  auth.get("ca_path"))

    consumer_conf: Dict[str, Any] = {
        "bootstrap.servers": config["bootstrap_server"],
        "group.id": group_id,
        "auto.offset.reset": offset_reset,
    }

    if method == "sasl":
        consumer_conf.update({
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "SCRAM-SHA-512",
            "sasl.username": auth["username"],
            "sasl.password": auth["password"],
            "ssl.ca.location": auth["ca_path"],
        })
        if skip_hostname_verify:
            consumer_conf["ssl.endpoint.identification.algorithm"] = ""

    else:
        raise ValueError(f"Unsupported auth method: {method}")

    consumer = Consumer(consumer_conf)
    consumer.subscribe([config["topic"]])
    return consumer
