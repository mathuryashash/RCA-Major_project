import yaml
import os


def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file, then applies environment variable
    overrides (for Docker networking where service names differ from localhost).

    Supported env vars:
        RCA_DB_HOST, RCA_DB_PORT, RCA_DB_USER, RCA_DB_PASSWORD, RCA_DB_NAME
        RCA_REDIS_HOST, RCA_REDIS_PORT
        RCA_PROMETHEUS_URL
    """
    if not os.path.exists(config_path):
        # Handle fallback if running from src or other subdirectories
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(base_dir, "config", "config.yaml")

    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error in configuration file: {exc}")
            return None

    if config is None:
        return None

    # --- Environment variable overrides ---

    # Database (TimescaleDB)
    db = config.setdefault("database", {})
    if os.environ.get("RCA_DB_HOST"):
        db["host"] = os.environ["RCA_DB_HOST"]
    if os.environ.get("RCA_DB_PORT"):
        db["port"] = int(os.environ["RCA_DB_PORT"])
    if os.environ.get("RCA_DB_USER"):
        db["user"] = os.environ["RCA_DB_USER"]
    if os.environ.get("RCA_DB_PASSWORD"):
        db["password"] = os.environ["RCA_DB_PASSWORD"]
    if os.environ.get("RCA_DB_NAME"):
        db["dbname"] = os.environ["RCA_DB_NAME"]

    # Redis
    rd = config.setdefault("redis", {})
    if os.environ.get("RCA_REDIS_HOST"):
        rd["host"] = os.environ["RCA_REDIS_HOST"]
    if os.environ.get("RCA_REDIS_PORT"):
        rd["port"] = int(os.environ["RCA_REDIS_PORT"])

    # Prometheus
    ms = config.setdefault("metric_sources", {})
    if os.environ.get("RCA_PROMETHEUS_URL"):
        ms["prometheus_url"] = os.environ["RCA_PROMETHEUS_URL"]

    return config


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
