from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from typing import Dict, List
from datetime import datetime

from config import INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET

logger = logging.getLogger(__name__)

class InfluxDBManager:
    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = INFLUXDB_BUCKET
        self.org = INFLUXDB_ORG

    def write_data(self, data_point: Dict):
        """Write a single data point to InfluxDB."""
        try:
            point = Point(data_point["measurement"]) \
                .time(data_point["time"], WritePrecision.MS) \
                .tag("market", data_point["tags"]["market"])

            for field_name, field_value in data_point["fields"].items():
                try:
                    # Convert numeric values to float to ensure consistent types
                    if isinstance(field_value, (int, float)):
                        point = point.field(field_name, float(field_value))
                    else:
                        point = point.field(field_name, str(field_value))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting field {field_name} with value {field_value}: {str(e)}")
                    continue

            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            logger.error(f"Error writing to InfluxDB: {e}")
            return False

    def write_batch(self, data_points: List[Dict]):
        """Write multiple data points to InfluxDB."""
        try:
            points = []
            for data_point in data_points:
                point = Point(data_point["measurement"]) \
                    .time(data_point["time"]) \
                    .tag("market", data_point["tags"][" market"])

                for field_name, field_value in data_point["fields"].items():
                    if isinstance(field_value, (int, float)):
                        point = point.field(field_name, field_value)
                    else:
                        point = point.field(field_name, str(field_value))
                points.append(point)

            self.write_api.write(bucket=self.bucket, org=INFLUXDB_ORG, record=points)
            return True
        except Exception as e:
            logger.error(f"Error writing batch to InfluxDB: {e}")
            return False

    def close(self):
        """Close the InfluxDB client connection."""
        self.client.close() 