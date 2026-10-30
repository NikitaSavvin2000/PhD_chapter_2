import csv
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class CSVLogger:
    BASE_COLUMNS = ["time", "level", "message"]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.timezone = ZoneInfo("Europe/Moscow")

        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            self.columns = self.BASE_COLUMNS.copy()

            with self.file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
        else:
            with self.file_path.open("r", newline="", encoding="utf-8") as f:
                self.columns = next(csv.reader(f))

    def info(self, message: str, meta_info: dict):
        self._log("INFO", message, meta_info)

    def error(self, message: str, meta_info: dict):
        self._log("ERROR", message, meta_info)

    def _log(self, level: str, message: str, meta_info: dict):
        if not isinstance(meta_info, dict):
            raise TypeError("meta_info должен быть словарём")

        metadata = {}

        for key, value in meta_info.items():
            key = str(key)

            # Защита обязательных колонок
            if key in self.BASE_COLUMNS:
                key = f"meta_{key}"

            if key in metadata:
                raise ValueError(f"Дублирующийся ключ метаданных: {key}")

            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, default=str)

            metadata[key] = value

        row = {
            "time": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": str(message),
            **metadata,
        }

        with self.lock:
            new_columns = [
                key for key in metadata
                if key not in self.columns
            ]

            if new_columns:
                self._extend_columns(new_columns)

            with self.file_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writerow(row)

    def _extend_columns(self, new_columns: list):
        updated_columns = self.columns + new_columns
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    newline="",
                    encoding="utf-8",
                    dir=self.file_path.parent,
                    delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

                writer = csv.DictWriter(
                    temp_file,
                    fieldnames=updated_columns,
                )
                writer.writeheader()

                with self.file_path.open(
                        "r", newline="", encoding="utf-8"
                ) as source:
                    reader = csv.DictReader(source)
                    writer.writerows(reader)

            os.replace(temp_path, self.file_path)
            self.columns = updated_columns

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
