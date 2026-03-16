from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


BASE_TRAFFIC_FEATURES = ["speed", "volume"]
BASE_WEATHER_CONTINUOUS = ["temp", "precipitation", "visibility", "wind_speed", "humidity", "pressure"]
WEATHER_BINARY_FLAGS = ["is_rain", "heavy_rain", "low_visibility", "freezing_risk"]
SUPPORTED_TARGETS = {"speed", "volume", "traffic_cost", "min_flow"}


@dataclass
class WindowedData:
    x: torch.Tensor  # [samples, time_in, nodes, features]
    y: torch.Tensor  # [samples, time_out, nodes, 1]
    timestamps: List[pd.Timestamp]
    sensor_ids: List[str]
    feature_names: List[str]


def _find_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _rename_weather_columns(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()

    ts_col = _find_first_existing(df.columns, ["timestamp", "datetime", "date_time", "time", "dt", "recorded_at", "date"])
    if ts_col is None:
        raise ValueError("Weather CSV needs a timestamp-like column.")

    sensor_col = _find_first_existing(df.columns, ["sensor_id", "station_id", "node_id", "site_id", "detector_id"])
    cluster_col = _find_first_existing(df.columns, ["cluster_id", "cluster", "group_id", "zone_id"])

    col_map = {ts_col: "timestamp"}

    alias_groups = {
        "temp": ["temp", "temperature", "air_temp", "t2m", "temperature_2m"],
        "precipitation": ["precipitation", "precip", "rain", "rain_1h", "rainfall", "prcp"],
        "visibility": ["visibility", "vis", "vis_km", "vis_miles"],
        "wind_speed": ["wind_speed", "wind", "wind_kph", "windspeed", "wind_speed_10m", "wind_speed_100m"],
        "humidity": ["humidity", "rh", "humid", "relative_humidity_2m"],
        "pressure": ["pressure", "pres", "sea_level_pressure", "surface_pressure"],
        "snow": ["snow", "snow_1h", "snowfall"],
        "weather_code": ["weather_code", "weather_id", "condition_code"],
    }

    found: Dict[str, str] = {}
    for std_col, aliases in alias_groups.items():
        c = _find_first_existing(df.columns, aliases)
        if c is not None:
            found[std_col] = c
            col_map[c] = std_col

    if "temp" not in found or "precipitation" not in found:
        raise ValueError("Weather CSV must include temp and precipitation columns.")

    if sensor_col is not None:
        col_map[sensor_col] = "sensor_id"
    if cluster_col is not None:
        col_map[cluster_col] = "cluster_id"

    df = df.rename(columns=col_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    if "sensor_id" in df.columns:
        df["sensor_id"] = df["sensor_id"].astype(str)
    if "cluster_id" in df.columns:
        df["cluster_id"] = df["cluster_id"].astype(str)

    keep_cols = ["timestamp"]
    if "sensor_id" in df.columns:
        keep_cols.append("sensor_id")
    if "cluster_id" in df.columns:
        keep_cols.append("cluster_id")

    weather_cols = [
        c
        for c in ["temp", "precipitation", "visibility", "wind_speed", "humidity", "pressure", "snow", "weather_code"]
        if c in df.columns
    ]
    keep_cols.extend(weather_cols)

    out = df[keep_cols].copy()
    # Ensure optional columns exist so downstream feature selection is stable.
    for optional_col in ["visibility", "wind_speed", "humidity", "pressure", "snow", "weather_code"]:
        if optional_col not in out.columns:
            out[optional_col] = np.nan

    return out


def _fit_stats(df: pd.DataFrame, cols: Sequence[str], method: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if method == "zscore":
        for c in cols:
            mean = float(df[c].mean())
            std = float(df[c].std(ddof=0))
            stats[c] = {"mean": mean, "std": 1.0 if std == 0.0 else std}
        return stats

    if method == "minmax":
        for c in cols:
            min_v = float(df[c].min())
            max_v = float(df[c].max())
            if max_v == min_v:
                max_v = min_v + 1.0
            stats[c] = {"min": min_v, "max": max_v}
        return stats

    raise ValueError("normalize_method must be one of: zscore, minmax")


def _apply_stats(df: pd.DataFrame, cols: Sequence[str], stats: Dict[str, Dict[str, float]], method: str) -> pd.DataFrame:
    out = df.copy()
    if method == "zscore":
        for c in cols:
            out[c] = (out[c] - stats[c]["mean"]) / stats[c]["std"]
        return out

    if method == "minmax":
        for c in cols:
            out[c] = (out[c] - stats[c]["min"]) / (stats[c]["max"] - stats[c]["min"])
        return out

    raise ValueError(f"Unsupported normalize method: {method}")


class TrafficDataPipeline:
    def __init__(
        self,
        input_len: int = 12,
        output_len: int = 12,
        normalize_method: str = "zscore",
        scaler_fit_ratio: float = 0.7,
        target_col: str = "speed",
        weather_strategy: str = "per_sensor",
        feature_set: str = "weather_plus_derived",
        rain_threshold: float = 0.3,
        low_visibility_threshold: float = 2.0,
        freezing_temp_threshold: float = 0.0,
        cluster_map_csv: Optional[str] = None,
    ) -> None:
        self.input_len = input_len
        self.output_len = output_len
        self.normalize_method = normalize_method
        self.scaler_fit_ratio = scaler_fit_ratio
        self.target_col = target_col

        self.weather_strategy = weather_strategy
        self.feature_set = feature_set
        self.rain_threshold = rain_threshold
        self.low_visibility_threshold = low_visibility_threshold
        self.freezing_temp_threshold = freezing_temp_threshold
        self.cluster_map_csv = cluster_map_csv

    def load_traffic(self, traffic_h5_path: str) -> pd.DataFrame:
        traffic_df = pd.read_hdf(traffic_h5_path)
        if not isinstance(traffic_df, pd.DataFrame):
            raise ValueError("Expected traffic .h5 to contain a pandas DataFrame")

        traffic_df.index = pd.to_datetime(traffic_df.index, utc=True, errors="coerce")
        traffic_df = traffic_df[~traffic_df.index.isna()].sort_index()

        long_df = (
            traffic_df.reset_index()
            .rename(columns={traffic_df.reset_index().columns[0]: "timestamp"})
            .melt(id_vars=["timestamp"], var_name="sensor_id", value_name="speed")
        )
        long_df["sensor_id"] = long_df["sensor_id"].astype(str)

        speed_max = float(np.nanmax(long_df["speed"].values))
        long_df["volume"] = np.clip(speed_max - long_df["speed"], 0.0, None)
        return long_df

    def load_sensor_distances(self, sensor_distances_csv: str) -> pd.DataFrame:
        dist_df = pd.read_csv(sensor_distances_csv)
        cols = [c.lower() for c in dist_df.columns]
        if not any(c in cols for c in ["from", "from_sensor", "source", "i"]):
            raise ValueError("sensor_distances.csv should include a source sensor column")
        if not any(c in cols for c in ["to", "to_sensor", "target", "j"]):
            raise ValueError("sensor_distances.csv should include a target sensor column")
        if not any(c in cols for c in ["distance", "dist", "cost", "weight"]):
            raise ValueError("sensor_distances.csv should include a distance column")
        return dist_df

    def load_cluster_map(self) -> Optional[pd.DataFrame]:
        if self.cluster_map_csv is None:
            return None
        cm = pd.read_csv(self.cluster_map_csv)
        sid = _find_first_existing(cm.columns, ["sensor_id", "node_id", "detector_id"])
        cid = _find_first_existing(cm.columns, ["cluster_id", "cluster", "group_id"])
        if sid is None or cid is None:
            raise ValueError("cluster_map_csv must include sensor_id and cluster_id columns")
        out = cm.rename(columns={sid: "sensor_id", cid: "cluster_id"})[["sensor_id", "cluster_id"]].copy()
        out["sensor_id"] = out["sensor_id"].astype(str)
        out["cluster_id"] = out["cluster_id"].astype(str)
        return out

    def load_weather(self, weather_csv_path: str) -> pd.DataFrame:
        weather_df = pd.read_csv(weather_csv_path)
        weather_df = _rename_weather_columns(weather_df)

        key_cols = ["timestamp"]
        if "sensor_id" in weather_df.columns:
            key_cols.append("sensor_id")
        if "cluster_id" in weather_df.columns:
            key_cols.append("cluster_id")

        numeric_cols = [c for c in weather_df.columns if c not in key_cols]
        for c in numeric_cols:
            weather_df[c] = pd.to_numeric(weather_df[c], errors="coerce")

        weather_df = weather_df.groupby(key_cols, as_index=False)[numeric_cols].mean().sort_values(key_cols)
        return weather_df

    def _resolve_weather_key(self, weather_df: pd.DataFrame) -> List[str]:
        if self.weather_strategy == "per_sensor":
            if "sensor_id" not in weather_df.columns:
                # model.py commonly exports global weather without sensor_id; allow safe fallback.
                return []
            return ["sensor_id"]

        if self.weather_strategy == "cluster":
            if "cluster_id" not in weather_df.columns:
                raise ValueError("Cluster weather strategy requires cluster_id in weather CSV")
            return ["cluster_id"]

        if self.weather_strategy == "global":
            return []

        raise ValueError("weather_strategy must be one of: per_sensor, cluster, global")

    def _resample_weather_to_traffic(
        self,
        weather_df: pd.DataFrame,
        traffic_timestamps: Sequence[pd.Timestamp],
        key_cols: List[str],
    ) -> pd.DataFrame:
        """Resample weather to exact traffic timestamps (5-minute typical)."""
        target_index = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(traffic_timestamps), utc=True).unique()))

        cont_cols = [c for c in BASE_WEATHER_CONTINUOUS if c in weather_df.columns]
        extra_cont = [c for c in ["snow"] if c in weather_df.columns]
        cont_cols = cont_cols + extra_cont
        cat_cols = [c for c in ["weather_code"] if c in weather_df.columns]

        if not key_cols:
            groups = [((), weather_df)]
        else:
            groups = list(weather_df.groupby(key_cols, sort=False))

        out_frames = []
        for key, g in groups:
            gi = g.sort_values("timestamp").set_index("timestamp")
            gi = gi[~gi.index.duplicated(keep="last")]

            re = gi.reindex(target_index)
            if cont_cols:
                re[cont_cols] = re[cont_cols].interpolate(method="time", limit_direction="both")
            if cat_cols:
                re[cat_cols] = re[cat_cols].ffill().bfill()

            re = re.reset_index().rename(columns={"index": "timestamp"})

            if key_cols:
                if len(key_cols) == 1:
                    value = key[0] if isinstance(key, tuple) else key
                    re[key_cols[0]] = value
                else:
                    for i, c in enumerate(key_cols):
                        re[c] = key[i]

            out_frames.append(re)

        out = pd.concat(out_frames, axis=0, ignore_index=True)
        return out

    def _merge_traffic_weather(
        self,
        traffic_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        cluster_map_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        df = traffic_df.copy()

        if self.weather_strategy == "cluster":
            if cluster_map_df is None:
                raise ValueError("Cluster strategy selected but no cluster_map_csv was provided")
            df = df.merge(cluster_map_df, on="sensor_id", how="left")
            if df["cluster_id"].isna().any():
                raise ValueError("Some sensor_ids are missing cluster_id mapping")

        key_cols = self._resolve_weather_key(weather_df)
        traffic_ts = sorted(pd.to_datetime(df["timestamp"].unique(), utc=True))
        weather_resampled = self._resample_weather_to_traffic(weather_df, traffic_ts, key_cols)

        if not key_cols:
            merged = df.merge(weather_resampled, on=["timestamp"], how="left")
        else:
            merged = df.merge(weather_resampled, on=["timestamp"] + key_cols, how="left")

        weather_cols = [c for c in BASE_WEATHER_CONTINUOUS if c in merged.columns] + [c for c in ["snow", "weather_code"] if c in merged.columns]

        merged = merged.sort_values(["sensor_id", "timestamp"]).copy()
        for col in weather_cols:
            merged[col] = (
                merged.groupby("sensor_id")[col]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both") if s.dtype.kind in "biufc" else s.ffill().bfill())
            )

        if weather_cols:
            merged[weather_cols] = merged[weather_cols].fillna(merged[weather_cols].median(numeric_only=True))
            fallback_values = {
                "temp": 20.0,
                "precipitation": 0.0,
                "visibility": 10.0,
                "wind_speed": 0.0,
                "humidity": 60.0,
                "pressure": 1013.0,
                "snow": 0.0,
                "weather_code": 0.0,
            }
            for c in weather_cols:
                if merged[c].isna().any():
                    merged[c] = merged[c].fillna(fallback_values.get(c, 0.0))

        return merged

    def _resolve_distance_column(self, dist_df: pd.DataFrame) -> str:
        col = _find_first_existing(dist_df.columns, ["distance", "dist", "cost", "weight"])
        if col is None:
            raise ValueError("sensor_distances.csv should include a distance-like column")
        return col

    def _compute_sensor_distance_bias(self, dist_df: pd.DataFrame) -> Dict[str, float]:
        src_col = _find_first_existing(dist_df.columns, ["from", "from_sensor", "source", "i"])
        dst_col = _find_first_existing(dist_df.columns, ["to", "to_sensor", "target", "j"])
        dist_col = self._resolve_distance_column(dist_df)

        if src_col is None or dst_col is None:
            return {}

        d = dist_df[[src_col, dst_col, dist_col]].rename(columns={src_col: "from", dst_col: "to", dist_col: "distance"})
        d["from"] = d["from"].astype(str)
        d["to"] = d["to"].astype(str)
        d["distance"] = pd.to_numeric(d["distance"], errors="coerce")
        d = d.dropna(subset=["distance"])
        d = d[(d["distance"] > 0) & (d["from"] != d["to"])]
        if d.empty:
            return {}

        grouped = d.groupby("from")["distance"].mean()
        fallback = float(grouped.mean()) if len(grouped) else 1.0
        return {str(k): float(v) for k, v in grouped.to_dict().items()} | {"__fallback__": fallback}

    def _add_derived_targets_and_flags(self, merged_df: pd.DataFrame, dist_df: pd.DataFrame) -> pd.DataFrame:
        df = merged_df.sort_values(["sensor_id", "timestamp"]).copy()

        df["min_flow"] = (
            df.groupby("sensor_id")["volume"]
            .transform(lambda s: s.rolling(window=self.input_len, min_periods=1).min())
            .astype(float)
        )

        distance_bias = self._compute_sensor_distance_bias(dist_df)
        fallback = float(distance_bias.get("__fallback__", 1.0))
        per_sensor_bias = df["sensor_id"].map(lambda sid: distance_bias.get(str(sid), fallback)).astype(float)
        speed_eps = np.clip(df["speed"].astype(float).values, 1.0, None)
        df["traffic_cost"] = per_sensor_bias.values / speed_eps

        rain = df["precipitation"].astype(float) if "precipitation" in df.columns else pd.Series(0.0, index=df.index)
        vis = df["visibility"].astype(float) if "visibility" in df.columns else pd.Series(10.0, index=df.index)
        temp = df["temp"].astype(float) if "temp" in df.columns else pd.Series(20.0, index=df.index)
        snow = df["snow"].astype(float) if "snow" in df.columns else pd.Series(0.0, index=df.index)

        df["is_rain"] = (rain > 0).astype(float)
        df["heavy_rain"] = (rain > self.rain_threshold).astype(float)
        df["low_visibility"] = (vis < self.low_visibility_threshold).astype(float)
        df["freezing_risk"] = ((temp <= self.freezing_temp_threshold) & ((rain > 0) | (snow > 0))).astype(float)

        return df

    def _resolve_feature_columns(self, merged_df: pd.DataFrame) -> List[str]:
        if self.feature_set not in {"traffic_only", "raw_weather", "weather_plus_derived"}:
            raise ValueError("feature_set must be one of: traffic_only, raw_weather, weather_plus_derived")

        features = BASE_TRAFFIC_FEATURES.copy()

        if self.feature_set in {"raw_weather", "weather_plus_derived"}:
            weather_cols = [c for c in BASE_WEATHER_CONTINUOUS if c in merged_df.columns]
            features.extend(weather_cols)

        if self.feature_set == "weather_plus_derived":
            flag_cols = [c for c in WEATHER_BINARY_FLAGS if c in merged_df.columns]
            features.extend(flag_cols)

        return features

    def normalize_features(
        self,
        merged_df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
        df = merged_df.sort_values(["timestamp", "sensor_id"]).copy()

        binary_cols = [c for c in WEATHER_BINARY_FLAGS if c in df.columns and c in feature_cols]
        continuous_cols = [c for c in feature_cols if c not in binary_cols]

        target_continuous = [self.target_col] if self.target_col not in binary_cols else []
        scale_cols = list(dict.fromkeys(continuous_cols + target_continuous))

        unique_ts = np.array(sorted(df["timestamp"].unique()))
        fit_cut_idx = max(1, int(len(unique_ts) * self.scaler_fit_ratio))
        fit_ts = set(unique_ts[:fit_cut_idx])

        fit_df = df[df["timestamp"].isin(fit_ts)]
        stats = _fit_stats(fit_df, scale_cols, self.normalize_method)
        scaled_df = _apply_stats(df, scale_cols, stats, self.normalize_method)

        scaler = {
            "method": self.normalize_method,
            "stats": stats,
            "scaled_columns": scale_cols,
            "binary_columns_not_scaled": binary_cols,
        }
        return scaled_df, scaler, binary_cols

    def build_feature_tensor(
        self,
        merged_df: pd.DataFrame,
        feature_names: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
        timestamps = sorted(pd.to_datetime(merged_df["timestamp"].unique(), utc=True))
        sensor_ids = sorted(merged_df["sensor_id"].astype(str).unique())

        ts_index = {ts: i for i, ts in enumerate(timestamps)}
        sensor_index = {sid: i for i, sid in enumerate(sensor_ids)}

        x_tnf = np.zeros((len(timestamps), len(sensor_ids), len(feature_names)), dtype=np.float32)
        y_tn = np.zeros((len(timestamps), len(sensor_ids)), dtype=np.float32)

        for row in merged_df.itertuples(index=False):
            t = ts_index[pd.Timestamp(row.timestamp)]
            n = sensor_index[str(row.sensor_id)]
            for fi, col in enumerate(feature_names):
                x_tnf[t, n, fi] = float(getattr(row, col))
            y_tn[t, n] = float(getattr(row, self.target_col))

        return x_tnf, y_tn, timestamps, sensor_ids

    def create_sliding_windows(self, x_tnf: np.ndarray, y_tn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total_t = x_tnf.shape[0]
        needed = self.input_len + self.output_len
        if total_t < needed:
            raise ValueError(f"Not enough timesteps ({total_t}) for windows of {self.input_len}+{self.output_len}.")

        xs, ys = [], []
        for start in range(0, total_t - needed + 1):
            mid = start + self.input_len
            end = mid + self.output_len
            xs.append(x_tnf[start:mid, :, :])
            ys.append(y_tn[mid:end, :][:, :, None])

        return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)

    def run(self, traffic_h5_path: str, sensor_distances_csv: str, weather_csv_path: str, output_dir: str) -> WindowedData:
        if self.target_col not in SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported target '{self.target_col}'. Choose one of {sorted(SUPPORTED_TARGETS)}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        traffic_df = self.load_traffic(traffic_h5_path)
        dist_df = self.load_sensor_distances(sensor_distances_csv)
        cluster_map_df = self.load_cluster_map()

        if self.feature_set == "traffic_only":
            merged_df = traffic_df.copy()
        else:
            weather_df = self.load_weather(weather_csv_path)
            merged_df = self._merge_traffic_weather(traffic_df, weather_df, cluster_map_df)

        merged_df = self._add_derived_targets_and_flags(merged_df, dist_df)

        feature_names = self._resolve_feature_columns(merged_df)
        merged_scaled_df, scaler, binary_cols = self.normalize_features(merged_df, feature_names)

        merged_df.to_csv(output_path / "merged_unscaled.csv", index=False)
        merged_scaled_df.to_csv(output_path / "merged_scaled.csv", index=False)

        x_tnf, y_tn, timestamps, sensor_ids = self.build_feature_tensor(merged_scaled_df, feature_names)
        x, y = self.create_sliding_windows(x_tnf, y_tn)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        torch.save(x_t, output_path / "x.pt")
        torch.save(y_t, output_path / "y.pt")

        metadata = {
            "input_len": self.input_len,
            "output_len": self.output_len,
            "num_samples": int(x_t.shape[0]),
            "num_timesteps": int(x_tnf.shape[0]),
            "num_nodes": int(x_tnf.shape[1]),
            "num_features": int(x_tnf.shape[2]),
            "feature_names": feature_names,
            "binary_feature_names": binary_cols,
            "sensor_ids": sensor_ids,
            "timestamps": [str(t) for t in timestamps],
            "scaler": scaler,
            "target": self.target_col,
            "feature_set": self.feature_set,
            "weather_strategy": self.weather_strategy,
            "tensor_shapes": {"x": list(x_t.shape), "y": list(y_t.shape)},
        }

        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return WindowedData(x=x_t, y=y_t, timestamps=timestamps, sensor_ids=sensor_ids, feature_names=feature_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather-aware data pipeline for STGCN")
    parser.add_argument("--traffic-h5", type=str, required=True)
    parser.add_argument("--sensor-distances", type=str, required=True)
    parser.add_argument("--weather-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed")

    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax"])
    parser.add_argument("--scaler-fit-ratio", type=float, default=0.7)

    parser.add_argument("--target", type=str, default="speed", choices=sorted(SUPPORTED_TARGETS))
    parser.add_argument("--weather-strategy", type=str, default="per_sensor", choices=["per_sensor", "cluster", "global"])
    parser.add_argument(
        "--feature-set",
        type=str,
        default="weather_plus_derived",
        choices=["traffic_only", "raw_weather", "weather_plus_derived"],
    )

    parser.add_argument("--cluster-map-csv", type=str, default=None)
    parser.add_argument("--rain-threshold", type=float, default=0.3)
    parser.add_argument("--low-visibility-threshold", type=float, default=2.0)
    parser.add_argument("--freezing-temp-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = TrafficDataPipeline(
        input_len=args.input_len,
        output_len=args.output_len,
        normalize_method=args.normalize,
        scaler_fit_ratio=args.scaler_fit_ratio,
        target_col=args.target,
        weather_strategy=args.weather_strategy,
        feature_set=args.feature_set,
        rain_threshold=args.rain_threshold,
        low_visibility_threshold=args.low_visibility_threshold,
        freezing_temp_threshold=args.freezing_temp_threshold,
        cluster_map_csv=args.cluster_map_csv,
    )

    result = pipeline.run(
        traffic_h5_path=args.traffic_h5,
        sensor_distances_csv=args.sensor_distances,
        weather_csv_path=args.weather_csv,
        output_dir=args.output_dir,
    )

    print("Data pipeline complete.")
    print(f"x shape: {tuple(result.x.shape)}")
    print(f"y shape: {tuple(result.y.shape)}")
    print(f"nodes: {len(result.sensor_ids)}")
    print(f"features: {result.feature_names}")
    print(f"target: {args.target}")
    print(f"feature_set: {args.feature_set}")
    print(f"weather_strategy: {args.weather_strategy}")


if __name__ == "__main__":
    main()
