"""
scenario_registry.py -- Single source of truth for scenario definitions.

Reads from DuckDB (006_scenario_pipeline.sql) and exposes typed objects
consumed by:
  - build_event_dataset.py  (events, county_fips, output_key)
  - _validate_contract.py   (SCENARIO_EVENTS, OUTPUT_KEYS)
  - geo_qa.py               (bounding boxes)
  - FEATURE_CONTRACT.yaml   (scenarios: lists validated against registry)

Usage:
    from db.scripts.scenario_registry import ScenarioRegistry

    reg = ScenarioRegistry.from_db(con)          # from live DuckDB
    reg = ScenarioRegistry.from_schema_dir()     # standalone (no running DB)

    scenario = reg.get("houston")                # by pipeline name
    scenario = reg.by_id("harris_houston_urban") # by canonical ID

    # Validator integration
    output_keys = reg.output_keys()       # {pipeline_name: s3_key}
    scenario_events = reg.event_map()     # {pipeline_name: [s3_event_key, ...]}
    bounds = reg.bounds()                 # {pipeline_name: (lat_min, lat_max, lon_min, lon_max)}
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb


@dataclass
class EventDef:
    """A single storm event within a scenario."""
    event_name: str
    dr_number: Optional[int]
    storm_id: Optional[str]
    window_start: str
    window_end: str
    s3_event_key: str


@dataclass
class BoundingBox:
    """EPSG:4326 bounding box for spatial QA."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max
                and self.lon_min <= lon <= self.lon_max)


@dataclass
class ScenarioDef:
    """Complete scenario definition joining identity + pipeline config."""
    # From 001_scenarios
    scenario_id: str
    display_name: str
    anchor_storm: str
    region: str
    flood_archetype: str
    build_phase: str

    # From 006_scenario_pipeline_config
    pipeline_name: str
    output_s3_key: Optional[str]
    status: str
    bounds: Optional[BoundingBox]

    # From scenario_events
    events: list[EventDef] = field(default_factory=list)

    # From scenario_zctas (loaded on demand)
    county_fips: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def s3_event_keys(self) -> list[str]:
        return [e.s3_event_key for e in self.events]

    @property
    def event_names(self) -> list[str]:
        return [e.event_name for e in self.events]


class ScenarioRegistry:
    """Registry of all scenarios, loaded from DuckDB schema."""

    def __init__(self, scenarios: list[ScenarioDef]):
        self._by_pipeline = {s.pipeline_name: s for s in scenarios}
        self._by_id = {s.scenario_id: s for s in scenarios}

    def get(self, pipeline_name: str) -> ScenarioDef:
        """Look up by pipeline short name (houston, nyc, etc.)."""
        if pipeline_name not in self._by_pipeline:
            valid = sorted(self._by_pipeline.keys())
            raise KeyError(
                f"Unknown pipeline name '{pipeline_name}'. "
                f"Valid: {valid}"
            )
        return self._by_pipeline[pipeline_name]

    def by_id(self, scenario_id: str) -> ScenarioDef:
        """Look up by canonical scenario_id (harris_houston_urban, etc.)."""
        if scenario_id not in self._by_id:
            raise KeyError(f"Unknown scenario_id '{scenario_id}'")
        return self._by_id[scenario_id]

    def active(self) -> list[ScenarioDef]:
        """All scenarios with status='active'."""
        return [s for s in self._by_pipeline.values() if s.is_active]

    def all(self) -> list[ScenarioDef]:
        """All scenarios regardless of status."""
        return list(self._by_pipeline.values())

    def pipeline_names(self) -> list[str]:
        """All pipeline short names."""
        return sorted(self._by_pipeline.keys())

    def active_names(self) -> list[str]:
        """Pipeline names of active scenarios only."""
        return sorted(s.pipeline_name for s in self.active())

    # --- Convenience dicts for validator/builder integration ---

    def output_keys(self) -> dict[str, str]:
        """Map pipeline_name -> output S3 key (for L3 validator)."""
        return {
            s.pipeline_name: s.output_s3_key
            for s in self.active()
            if s.output_s3_key
        }

    def event_map(self) -> dict[str, list[str]]:
        """Map pipeline_name -> list of S3 event keys (for L1 validator)."""
        return {
            s.pipeline_name: s.s3_event_keys
            for s in self.active()
        }

    def bounds_map(self) -> dict[str, BoundingBox]:
        """Map pipeline_name -> bounding box (for geo_qa)."""
        return {
            s.pipeline_name: s.bounds
            for s in self.active()
            if s.bounds
        }

    def id_to_pipeline(self) -> dict[str, str]:
        """Map scenario_id -> pipeline_name."""
        return {s.scenario_id: s.pipeline_name for s in self.all()}

    def pipeline_to_id(self) -> dict[str, str]:
        """Map pipeline_name -> scenario_id."""
        return {s.pipeline_name: s.scenario_id for s in self.all()}

    # --- Constructors ---

    @classmethod
    def from_db(cls, con: duckdb.DuckDBPyConnection) -> "ScenarioRegistry":
        """Load from a live DuckDB connection (db_init already applied)."""
        rows = con.execute("""
            SELECT
                s.scenario_id, s.display_name, s.anchor_storm,
                s.region, s.flood_archetype, s.build_phase,
                p.pipeline_name, p.output_s3_key, p.status,
                p.lat_min, p.lat_max, p.lon_min, p.lon_max
            FROM scenarios s
            JOIN scenario_pipeline_config p USING (scenario_id)
        """).fetchall()

        scenarios = []
        for row in rows:
            (sid, dname, storm, region, archetype, phase,
             pname, out_key, status,
             lat_min, lat_max, lon_min, lon_max) = row

            bounds = None
            if all(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
                bounds = BoundingBox(lat_min, lat_max, lon_min, lon_max)

            # Load events
            events_raw = con.execute("""
                SELECT event_name, dr_number, storm_id,
                       window_start, window_end, s3_event_key
                FROM scenario_events
                WHERE scenario_id = ?
                ORDER BY window_start
            """, [sid]).fetchall()

            events = [
                EventDef(
                    event_name=e[0], dr_number=e[1], storm_id=e[2],
                    window_start=str(e[3]), window_end=str(e[4]),
                    s3_event_key=e[5] or e[0],
                )
                for e in events_raw
            ]

            # Load county FIPS from scenario_zctas
            fips_raw = con.execute("""
                SELECT DISTINCT county_fips
                FROM scenario_zctas
                WHERE scenario_id = ? AND county_fips IS NOT NULL
            """, [sid]).fetchall()
            county_fips = sorted(set(r[0] for r in fips_raw)) if fips_raw else []

            scenarios.append(ScenarioDef(
                scenario_id=sid, display_name=dname, anchor_storm=storm,
                region=region, flood_archetype=archetype, build_phase=phase,
                pipeline_name=pname, output_s3_key=out_key, status=status,
                bounds=bounds, events=events, county_fips=county_fips,
            ))

        return cls(scenarios)

    @classmethod
    def from_schema_dir(
        cls, schema_dir: Optional[Path] = None
    ) -> "ScenarioRegistry":
        """Load from schema SQL files directly (no running DB needed).

        Creates an in-memory DuckDB, applies all schema files, and reads
        the registry. Use this in pipeline scripts that don't have a
        persistent DB connection.
        """
        sd = schema_dir or Path(__file__).parent.parent / "schema"
        con = duckdb.connect(":memory:")
        for sql_file in sorted(sd.glob("*.sql")):
            con.execute(sql_file.read_text())
        return cls.from_db(con)


# --- Module-level convenience for one-liner imports ---

_CACHED: Optional[ScenarioRegistry] = None


def get_registry() -> ScenarioRegistry:
    """Get or create a cached registry (from schema files)."""
    global _CACHED
    if _CACHED is None:
        _CACHED = ScenarioRegistry.from_schema_dir()
    return _CACHED
