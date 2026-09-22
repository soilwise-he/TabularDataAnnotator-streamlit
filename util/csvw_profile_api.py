"""Convert DataAnnotator metadata to SOSA-aware CSVW JSON.

The module is dependency-free and can be called directly from Streamlit::

    from util.csvw_profile_api import build_csvw_document
    csvw_json = build_csvw_document(payload)

``create_fastapi_app`` is an optional adapter for a future HTTP deployment;
it imports FastAPI only when that adapter is used.  All exporter logic remains
stateless and independent of Streamlit session state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


SOSA_CONTEXT = [
    "http://www.w3.org/ns/csvw",
    {
        "sosa": "http://www.w3.org/ns/sosa/",
        "qudt": "http://qudt.org/1.1/schema/qudt#",
        "dcterms": "http://purl.org/dc/terms/",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "schema": "https://schema.org/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    },
]

SPATIAL_ROLE_TO_PROPERTY = {
    "X": "geo:long",
    "Y": "geo:lat",
    "Z": "geo:alt",
    "WKT geometry": "geo:asWKT",
    "BBOX": "geo:asWKT",
}

FOI_BUCKET = "Feature of Interest (FOI) - ID"
SPATIAL_BUCKET = "FOI - Spatial Information"
ATTRIBUTE_BUCKET = "FOI - Attribute"
OBSERVATION_BUCKET = "Observed Property"
TEMPORAL_BUCKET = "Temporal"
UNSORTED_BUCKET = "Unsorted"

CONCEPT_TYPE_TO_BUCKET = {
    "sosa:FeatureOfInterest": FOI_BUCKET,
    "sosa:Property": OBSERVATION_BUCKET,
    "geo:Feature": SPATIAL_BUCKET,
    "schema:Property": ATTRIBUTE_BUCKET,
    "sosa:phenomenonTime": TEMPORAL_BUCKET,
    "sosa:resultTime": TEMPORAL_BUCKET,
}


@dataclass
class CSVWExportRequest:
    """JSON accepted by the CSVW exporter.

    ``metadata_by_table`` is the app's ``metadata_df`` represented as JSON:
    each table key maps to an array of metadata records.  ``column_buckets``
    uses the bucket names created on the Column Sorting page.
    """

    metadata_by_table: dict[str, list[dict[str, Any]]]
    base_url: str
    filename_dict: dict[str, str] = field(default_factory=dict)
    column_buckets: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    temporal_deepdive: dict[str, dict[str, Any]] = field(default_factory=dict)
    spatial_types: dict[str, dict[str, str]] = field(default_factory=dict)
    spatial_fit_for_all: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CSVWExportRequest":
        """Create a request object from JSON-compatible app state."""
        if not isinstance(payload, dict):
            raise ValueError("The CSVW export payload must be a JSON object.")
        metadata_by_table = payload.get("metadata_by_table")
        base_url = payload.get("base_url")
        if not isinstance(metadata_by_table, dict):
            raise ValueError("metadata_by_table must be an object keyed by table name.")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be a string.")

        def mapping(name: str) -> dict[str, Any]:
            value = payload.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be an object.")
            return value

        relationships = payload.get("relationships", [])
        if not isinstance(relationships, list):
            raise ValueError("relationships must be an array.")
        for table_key, rows in metadata_by_table.items():
            if not isinstance(table_key, str) or not isinstance(rows, list):
                raise ValueError("Each metadata_by_table value must be an array of records.")
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError(f"Metadata rows for '{table_key}' must be objects.")

        return cls(
            metadata_by_table=metadata_by_table,
            base_url=base_url,
            filename_dict=mapping("filename_dict"),
            column_buckets=mapping("column_buckets"),
            temporal_deepdive=mapping("temporal_deepdive"),
            spatial_types=mapping("spatial_types"),
            spatial_fit_for_all=mapping("spatial_fit_for_all"),
            relationships=relationships,
        )


def _text(value: Any) -> str:
    """Return a stripped string, treating null-like values as absent."""
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "null", "none", "nan"} else value


def _metadata_rows(rows: list[dict[str, Any]], table_key: str) -> dict[str, dict[str, Any]]:
    """Normalize legacy field names and index metadata records by column name."""
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if not _text(row.get("column_type")):
            row["column_type"] = row.get("datatype", "")
        if not _text(row.get("column_format")):
            row["column_format"] = row.get("dateTime format", "")
        if not _text(row.get("concept_uri")):
            row["concept_uri"] = row.get("element_uri", "")

        name = _text(row.get("name"))
        if not name:
            raise ValueError(f"Table '{table_key}' has a metadata record without a name.")
        if name in indexed:
            raise ValueError(f"Table '{table_key}' has duplicate metadata for column '{name}'.")
        indexed[name] = row
    return indexed


def _is_true(value: Any) -> bool:
    """Interpret JSON and CSV boolean values without treating text as truthy."""
    return value is True or _text(value).lower() in {"true", "1", "yes"}


def _effective_buckets(
    metadata: dict[str, dict[str, Any]],
    saved_buckets: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Derive CSVW roles from persisted metadata, using UI buckets as a fallback.

    ``concept_type`` is part of the metadata dataframe.  It is therefore authoritative when present.
    The transient Column Sorting session state is retained only for unannotated
    columns and for role detail not stored in the dataframe.
    """
    buckets = {bucket: [] for bucket in (*CONCEPT_TYPE_TO_BUCKET.values(), UNSORTED_BUCKET)}
    assigned: set[str] = set()

    for column_name, row in metadata.items():
        bucket = CONCEPT_TYPE_TO_BUCKET.get(_text(row.get("concept_type")))
        if bucket:
            buckets[bucket].append(column_name)
            assigned.add(column_name)

    # A declared primary key is a useful persisted fallback when FOI semantics
    # have not yet been annotated. Do not guess if several columns claim it.
    if not buckets[FOI_BUCKET]:
        primary_keys = [
            column_name
            for column_name, row in metadata.items()
            if _is_true(row.get("primary_key"))
        ]
        if len(primary_keys) == 1:
            buckets[FOI_BUCKET].append(primary_keys[0])
            assigned.add(primary_keys[0])

    for bucket, column_names in saved_buckets.items():
        if not isinstance(column_names, list):
            continue
        target = buckets.setdefault(bucket, [])
        for column_name in column_names:
            if column_name in metadata and column_name not in assigned and column_name not in target:
                target.append(column_name)
                assigned.add(column_name)

    return buckets


def _node(base_url: str, foi_column: str | None, suffix: str = "") -> str:
    identity = f"{{{foi_column}}}" if foi_column else "{_row}"
    return f"{base_url}{identity}{suffix}"


def _datatype(row: dict[str, Any], fallback: str = "string") -> str | dict[str, str]:
    datatype = _text(row.get("column_type")) or fallback
    date_format = _text(row.get("column_format"))
    if datatype in {"date", "dateTime", "time"} and date_format:
        return {"base": datatype, "format": date_format}
    return datatype


def _column_description(column: dict[str, Any], row: dict[str, Any]) -> None:
    title = _text(row.get("element")) or _text(row.get("concept"))
    if title:
        column["titles"] = title
    description = _text(row.get("description"))
    if description:
        column["dc:description"] = description


def _parse_fk_targets(value: Any) -> list[str]:
    text = _text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return [_text(target) for target in parsed if _text(target)] if isinstance(parsed, list) else []
    return [text]


def _foreign_keys(
    table_key: str,
    metadata: dict[str, dict[str, Any]],
    filenames: dict[str, str],
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build CSVW foreign keys from app relationships and persisted ``fk_target`` values."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(local: Any, target_table: Any, target_column: Any) -> None:
        local_name = _text(local)
        remote_table = _text(target_table)
        remote_column = _text(target_column)
        if not all((local_name, remote_table, remote_column)):
            return
        key = (local_name, remote_table, remote_column)
        if key in seen:
            return
        seen.add(key)
        result.append({
            "columnReference": local_name,
            "reference": {
                "resource": filenames.get(remote_table, f"{remote_table}.csv"),
                "columnReference": remote_column,
            },
        })

    for column_name, row in metadata.items():
        for target in _parse_fk_targets(row.get("fk_target")):
            if "." in target:
                target_table, target_column = target.split(".", 1)
                add(column_name, target_table, target_column)

    for relationship in relationships:
        if _text(relationship.get("left_table")) != table_key:
            continue
        if _text(relationship.get("relation")) == "not-linked":
            continue
        add(relationship.get("left_id"), relationship.get("right_table"), relationship.get("right_id"))
    return result


def _virtual_observation_columns(
    observation_column: str,
    foi_column: str | None,
    base_url: str,
    row: dict[str, Any],
    result_time_default: str | None,
) -> list[dict[str, Any]]:
    observation = _node(base_url, foi_column, f"/{observation_column}")
    quantity_value = f"{observation}/QV"
    columns = [
        {"virtual": True, "aboutUrl": observation, "valueUrl": "sosa:Observation", "propertyUrl": "rdf:type"},
        {"virtual": True, "aboutUrl": observation, "valueUrl": _node(base_url, foi_column), "propertyUrl": "sosa:hasFeatureOfInterest"},
        {"virtual": True, "aboutUrl": observation, "valueUrl": quantity_value, "propertyUrl": "sosa:hasResult"},
        {"virtual": True, "aboutUrl": quantity_value, "valueUrl": "qudt:QuantityValue", "propertyUrl": "rdf:type"},
    ]
    concept_uri = _text(row.get("concept_uri"))
    if concept_uri:
        columns.append({"virtual": True, "aboutUrl": observation, "valueUrl": concept_uri, "propertyUrl": "sosa:observedProperty"})
    method_uri = _text(row.get("method_uri"))
    if method_uri:
        columns.append({"virtual": True, "aboutUrl": observation, "valueUrl": method_uri, "propertyUrl": "sosa:usedProcedure"})
    if result_time_default is not None:
        time_column: dict[str, Any] = {
            "virtual": True,
            "aboutUrl": observation,
            "propertyUrl": "sosa:resultTime",
            "datatype": "dateTime",
        }
        if result_time_default:
            time_column["default"] = result_time_default
        columns.append(time_column)
    unit_uri = _text(row.get("unit_uri"))
    if unit_uri:
        columns.append({"virtual": True, "aboutUrl": quantity_value, "valueUrl": unit_uri, "propertyUrl": "qudt:hasUnit"})
    quantity_kind_uri = _text(row.get("quantity_kind_uri"))
    if quantity_kind_uri:
        columns.append({"virtual": True, "aboutUrl": quantity_value, "valueUrl": quantity_kind_uri, "propertyUrl": "qudt:hasQuantityKind"})
    return columns


def _build_table(
    table_key: str,
    rows: list[dict[str, Any]],
    request: CSVWExportRequest,
) -> dict[str, Any]:
    metadata = _metadata_rows(rows, table_key)
    buckets = _effective_buckets(metadata, request.column_buckets.get(table_key, {}))
    foi_columns = buckets.get(FOI_BUCKET, [])
    if len(foi_columns) > 1:
        raise ValueError(f"Table '{table_key}' must have at most one '{FOI_BUCKET}' column.")
    foi_column = foi_columns[0] if foi_columns else None
    if foi_column and foi_column not in metadata:
        raise ValueError(f"Table '{table_key}' refers to missing FOI column '{foi_column}'.")

    base_url = request.base_url.rstrip("/") + "/"
    temporal_roles = request.temporal_deepdive.get(table_key, {})
    spatial_roles = request.spatial_types.get(table_key, {})
    spatial_defaults = request.spatial_fit_for_all.get(table_key, {})
    real_columns: list[dict[str, Any]] = []
    virtual_columns: list[dict[str, Any]] = []

    def row_for(column_name: str) -> dict[str, Any]:
        if column_name not in metadata:
            raise ValueError(f"Table '{table_key}' bucket refers to unknown column '{column_name}'.")
        return metadata[column_name]

    for column_name in foi_columns:
        row = row_for(column_name)
        column = {"name": column_name, "propertyUrl": "dcterms:identifier", "datatype": _datatype(row)}
        _column_description(column, row)
        real_columns.append(column)

    spatial_columns = buckets.get(SPATIAL_BUCKET, [])
    if spatial_columns or spatial_defaults:
        geo_node = _node(base_url, foi_column, "/geo")
        virtual_columns.extend([
            {"virtual": True, "aboutUrl": _node(base_url, foi_column), "valueUrl": geo_node, "propertyUrl": "schema:geo"},
            {"virtual": True, "aboutUrl": geo_node, "valueUrl": "geo:Point", "propertyUrl": "rdf:type"},
        ])
    for column_name in spatial_columns:
        row = row_for(column_name)
        role = spatial_roles.get(column_name)
        column = {
            "name": column_name,
            "aboutUrl": _node(base_url, foi_column, "/geo"),
            "propertyUrl": SPATIAL_ROLE_TO_PROPERTY.get(role, "geo:location"),
            "datatype": _datatype(row),
        }
        _column_description(column, row)
        real_columns.append(column)
    for role, value in spatial_defaults.items():
        if not _text(value):
            continue
        geo_node = _node(base_url, foi_column, "/geo")
        if "reference system" in role.lower():
            try:
                value_url = f"http://www.opengis.net/def/crs/EPSG/0/{int(value)}"
            except (TypeError, ValueError):
                continue
            virtual_columns.append({"virtual": True, "aboutUrl": geo_node, "valueUrl": value_url, "propertyUrl": "dcterms:conformsTo"})
        elif role in SPATIAL_ROLE_TO_PROPERTY:
            virtual_columns.append({"virtual": True, "aboutUrl": geo_node, "default": str(value), "propertyUrl": SPATIAL_ROLE_TO_PROPERTY[role]})

    observation_columns = buckets.get(OBSERVATION_BUCKET, [])
    temporal_columns = buckets.get(TEMPORAL_BUCKET, [])
    result_time_columns: list[str] = []
    for column_name in temporal_columns:
        row = row_for(column_name)
        role = temporal_roles.get(column_name) or _text(row.get("concept_type")) or "sosa:phenomenonTime"
        if isinstance(role, dict):
            role = next(iter(role.values()), "sosa:phenomenonTime")
        if role == "sosa:resultTime":
            result_time_columns.append(column_name)
            continue
        column = {"name": column_name, "propertyUrl": "sosa:phenomenonTime", "datatype": _datatype(row, "dateTime")}
        _column_description(column, row)
        real_columns.append(column)

    for column_name in buckets.get(ATTRIBUTE_BUCKET, []):
        row = row_for(column_name)
        property_uri = _text(row.get("concept_uri")) or _text(row.get("concept"))
        column = {"name": column_name, "datatype": _datatype(row)}
        if property_uri:
            column["propertyUrl"] = property_uri
        _column_description(column, row)
        real_columns.append(column)

    for column_name in observation_columns:
        row = row_for(column_name)
        observation = _node(base_url, foi_column, f"/{column_name}")
        quantity_value = f"{observation}/QV"
        column = {"name": column_name, "aboutUrl": quantity_value, "propertyUrl": "qudt:value", "datatype": _datatype(row, "number")}
        _column_description(column, row)
        real_columns.append(column)
        default_result_time = "" if result_time_columns and len(observation_columns) > 1 else None
        virtual_columns.extend(_virtual_observation_columns(column_name, foi_column, base_url, row, default_result_time))

    for column_name in result_time_columns:
        row = row_for(column_name)
        if len(observation_columns) == 1:
            about_url = _node(base_url, foi_column, f"/{observation_columns[0]}")
        else:
            about_url = _node(base_url, foi_column)
        column = {
            "name": column_name,
            "aboutUrl": about_url,
            "propertyUrl": "sosa:resultTime",
            "datatype": _datatype(row, "dateTime"),
        }
        _column_description(column, row)
        real_columns.append(column)

    assigned = {column for names in buckets.values() for column in names}
    unsorted_columns = [
        *buckets.get(UNSORTED_BUCKET, []),
        *(column_name for column_name in metadata if column_name not in assigned),
    ]
    for column_name in dict.fromkeys(unsorted_columns):
        row = row_for(column_name)
        column = {"name": column_name, "datatype": _datatype(row)}
        _column_description(column, row)
        real_columns.append(column)

    if foi_column:
        foi_type = _text(metadata[foi_column].get("concept_uri"))
        if foi_type:
            virtual_columns.insert(0, {"virtual": True, "aboutUrl": _node(base_url, foi_column), "valueUrl": foi_type, "propertyUrl": "rdf:type"})

    schema: dict[str, Any] = {"columns": real_columns + virtual_columns}
    table: dict[str, Any] = {
        "url": request.filename_dict.get(table_key, f"{table_key}.csv"),
        "aboutUrl": _node(base_url, foi_column),
    }
    if foi_column:
        schema["primaryKey"] = foi_column
    foreign_keys = _foreign_keys(table_key, metadata, request.filename_dict, request.relationships)
    if foreign_keys:
        schema["foreignKeys"] = foreign_keys
    table["tableSchema"] = schema
    return table


def build_csvw_document(payload: CSVWExportRequest | dict[str, Any]) -> dict[str, Any]:
    """Build a CSVW document from a request object or JSON-compatible mapping."""
    request = payload if isinstance(payload, CSVWExportRequest) else CSVWExportRequest.from_mapping(payload)
    if not _text(request.base_url):
        raise ValueError("base_url must not be empty.")
    tables = [_build_table(table_key, rows, request) for table_key, rows in request.metadata_by_table.items()]
    if not tables:
        raise ValueError("metadata_by_table must contain at least one table.")
    if len(tables) == 1:
        return {"@context": SOSA_CONTEXT, **tables[0]}
    return {"@context": SOSA_CONTEXT, "tables": tables}


def build_csvw_from_app_metadata(
    metadata_by_table: dict[str, Any],
    *,
    base_url: str,
    filename_dict: dict[str, str] | None = None,
    column_buckets: dict[str, dict[str, list[str]]] | None = None,
    temporal_deepdive: dict[str, dict[str, Any]] | None = None,
    spatial_types: dict[str, dict[str, str]] | None = None,
    spatial_fit_for_all: dict[str, dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build CSVW directly from the DataAnnotator app's metadata mapping.

    Each table may be a pandas DataFrame (as stored in ``metadata_df``) or an
    already JSON-compatible list of metadata records.  Keeping the conversion
    here avoids a Streamlit or pandas dependency in this module.
    """
    records_by_table: dict[str, list[dict[str, Any]]] = {}
    for table_key, table_metadata in metadata_by_table.items():
        if hasattr(table_metadata, "to_dict"):
            table_metadata = table_metadata.to_dict(orient="records")
        records_by_table[table_key] = table_metadata

    return build_csvw_document({
        "metadata_by_table": records_by_table,
        "base_url": base_url,
        "filename_dict": filename_dict or {},
        "column_buckets": column_buckets or {},
        "temporal_deepdive": temporal_deepdive or {},
        "spatial_types": spatial_types or {},
        "spatial_fit_for_all": spatial_fit_for_all or {},
        "relationships": relationships or [],
    })


def create_fastapi_app():
    """Return an optional FastAPI wrapper without coupling direct use to FastAPI.

    A service host can install ``fastapi`` and run the returned application with
    any ASGI server.  The direct Streamlit path needs neither dependency.
    """
    try:
        fastapi = import_module("fastapi")
    except ImportError as error:
        raise RuntimeError(
            "FastAPI is required only to host this exporter as an HTTP API. "
            "Install it in the service environment first."
        ) from error

    api = fastapi.FastAPI(title="DataAnnotator CSVW Profile API", version="1.0.0")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/csvw")
    def create_csvw(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return build_csvw_document(payload)
        except ValueError as error:
            raise fastapi.HTTPException(status_code=422, detail=str(error)) from error

    return api