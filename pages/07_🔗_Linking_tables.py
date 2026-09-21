import itertools
import re
from time import perf_counter

import pandas as pd
import streamlit as st

from ui.blocks import add_Soilwise_contact_sidebar, add_Soilwise_logo, add_clear_cache_button
from util.metadata import normalize_metadata_columns, sync_relationships_to_metadata


st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")
add_Soilwise_logo()
add_clear_cache_button(key_prefix="linking_tables_page")

meta_key = "metadata_df"
data_key = "tabular_data_dict"

if isinstance(st.session_state.get(meta_key), dict):
	st.session_state[meta_key] = normalize_metadata_columns(st.session_state[meta_key])

RELATION_OPTIONS = [
	"not-linked",
	"one-to-one",
	"one-to-many",
	"many-to-one",
	"many-to-many",
]

RELATION_ARROW = {
	"not-linked": "∅",
	"one-to-one": "1 ↔ 1",
	"one-to-many": "1 → N",
	"many-to-one": "N → 1",
	"many-to-many": "N ↔ N",
}

TYPE_PREFERENCE = {
	"string": 3,
	"integer": 2,
	"float": 1,
	"other": 0,
}

PROFILE_SAMPLE_LIMIT = 5000
OVERLAP_SAMPLE_LIMIT = 20000


# -------------------- Helper data and functions --------------------

st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #F0F2F6;
            border-radius: 4px 4px 0px 0px;
            padding-left: 12px;
            padding-right: 12px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFBF1;
        }

    </style>""", unsafe_allow_html=True)

def _to_text(value) -> str:
	if value is None:
		return ""
	try:
		if pd.isna(value):
			return ""
	except (TypeError, ValueError):
		pass
	return str(value)


def _normalized_text_series(series: pd.Series | None) -> pd.Series:
	if series is None:
		return pd.Series(dtype="object")
	values = [_to_text(value).strip() for value in series.dropna().tolist()]
	return pd.Series([value for value in values if value], dtype="object")


def _normalize_name(value: str) -> str:
	return re.sub(r"[^a-z0-9]+", " ", _to_text(value).strip().lower()).strip()


def _name_match_bonus(left_name: str, right_name: str) -> float:
	left_normalized = _normalize_name(left_name)
	right_normalized = _normalize_name(right_name)
	if not left_normalized or not right_normalized:
		return 0.0
	if left_normalized == right_normalized:
		return 50.0
	left_tokens = set(left_normalized.split())
	right_tokens = set(right_normalized.split())
	shared_tokens = left_tokens & right_tokens
	bonus = 12.0 * len(shared_tokens)
	identifier_tokens = {"id", "identifier", "key", "code", "uuid", "guid"}
	if shared_tokens & identifier_tokens:
		bonus += 18.0
	return bonus


def _common_name_bonus(column_name: str) -> float:
	normalized = _normalize_name(column_name)
	tokens = set(normalized.split())
	bonus = 0.0
	if normalized in {"id", "identifier", "uuid", "guid", "key"}:
		bonus += 40.0
	if "id" in tokens or normalized.endswith(" id") or normalized.endswith("_id"):
		bonus += 28.0
	if tokens & {"identifier", "uuid", "guid", "key", "code"}:
		bonus += 18.0
	if tokens & {"site", "sample", "station", "plot", "record", "observation", "obs", "location"}:
		bonus += 10.0
	return bonus


def _clean_series(series: pd.Series) -> pd.Series:
	if series is None:
		return pd.Series(dtype="object")
	if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
		return _normalized_text_series(series)
	return series.dropna()


def _sample_series(series: pd.Series | None, sample_limit: int | None = None) -> pd.Series:
	if series is None:
		return pd.Series(dtype="object")
	sampled = series.dropna()
	if sample_limit and len(sampled) > sample_limit:
		sampled = sampled.sample(n=sample_limit, random_state=0)
	return sampled


def _clean_series_sample(series: pd.Series | None, sample_limit: int | None = None) -> pd.Series:
	raw_sample = _sample_series(series, sample_limit)
	if raw_sample.empty:
		return raw_sample
	if pd.api.types.is_object_dtype(raw_sample) or pd.api.types.is_string_dtype(raw_sample):
		return _normalized_text_series(raw_sample)
	return raw_sample


def _series_kind(series: pd.Series) -> str:
	if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
		return "string"
	if pd.api.types.is_integer_dtype(series):
		return "integer"
	if pd.api.types.is_float_dtype(series):
		return "float"
	return "other"


def _column_profile(df: pd.DataFrame | None, column_name: str) -> dict:
	if df is None or column_name not in df.columns:
		return {
			"column": column_name,
			"score": _common_name_bonus(column_name),
			"unique_ratio": 0.0,
			"is_unique": False,
			"kind": "other",
		}

	series = df[column_name]
	kind = _series_kind(series)
	cleaned = _clean_series_sample(series, PROFILE_SAMPLE_LIMIT)
	if cleaned.empty:
		return {
			"column": column_name,
			"score": _common_name_bonus(column_name),
			"unique_ratio": 0.0,
			"is_unique": False,
			"kind": kind,
		}

	total_count = len(cleaned)
	unique_count = cleaned.nunique(dropna=True)
	unique_ratio = unique_count / total_count if total_count else 0.0
	original_non_null_count = int(series.notna().sum())
	is_sampled = original_non_null_count > PROFILE_SAMPLE_LIMIT
	is_unique = total_count > 0 and unique_count == total_count and not is_sampled
	if is_sampled and unique_ratio >= 0.995:
		is_unique = True
	score = (
		(unique_ratio * 100.0)
		+ (35.0 if is_unique else 0.0)
		+ (TYPE_PREFERENCE[kind] * 12.0)
		+ _common_name_bonus(column_name)
	)
	return {
		"column": column_name,
		"score": score,
		"unique_ratio": unique_ratio,
		"is_unique": is_unique,
		"kind": kind,
	}


def _value_overlap_metrics(left_df: pd.DataFrame | None, left_column: str, right_df: pd.DataFrame | None, right_column: str) -> tuple[float, int]:
	if left_df is None or right_df is None or left_column not in left_df.columns or right_column not in right_df.columns:
		return 0.0, 0
	left_values = set(_normalized_text_series(_clean_series(left_df[left_column])))
	right_values = set(_normalized_text_series(_clean_series(right_df[right_column])))
	left_values.discard("")
	right_values.discard("")
	if not left_values or not right_values:
		return 0.0, 0
	overlap_count = len(left_values & right_values)
	overlap_ratio = overlap_count / min(len(left_values), len(right_values)) if min(len(left_values), len(right_values)) else 0.0
	return overlap_ratio, overlap_count


def _column_value_set(df: pd.DataFrame | None, column_name: str) -> set[str]:
	if df is None or column_name not in df.columns:
		return set()
	cleaned = _clean_series_sample(df[column_name], OVERLAP_SAMPLE_LIMIT)
	values = set(_normalized_text_series(cleaned))
	values.discard("")
	return values


def _value_overlap_from_sets(left_values: set[str], right_values: set[str]) -> tuple[float, int]:
	if not left_values or not right_values:
		return 0.0, 0
	min_size = min(len(left_values), len(right_values))
	if min_size == 0:
		return 0.0, 0
	overlap_count = len(left_values & right_values)
	overlap_ratio = overlap_count / min_size
	return overlap_ratio, overlap_count


def _infer_relation_from_profiles(left_profile: dict, right_profile: dict) -> str:
	if left_profile["is_unique"] and right_profile["is_unique"]:
		return "one-to-one"
	if left_profile["is_unique"] and not right_profile["is_unique"]:
		return "one-to-many"
	if not left_profile["is_unique"] and right_profile["is_unique"]:
		return "many-to-one"
	return "many-to-many"


def _relationship_fk_edges(relationship_records: list[dict]) -> list[tuple[str, str]]:
	"""Return the logical FK direction for active single-target relationships."""
	edges = []
	for record in relationship_records:
		if not isinstance(record, dict):
			continue

		left_table = _to_text(record.get("left_table")).strip()
		right_table = _to_text(record.get("right_table")).strip()
		relation = record.get("relation")
		if not left_table or not right_table or relation in (None, "not-linked", "many-to-many"):
			continue

		if relation == "one-to-many":
			edges.append((right_table, left_table))
		else:
			edges.append((left_table, right_table))
	return list(dict.fromkeys(edges))


def _find_relationship_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
	"""Find unique table cycles in a directed relationship graph."""
	adjacency: dict[str, list[str]] = {}
	for source, target in edges:
		adjacency.setdefault(source, []).append(target)

	cycles: dict[tuple[str, ...], list[str]] = {}
	for start in adjacency:
		def visit(current: str, path: list[str]) -> None:
			for target in adjacency.get(current, []):
				if target == start:
					cycle = path + [start]
					rotations = [tuple(cycle[index:-1] + cycle[:index]) for index in range(len(cycle) - 1)]
					cycles.setdefault(min(rotations), cycle)
				elif target not in path:
					visit(target, path + [target])

		visit(start, [start])
	return list(cycles.values())


def _find_direct_indirect_path_conflicts(edges: list[tuple[str, str]]) -> list[tuple[str, str, list[str]]]:
	"""Find direct dependencies that also have an alternative indirect path."""
	adjacency: dict[str, list[str]] = {}
	for source, target in edges:
		adjacency.setdefault(source, []).append(target)

	conflicts = []
	for source, target in edges:
		queue = [(source, [source])]
		visited = {source}
		while queue:
			current, path = queue.pop(0)
			for next_table in adjacency.get(current, []):
				if current == source and next_table == target:
					continue
				if next_table == target:
					conflicts.append((source, target, path + [target]))
					queue = []
					break
				if next_table not in visited:
					visited.add(next_table)
					queue.append((next_table, path + [next_table]))
	return conflicts


def _find_table_relationships(left_table: str, right_table: str, table_columns: dict, data_dict: dict) -> dict:
	with st.spinner(f"Find relation in tables; {left_table} → {right_table}"):
		started_at = perf_counter()
		left_df = data_dict.get(left_table)
		right_df = data_dict.get(right_table)
		best_suggestion = {
			"left_id": "",
			"right_id": "",
			"relation": "not-linked",
			"score": -1.0,
			"reason": "",
		}

		profile_started_at = perf_counter() #debug
		left_profiles = {
			column: _column_profile(left_df, column)
			for column in table_columns[left_table]
		}
		right_profiles = {
			column: _column_profile(right_df, column)
			for column in table_columns[right_table]
		}
		profile_elapsed = perf_counter() - profile_started_at #debug

		value_set_started_at = perf_counter() #debug
		left_value_sets = {
			column: _column_value_set(left_df, column)
			for column in table_columns[left_table]
		}
		right_value_sets = {
			column: _column_value_set(right_df, column)
			for column in table_columns[right_table]
		}
		value_set_elapsed = perf_counter() - value_set_started_at #debug

		scoring_started_at = perf_counter() #debug
		for left_column in table_columns[left_table]:
			left_profile = left_profiles[left_column]
			for right_column in table_columns[right_table]:
				right_profile = right_profiles[right_column]
				name_bonus = _name_match_bonus(left_column, right_column)
				overlap_ratio, overlap_count = _value_overlap_from_sets(
					left_value_sets[left_column],
					right_value_sets[right_column],
				)
				overlap_bonus = overlap_ratio * 140.0 + min(overlap_count, 5) * 6.0
				passes_minimum_signal = overlap_count > 0 or name_bonus >= 18.0
				if not passes_minimum_signal:
					continue

				pair_score = left_profile["score"] + right_profile["score"] + name_bonus + overlap_bonus
				if pair_score <= best_suggestion["score"]:
					continue

				reasons = []
				if name_bonus >= 18.0:
					reasons.append("common column naming")
				if overlap_count > 0:
					reasons.append("overlapping values")
				if left_profile["is_unique"] or right_profile["is_unique"]:
					reasons.append("identifier-like uniqueness")
				best_suggestion = {
					"left_id": left_column,
					"right_id": right_column,
					"relation": _infer_relation_from_profiles(left_profile, right_profile),
					"score": pair_score,
					"reason": ", ".join(reasons),
				}
		scoring_elapsed = perf_counter() - scoring_started_at #debug
		total_elapsed = perf_counter() - started_at #debug

		if best_suggestion["score"] < 0.0:
			return {
				"left_id": "",
				"right_id": "",
				"relation": "not-linked",
				"score": 0.0,
				"reason": "",
				"timing": { #debug
					"profile_seconds": round(profile_elapsed, 3),
					"value_set_seconds": round(value_set_elapsed, 3),
					"scoring_seconds": round(scoring_elapsed, 3),
					"total_seconds": round(total_elapsed, 3),
				},
			}

		best_suggestion["timing"] = { #debug
			"profile_seconds": round(profile_elapsed, 3),
			"value_set_seconds": round(value_set_elapsed, 3),
			"scoring_seconds": round(scoring_elapsed, 3),
			"total_seconds": round(total_elapsed, 3),
		}
	return best_suggestion


def _get_table_columns(metadata_df: pd.DataFrame) -> list[str]:
	if metadata_df is None or metadata_df.empty or "name" not in metadata_df.columns:
		return []
	seen = set()
	columns = []
	for value in metadata_df["name"].tolist():
		name = _to_text(value).strip()
		if not name or name in seen:
			continue
		seen.add(name)
		columns.append(name)
	return columns


st.title("🔗 Linking Tables")


# Data preview for each table
preview_tables = st.session_state.get("tabular_data_dict_preview", {})
if preview_tables:
	with st.expander("Recap Data", expanded=False):
		table_names = list(preview_tables.keys())
		table_tabs = st.tabs([f"{table_name}" for table_name in table_names])
		for tab, table_name in zip(table_tabs, table_names):
			with tab:
				st.dataframe(preview_tables[table_name], width="stretch")


st.space("small")	
st.markdown(
	"""
	#### Define how tables relate to each other.

	
	For each pair of tables, choose the relation type and declare the left and right identifier columns used to link them.
	"""
)

if meta_key not in st.session_state or not isinstance(st.session_state.get(meta_key), dict) or not st.session_state[meta_key]:
	st.info("No table metadata found. Complete the input step first.")
	st.stop()

metadata_dict = st.session_state[meta_key]
data_dict = st.session_state.get(data_key, {})
table_keys = list(metadata_dict.keys())

if len(table_keys) < 2:
	st.info("Only one table detected. Upload multiple tables to configure linking relationships.")
	st.stop()

FOI_ID_BUCKET = "Feature of Interest (FOI) - ID"
UNSORTED_BUCKET = "Unsorted"
column_buckets = st.session_state.get("column_buckets", {})

table_columns = {
	table_key: list(dict.fromkeys(
		column_buckets.get(table_key, {}).get(FOI_ID_BUCKET, [])
		+ column_buckets.get(table_key, {}).get(UNSORTED_BUCKET, [])
	))
	for table_key in table_keys
}

if not any(table_columns.values()):
	st.info(
		"No candidate ID columns are available yet. "
		"Go to the **Column Sorting** page and place at least one column in the "
		"*Feature of Interest (FOI) - ID* bucket, or leave a suitable identifier in the "
		"*Unsorted* bucket, for each table that you want to link."
	)
	st.stop()

table_pairs = list(itertools.combinations(table_keys, 2))

if "table_relationships" not in st.session_state:
	st.session_state["table_relationships"] = {}	
relationship_records = []

for left_table, right_table in table_pairs:
	pair_key = f"{left_table}|||{right_table}"
	stored_relationship = st.session_state["table_relationships"].get(pair_key, {})

	suggested_link = _find_table_relationships(left_table, right_table, table_columns, data_dict)

	default_relation = stored_relationship.get("relation", suggested_link["relation"])

	# Use last_left_id/last_right_id so the selection survives a round-trip through "not-linked"
	default_left_id = stored_relationship.get("last_left_id") or stored_relationship.get("left_id") or suggested_link["left_id"]
	default_right_id = stored_relationship.get("last_right_id") or stored_relationship.get("right_id") or suggested_link["right_id"]

	with st.container(border=True):

		# Header
		header_left, header_mid, header_right = st.columns([5, 1, 5])

		# Check if either table has no candidate ID columns
		if not table_columns[left_table] or not table_columns[right_table]:
			missing = [t for t in (left_table, right_table) if not table_columns[t]]
			header_left.markdown(f"#### {left_table}")
			header_right.markdown(f"#### {right_table}")
			st.info(
				f"No columns are marked as **Feature of Interest (FOI) - ID** for "
				f"{', '.join(missing)}. "
				"Go to the **Column Sorting** page and assign at least one column to the "
				"*Feature of Interest (FOI) - ID* bucket before linking these tables."
			)
			continue

		# Selection relation type
		relation = st.selectbox(
			"Relation",
			options=RELATION_OPTIONS,
			index=RELATION_OPTIONS.index(default_relation) if default_relation in RELATION_OPTIONS else 0,
			key=f"relation_{pair_key}",
		)

		relation_arrow = RELATION_ARROW.get(relation, "↔")

		
		header_left.markdown(f"#### {left_table}")
		header_mid.markdown(
			f"<div style='text-align:center;padding-top:0.8rem;font-weight:700;'>{relation_arrow}</div>",
			unsafe_allow_html=True,
		)
		header_right.markdown(f"#### {right_table}")

		# Selection of left/right id columns
		if relation == "not-linked":
			left_id = ""
			right_id = ""
		else:
			left_col, arrow_col, right_col = st.columns([5, 1, 5])
			left_id = left_col.selectbox(
				"Left id",
				options=[""] + table_columns[left_table],
				index=([""] + table_columns[left_table]).index(
					default_left_id
				)
				if default_left_id in ([""] + table_columns[left_table])
				else 0,
				key=f"left_id_{pair_key}",
			)
			arrow_col.markdown(
				f"<div style='text-align:center;padding-top:2rem;font-weight:700;'>{relation_arrow}</div>",
				unsafe_allow_html=True,
			)
			right_id = right_col.selectbox(
				"Right id",
				options=[""] + table_columns[right_table],
				index=([""] + table_columns[right_table]).index(
					default_right_id
				)
				if default_right_id in ([""] + table_columns[right_table])
				else 0,
				key=f"right_id_{pair_key}",
			)

		# Keeping suggested link display outside of the "not-linked" condition so that it can be shown even if the user has selected a different relation type.
		if relation != "not-linked" and suggested_link["left_id"] and suggested_link["right_id"]:
			st.caption(
				f"Suggested link: {suggested_link['left_id']} {RELATION_ARROW.get(suggested_link['relation'], '↔')} {suggested_link['right_id']}"
				+ (f" based on {suggested_link['reason']}." if suggested_link["reason"] else ".")
			)


		# Cardinality check with possible warning feedback
		if relation != "not-linked" and (not left_id or not right_id):
			st.warning("This pair has a relation but no left/right id selected.")
		elif relation != "not-linked":
			left_profile = _column_profile(data_dict.get(left_table), left_id)
			right_profile = _column_profile(data_dict.get(right_table), right_id)
			expected_relation = _infer_relation_from_profiles(left_profile, right_profile)
			if relation != expected_relation:
				st.warning(
					f"**Cardinality warning**: this declaration is **{relation}**, but the selected "
					f"columns appear to be **{expected_relation}** based on their values. "
					f"Left id: {left_id} is {'unique' if left_profile['is_unique'] else 'not unique'}; "
					f"Right id: {right_id} is {'unique' if right_profile['is_unique'] else 'not unique'}.",
					icon="🚨"
				)

		relationship_record = {
			"left_table": left_table,
			"right_table": right_table,
			"relation": relation,
			"left_id": left_id,
			"right_id": right_id,
			# Preserve the last non-empty selection so it can be restored when switching back
			"last_left_id": left_id if left_id else stored_relationship.get("last_left_id", default_left_id),
			"last_right_id": right_id if right_id else stored_relationship.get("last_right_id", default_right_id),
		}
		relationship_records.append(relationship_record)
		st.session_state["table_relationships"][pair_key] = relationship_record

st.divider()
st.markdown("#### Relationship Summary")

summary_df = pd.DataFrame(
	[
		{
			"left_table": record["left_table"],
			"right_table": record["right_table"],
			"relation": record["relation"],
			"left_id": record["left_id"],
			"right_id": record["right_id"],
		}
		for record in relationship_records
	]
)

st.dataframe(summary_df, width="stretch")

relationship_edges = _relationship_fk_edges(relationship_records)
relationship_cycles = _find_relationship_cycles(relationship_edges)


for cycle in relationship_cycles:
    st.warning(
        f'''
		Circular dependency detected:
        '''
		f'''

        {' → '.join(cycle)}.
		
		'''
        f'''Check whether this loop is intentional and whether at least one link can be optional.'''
    )
path_conflicts = _find_direct_indirect_path_conflicts(relationship_edges)
for source, target, indirect_path in path_conflicts:
	st.warning(
		f'''
		Multiple dependency paths detected: 
		
		{source} directly references {target}

		'''
		f'''but it also reaches it through
		
		{' → '.join(indirect_path)}.
		
		'''

		f'''Confirm that the direct link is intentional and cannot conflict with the indirect meaning.''',

		icon="🚩"
	)

st.session_state[meta_key] = sync_relationships_to_metadata(st.session_state.get(meta_key, {}), relationship_records)
if "table_relationships_summary_df" not in st.session_state:
	st.session_state["table_relationships_summary_df"] = pd.DataFrame()
st.session_state["table_relationships_summary_df"] = summary_df.copy()

st.markdown("#### Current metadata table")

meta_tabs = st.tabs(list(st.session_state[meta_key].keys()))
for tab, table_key in zip(meta_tabs, st.session_state[meta_key].keys()):
	with tab:
		st.dataframe(st.session_state[meta_key][table_key], width="stretch")



# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()

