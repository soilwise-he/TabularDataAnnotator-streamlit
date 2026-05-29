import json
import re

import pandas as pd
import streamlit as st

from ui.blocks import add_Soilwise_contact_sidebar, add_Soilwise_logo, add_clear_cache_button
from util.metadata import apply_new_metadata_info


st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")
add_Soilwise_logo()
add_clear_cache_button(key_prefix="methods_page")

meta_key = "metadata_df"
#_EXCLUDED_ELEMENTS = {"sosa:FeatureOfInterest", "ssn:Property"}
_INCLUDE_ELEMENTS = {"sosa:observedProperty"}


ISO_PATTERN = re.compile(r"\bISO\s*[A-Z]*\s*\d+(?:-\d+)?(?::\d{4})?\b", re.IGNORECASE)
CEN_PATTERN = re.compile(r"\b(?:CEN|EN)\s*\d+(?:-\d+)?(?::\d{4})?\b", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b(10[.][0-9]{3,}(?:[.][0-9]+)*(?:(?![\"&\'])\S)+)\b")
URL_PATTERN = re.compile(r"https?://[^\s<>)\]}\"]+")

# -------------------- Helper data and functions --------------------

REFERENCE_TYPE_STYLES = {
	"ISO": {"label": "ISO", "color": "blue" , "icon": ":material/award_star:"},
	"CEN": {"label": "CEN",  "color": "violet", "icon": ":material/thumb_up:"},
	"DOI": {"label": "DOI", "color": "green", "icon": ":material/thumb_up:"},
	"URL": {"label": "URL", "color": "orange", "icon": ":material/link:"},
	"STRING": {"label": "String", "color": "gray", "icon": ":material/check:"},
	"EMPTY": {"label": "Empty", "color": "red", "icon": None},
}

def _first_match(text: str, pattern: re.Pattern) -> str:
	if not text:
		return ""
	m = pattern.search(text)
	return m.group(0) if m else ""


def _to_text(value) -> str:
	if isinstance(value, dict):
		return json.dumps(value, ensure_ascii=False)
	if value is None:
		return ""
	if isinstance(value, float) and pd.isna(value):
		return ""
	return str(value)


def _strip_trailing(text: str) -> str:
	return str(text).strip().rstrip(".,;:)]}>")


def _first_non_empty(*values) -> str:
	for value in values:
		text = _to_text(value).strip()
		if text:
			return text
	return ""


def _extract_first_doi(text: str) -> str:
	if not text:
		return ""

	found = []
	seen = set()

	def validate_part(candidate: str) -> bool:
		return bool(candidate and DOI_PATTERN.search(candidate))

	# Direct DOI token matches.
	for m in DOI_PATTERN.finditer(text):
		candidate = _strip_trailing(m.group(1) if m.groups() else m.group(0))
		if validate_part(candidate) and candidate not in seen:
			found.append(candidate)
			seen.add(candidate)

	# Find doi.org / dx.doi.org links.
	for m in re.finditer(r'https?://(?:dx\.)?doi\.org/([^?\s#<>]+)', text, flags=re.IGNORECASE):
		candidate = _strip_trailing(m.group(1))
		if validate_part(candidate) and candidate not in seen:
			found.append(candidate)
			seen.add(candidate)

	# Find doi: prefixed tokens.
	for m in re.finditer(r'doi:\s*(10\.\d{4,9}/[^\s"<>\)\]\;:,]+)', text, flags=re.IGNORECASE):
		candidate = _strip_trailing(m.group(1))
		if validate_part(candidate) and candidate not in seen:
			found.append(candidate)
			seen.add(candidate)

	return found[0] if found else ""


def _reference_type_badge_markup(reference_kind: str) -> str:
	style = REFERENCE_TYPE_STYLES[reference_kind]
	return (
		"<div style=\"display:flex;align-items:center;height:100%;padding-top:1.9rem;\">"
		f"<span style=\"display:inline-block;padding:0.35rem 0.7rem;border-radius:999px;"
		f"background:{style['background']};color:{style['color']};font-size:0.8rem;font-weight:600;\">"
		f"{style['label']}"
		"</span></div>"
	)



def find_var_column(
	df: pd.DataFrame,
	variable_names: list[str],
	min_overlap: float = 0.6,
) -> tuple[str | None, float]:
	if df is None or df.empty or not variable_names:
		return None, 0.0

	normalized_targets = {
		_to_text(value).strip().lower()
		for value in variable_names
		if _to_text(value).strip()
	}
	if not normalized_targets:
		return None, 0.0

	best_column = None
	best_overlap = 0.0

	for col in df.columns:
		column_values = {
			_to_text(value).strip().lower()
			for value in df[col].dropna()
			if _to_text(value).strip()
		}
		if not column_values:
			continue

		overlap = len(column_values & normalized_targets) / len(normalized_targets)
		if overlap > best_overlap:
			best_overlap = overlap
			best_column = col

	if best_overlap < min_overlap:
		return None, best_overlap

	return best_column, best_overlap


def _order_by_metadata_names(df: pd.DataFrame, ordered_names: list[str]) -> pd.DataFrame:
	"""Keep rows aligned with the original metadata variable order."""
	if df is None or df.empty or "name" not in df.columns:
		return df

	name_order = {
		_to_text(name).strip(): idx
		for idx, name in enumerate(ordered_names)
	}

	ordered_df = df.copy()
	ordered_df["_meta_order"] = ordered_df["name"].map(lambda v: name_order.get(_to_text(v).strip(), 10**9))
	ordered_df = ordered_df.sort_values(by="_meta_order", kind="stable").drop(columns=["_meta_order"])
	return ordered_df.reset_index(drop=True)

@st.cache_resource()
def _ensure_method_review_columns(df: pd.DataFrame) -> pd.DataFrame:
	expected_columns = [
		"name",
		"method_name",
		"description",
		"confidence",
		"reference",
		"iso_standard",
		"cen_standard",
		"doi_reference",
		"other_url_reference",
	]

	if df is None or df.empty:
		return pd.DataFrame(columns=expected_columns)

	out = df.copy()
	if "name" not in out.columns:
		out = out.reset_index().rename(columns={"index": "name"})

	for col in (col for col in expected_columns if col != "name"):
		if col not in out.columns:
			out[col] = ""

	for idx, row in out.iterrows():
		reference = _first_non_empty(
			row.get("reference", ""),
			row.get("iso_standard", ""),
			row.get("cen_standard", ""),
			row.get("doi_reference", ""),
			row.get("other_url_reference", ""),
		)
		method_name = _to_text(row.get("method_name", ""))
		description = _to_text(row.get("description", ""))
		searchable_text = "\n".join([reference, method_name, description])

		out.at[idx, "method_name"] = method_name
		out.at[idx, "description"] = description
		out.at[idx, "confidence"] = _to_text(row.get("confidence", ""))

		derived_reference = reference or _first_non_empty(
			_first_match(searchable_text, ISO_PATTERN),
			_first_match(searchable_text, CEN_PATTERN),
			_extract_first_doi(searchable_text),
			_first_match(searchable_text, URL_PATTERN),
		)
		out.at[idx, "reference"] = derived_reference

		if not _to_text(row.get("iso_standard", "")).strip():
			out.at[idx, "iso_standard"] = _first_match(searchable_text, ISO_PATTERN)
		if not _to_text(row.get("cen_standard", "")).strip():
			out.at[idx, "cen_standard"] = _first_match(searchable_text, CEN_PATTERN)
		if not _to_text(row.get("doi_reference", "")).strip():
			out.at[idx, "doi_reference"] = _extract_first_doi(searchable_text)
		if not _to_text(row.get("other_url_reference", "")).strip():
			out.at[idx, "other_url_reference"] = _first_match(searchable_text, URL_PATTERN)

	return out

def extracted_reference_methode_from_string(text: str) -> tuple[str, str]:
	"""Extracts a reference methode from the given text and identifies its type (ISO, CEN, DOI, URL, STRING or EMPTY)."""

	searchable_text = _to_text(text).strip()
	extractors: list[tuple[str, callable]] = [
		("ISO", lambda t: _first_match(t, ISO_PATTERN)),
		("CEN", lambda t: _first_match(t, CEN_PATTERN)),
		("DOI", _extract_first_doi),
		("URL", lambda t: _first_match(t, URL_PATTERN)),
	]
	for kind, extractor in extractors:
		value = extractor(searchable_text)
		if value:
			return value, kind
	return "", "EMPTY"



# PIN -------------------- UI --------------------

st.title("⚙️ Method Annotation")
st.markdown(
	"""
Review AI-extracted methods and curate a single final reference for each variable.

The reference field stays editable, and the badge next to it shows whether the current value looks like ISO, CEN, DOI, URL, or free text.

☝️ HINT: If the method information is available in a column, leave this column in the 'Unsorted' bucket on the 'Column Sorting' page and this wil come available as a selection for each variable here. 
"""
)

if meta_key not in st.session_state or st.session_state.get(meta_key) is None:
	st.info("No metadata found. Complete step 1 first.")
	st.stop()

if "AI_var_Methode" not in st.session_state or not st.session_state["AI_var_Methode"]:
	st.info("No AI augmented method output found yet. This can help you process this step, if you still want to get some help based on context documents you already have go to page '✒️ Context' and provide some context documents.")
	# st.stop()
else:
	method_tables = st.session_state["AI_var_Methode"]

if "method_review_tables" not in st.session_state:
	st.session_state["method_review_tables"] = {}


tab_labels = list(st.session_state[meta_key].keys())
tabs = st.tabs(tab_labels)

for tab, table_key in zip(tabs, tab_labels):
	with tab:
		table_meta_df = st.session_state[meta_key].get(table_key, pd.DataFrame())
		ordered_meta_names = table_meta_df["name"].tolist() if "name" in table_meta_df.columns else []
		excluded_names: set[str] = set()
		if not table_meta_df.empty and {"name", "element"}.issubset(table_meta_df.columns):
			element_values = table_meta_df["element"].astype(str).str.strip()
			excluded_names = set(
				table_meta_df.loc[~element_values.isin(_INCLUDE_ELEMENTS), "name"].map(_to_text).str.strip()
			)

		# First fill the table with augmentations from LLM extraction and Ansis vocab matching
		if table_key not in st.session_state["method_review_tables"]:
			if "df_selection_keywords" in st.session_state and not st.session_state["df_selection_keywords"].get(table_key, pd.DataFrame()).empty: 
				# Identifying headers matched to Ansis vocab
				df_selection_keywords = st.session_state["df_selection_keywords"].get(table_key, pd.DataFrame())

				df_selection_keywords_Ansis = df_selection_keywords[df_selection_keywords["source"] == "Ansis"]
				query_uri_mapping_Ansis_TableKey = {}
				if {"query", "uri"}.issubset(df_selection_keywords_Ansis.columns):
					query_uri_mapping_Ansis_TableKey = (
						df_selection_keywords_Ansis.assign(
							query=df_selection_keywords_Ansis["query"].map(_to_text).str.strip(),
							uri=df_selection_keywords_Ansis["uri"].map(_to_text).str.strip(),
						)
						.query("query != '' and uri != ''")
						.loc[lambda d: ~d["query"].isin(excluded_names)]
						.drop_duplicates(subset=["query"], keep="first")
						.set_index("query")["uri"]
						.to_dict()
					)
				st.session_state["method_review_tables"][table_key] = pd.DataFrame({
					"name": list(query_uri_mapping_Ansis_TableKey.keys()),
					"reference": list(query_uri_mapping_Ansis_TableKey.values()),
					"kind":["URI"] * len(query_uri_mapping_Ansis_TableKey),
				})
			else:
				st.session_state["method_review_tables"][table_key] = pd.DataFrame(columns=["name", "reference", "kind"])
			
			
			if method_tables:
				## Ensure header column is named "name" for better matching with metadata_df
				AI_guess_methode_df = method_tables.get(table_key, pd.DataFrame())
				meta_names = [
					name
					for name in st.session_state[meta_key][table_key]["name"].tolist()
					if _to_text(name).strip() not in excluded_names
				]
				matched_column = "name" if "name" in AI_guess_methode_df.columns else None
				overlap_ratio = 1.0 if matched_column else 0.0
				if matched_column is None:
					matched_column, overlap_ratio = find_var_column(AI_guess_methode_df, meta_names, min_overlap=0.6)
					if matched_column:
						AI_guess_methode_df = AI_guess_methode_df.rename(columns={matched_column: "name"})

				# Identifying AI guesses

				cols = AI_guess_methode_df.columns.tolist()
				cols_info = [col for col in cols if col != "name"]

				if "reference" in cols_info: # prioritize reference column in extraction (if exists)
					cols_info = ["reference"] + [col for col in cols_info if col != "reference"]

				rows_to_add = []
				for idx, row in AI_guess_methode_df.iterrows():
					name_value = row.get("name")
					name_value_text = _to_text(name_value).strip()
					if name_value_text in excluded_names:
						continue

					# prioritize Ansis vocab matches
					if "df_selection_keywords" in st.session_state and (not name_value_text or name_value_text in query_uri_mapping_Ansis_TableKey.keys()): 
						continue
					
					context_info = " | ".join([f"{col}: {_to_text(row.get(col, '')).strip()}" for col in cols_info])
					
					extracted_reference, kind =  extracted_reference_methode_from_string(context_info)
					
					if kind == "EMPTY":
						extracted_reference = row.get(cols_info[0], "")
						kind = "STRING" if _to_text(extracted_reference).strip() else "EMPTY"
					rows_to_add.append({
						"name": name_value,
						"reference": extracted_reference,
						"kind": kind,
					})

				# Merge Ansis vocab matches and AI guesses, Ansis vocab are prioritized
				if rows_to_add:
					st.session_state["method_review_tables"][table_key] = pd.concat(
						[
							st.session_state["method_review_tables"][table_key],
							pd.DataFrame(rows_to_add),
						],
						ignore_index=True,
					)

			# Keep merged rows in the same order as metadata names.
			st.session_state["method_review_tables"][table_key] = _order_by_metadata_names(
				st.session_state["method_review_tables"][table_key],
				ordered_meta_names,
			)

		# Unsorted columns available for use as a method (procedure) column
		unsorted_cols: list[str] = (
			st.session_state.get("column_buckets", {})
			.get(table_key, {})
			.get("Unsorted", [])
		)

		st.caption("Inspect and edit the method details. The reference field is the final value that will be stored.")
		current_df = st.session_state["method_review_tables"][table_key].reset_index(drop=True)

		current_df = _order_by_metadata_names(current_df, ordered_meta_names)
		#BUG: can't handle empty dataframe with empty rows?
		if "name" in current_df.columns:
			current_df = current_df[
				~current_df["name"].map(_to_text).str.strip().isin(excluded_names)
			].reset_index(drop=True)

		edited_rows = []

		for idx, row in current_df.iterrows():
			name_col, input_col, result_col = st.columns([1, 5, 4])

			name_col.markdown(f"**{_to_text(row.get('name', '')).strip()}**")

			reference = input_col.text_input(
				"Reference Input",
				value=_to_text(row.get("reference", "")).strip(),
				key=f"reference_{table_key}_{idx}",
				label_visibility="collapsed",
			)

			# Optional: pick an Unsorted column as the procedure column
			previous_method_col = _to_text(row.get("method_column", "")).strip()
			if unsorted_cols:
				method_col_options = ["-"] + unsorted_cols
				default_idx = (
					method_col_options.index(previous_method_col)
					if previous_method_col in method_col_options
					else 0
				)
				input_col.space("medium")
				selected_method_col = input_col.selectbox(
					"Or select a column with method information:",
					options=method_col_options,
					index=default_idx,
					key=f"method_col_{table_key}_{idx}",
					help="An **unsorted column** that holds flexible methode information for the observed property can be chosen here.",
				)
				if selected_method_col == "-":
					selected_method_col = ""
			else:
				selected_method_col = ""

			extracted_reference, kind = extracted_reference_methode_from_string(reference)

			if kind == "EMPTY" and reference.strip():
				extracted_reference = reference
				kind = "STRING" if _to_text(extracted_reference).strip() else "EMPTY"


			if not selected_method_col:
				result_col.write(f"**Extracted Reference:** {extracted_reference}")
				result_col.badge(
					REFERENCE_TYPE_STYLES[kind]["label"],
					color=REFERENCE_TYPE_STYLES[kind]["color"],
					icon=REFERENCE_TYPE_STYLES[kind]["icon"],
				)

			else:
				result_col.space(110)
				result_col.info(
					f"Procedure column: **{selected_method_col}**",
					icon=":material/settings:",
				)


			edited_rows.append({
				"name": row["name"],
				"reference": extracted_reference,
				"method": f"#[{selected_method_col}]" if selected_method_col else extracted_reference,
				"kind": kind,
				"method_column": selected_method_col,
			})
			st.divider()
		st.session_state["method_review_tables"][table_key] = pd.DataFrame(edited_rows)


# Apply method/reference — drop method_column so it is not written to the metadata schema.
apply_tables = {
	key: df.drop(columns=["method_column"], errors="ignore")
	for key, df in st.session_state["method_review_tables"].items()
}
st.session_state[meta_key] = apply_new_metadata_info(
		apply_tables,
		st.session_state[meta_key],
		overwrite="yes_incl_blanks",
	)

# For rows where a procedure column was chosen, write sosa:Procedure to the element field.
for _tbl_key, _review_df in st.session_state["method_review_tables"].items():
	if "method_column" not in _review_df.columns:
		continue
	_meta_df = st.session_state[meta_key].get(_tbl_key)
	if _meta_df is None or "element" not in _meta_df.columns:
		continue
	for _, _rev_row in _review_df.iterrows():
		_method_col_val = _to_text(_rev_row.get("method_column", "")).strip()
		if not _method_col_val:
			continue
		_mask = _meta_df["name"].map(_to_text).str.strip() == _method_col_val
		if _mask.any():
			_meta_df.loc[_mask, "element"] = "sosa:Procedure"
	st.session_state[meta_key][_tbl_key] = _meta_df


st.markdown("#### Current metadata table")

meta_tabs = st.tabs(list(st.session_state[meta_key].keys()))
for tab, table_key in zip(meta_tabs, st.session_state[meta_key].keys()):
	with tab:
		st.dataframe(st.session_state[meta_key][table_key], width="stretch")

# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()

