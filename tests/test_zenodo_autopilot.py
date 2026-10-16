from pathlib import Path
import types

import numpy as np
import pandas as pd

from zenodo_autopilot import ROOT, build_human_in_the_loop_guide, parse_args


def test_build_human_in_the_loop_guide_maps_page_flow_and_best_guesses():
    metadata = {
        "example_table": pd.DataFrame(
            {
                "name": ["site_id", "temperature_c"],
                "description": ["Site identifier", "Temperature in degrees C"],
                "column_type": ["string", "numeric"],
            }
        )
    }

    guide = build_human_in_the_loop_guide(metadata)

    assert guide["workflow"][0]["page"] == "Input"
    assert any(step["page"] == "Context" for step in guide["workflow"])
    assert any(step["page"] == "Column Sorting" for step in guide["workflow"])
    assert "example_table" in guide["tables"]
    assert "best_guess" in guide["tables"]["example_table"]["context"]
    assert guide["tables"]["example_table"]["context"]["human_action"]


def test_export_metadata_columns_are_not_added_to_raw_or_export_tables():
    raw = pd.DataFrame({"site_id": ["A1", "A2"], "temperature_c": [12.5, 13.1]})
    export_df = raw.copy()

    assert list(raw.columns) == ["site_id", "temperature_c"]
    assert list(export_df.columns) == ["site_id", "temperature_c"]
    assert "fk_target" not in raw.columns
    assert "codelist_annotation_mode" not in raw.columns
    assert "codelist_uri" not in raw.columns
    assert "codelist_items" not in raw.columns
    assert "fk_target" not in export_df.columns
    assert "codelist_annotation_mode" not in export_df.columns
    assert "codelist_uri" not in export_df.columns
    assert "codelist_items" not in export_df.columns


def test_parse_args_defaults_to_repo_link_list(monkeypatch):
    monkeypatch.setattr("sys.argv", ["zenodo_autopilot.py"])

    args = parse_args()

    assert args.links_file == ROOT / "data" / "Link_List.txt"
    assert args.links == []


def test_build_record_summary_inferrs_column_types():
    df = pd.DataFrame(
        {
            "site_id": ["A1", "A2"],
            "temperature_c": [12.5, 13.1],
            "sample_date": ["2024-01-01", "2024-01-02"],
        }
    )

    summary = __import__("zenodo_autopilot").build_record_summary("example", df)

    assert summary["column_types"]["site_id"] == "string"
    assert summary["column_types"]["temperature_c"] in {"float", "integer"}
    assert summary["column_types"]["sample_date"] == "date"


def test_generate_llm_descriptions_use_metadata_name_description_rows():
    autopilot = __import__("zenodo_autopilot")
    df = pd.DataFrame(
        {
            "name": ["site_id", "temperature_c"],
            "description": ["Unique site identifier", "Temperature in degrees Celsius"],
        }
    )

    result_by_name = {
        row["name"]: row["description"]
        for row in [
            {"name": "site_id", "description": autopilot._resolve_variable_description(df, "site_id")},
            {"name": "temperature_c", "description": autopilot._resolve_variable_description(df, "temperature_c")},
        ]
    }

    assert result_by_name["site_id"] == "Unique site identifier"
    assert result_by_name["temperature_c"] == "Temperature in degrees Celsius"


def test_guess_bucket_from_name_uses_sorting_page_preferred_bucket(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    fake_sorting_module = types.SimpleNamespace(
        _preferred_bucket_for_column=lambda tbl, col_name: "Observed Property",
        _guess_bucket=lambda info: "FOI - Attribute",
    )
    monkeypatch.setattr(autopilot, "_load_module", lambda *args, **kwargs: fake_sorting_module)

    assert autopilot._guess_bucket_from_name("soil_moisture", "volumetric water content", "float") == "Observed Property"


def test_guess_concept_type_uses_spatial_deepdive_role(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    fake_sorting_module = types.SimpleNamespace(
        _guess_spatial_role=lambda col_name: "Y",
        _guess_bucket=lambda info: "FOI - Spatial Information",
    )
    monkeypatch.setattr(autopilot, "_load_module", lambda *args, **kwargs: fake_sorting_module)

    assert autopilot._guess_concept_type_from_name("latitude", "latitude in decimal degrees", "float") == "geo:lat"


def test_generate_keyword_matches_reuses_app_vocab_matching_helpers(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    fake_matcher_module = types.SimpleNamespace()
    fake_matcher_module.load_vocab_indexes = lambda modelname="all-MiniLM-L6-v2": (
        "fake-index",
        {
            1: {"uri": "http://example.com/soil", "label": "soil", "source": "ansis", "QC_label": "prefLabel", "definition": "soil"},
            2: {"uri": "http://example.com/temperature", "label": "temperature", "source": "custom", "QC_label": "altLabel", "definition": "temperature"},
            3: {"uri": "http://example.com/date", "label": "date", "source": "ansis", "QC_label": "prefLabel", "definition": "date"},
        },
    )
    fake_matcher_module.load_sentence_model = lambda modelname: object()

    def fake_find_nearest_vocab(df_descriptions, _index_lib, meta_data_dict, k_nearest=4):
        assert fake_matcher_module.model is not None
        return [
            {"query": "soil_moisture", "Distance": 0.1, "uri": "http://example.com/soil", "source": "ansis", "label": "soil", "QC_label": "prefLabel"},
            {"query": "soil_moisture", "Distance": 0.9, "uri": "http://example.com/temperature", "source": "custom", "label": "temperature", "QC_label": "altLabel"},
            {"query": "sample_date", "Distance": 0.2, "uri": "http://example.com/date", "source": "ansis", "label": "date", "QC_label": "prefLabel"},
        ]

    fake_matcher_module.find_nearest_vocab = fake_find_nearest_vocab
    monkeypatch.setattr(autopilot, "_load_module", lambda *args, **kwargs: fake_matcher_module)

    metadata = {
        "example_table": pd.DataFrame(
            {
                "name": ["soil_moisture", "sample_date"],
                "description": ["soil moisture value", "sampling date"],
            }
        )
    }

    result = autopilot._generate_keyword_matches_for_metadata(metadata, source_filter={"ansis"})

    assert sorted(result["example_table"].keys()) == ["sample_date", "soil_moisture"]
    assert result["example_table"]["soil_moisture"][0]["uri"] == "http://example.com/soil"
    assert result["example_table"]["sample_date"][0]["uri"] == "http://example.com/date"


def test_generate_keyword_matches_falls_back_to_direct_batch_matching():
    autopilot = __import__("zenodo_autopilot")

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            values = {
                "soil_moisture": np.array([1.0, 0.0], dtype="float32"),
                "sample_date": np.array([0.0, 1.0], dtype="float32"),
            }
            return values.get(str(text), np.array([0.5, 0.5], dtype="float32"))

    fake_index = types.SimpleNamespace()
    def fake_search(vec, k):
        if vec[0, 0] > 0.5:
            return (
                np.array([[0.8, 0.2]], dtype="float32"),
                np.array([[1, 2]], dtype="int32"),
            )
        return (
            np.array([[0.8, 0.2]], dtype="float32"),
            np.array([[2, 1]], dtype="int32"),
        )

    fake_index.search = fake_search

    metadata = {
        "example_table": pd.DataFrame(
            {
                "name": ["soil_moisture", "sample_date"],
                "description": ["soil moisture value", "sampling date"],
            }
        )
    }

    result = autopilot._match_keyword_candidates_direct(
        metadata["example_table"],
        fake_index,
        {
            1: {"uri": "http://example.com/soil", "label": "soil", "source": "ansis", "QC_label": "prefLabel"},
            2: {"uri": "http://example.com/date", "label": "date", "source": "ansis", "QC_label": "prefLabel"},
        },
        FakeModel(),
        source_filter={"ansis"},
        k_nearest=2,
    )

    assert result["soil_moisture"][0]["uri"] == "http://example.com/soil"
    assert result["sample_date"][0]["uri"] == "http://example.com/date"


def test_generate_keyword_matches_handles_raw_faiss_metadata_without_normalized_keys():
    autopilot = __import__("zenodo_autopilot")

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            values = {
                "soil_moisture": np.array([1.0, 0.0], dtype="float32"),
                "sample_date": np.array([0.0, 1.0], dtype="float32"),
            }
            return values.get(str(text), np.array([0.5, 0.5], dtype="float32"))

    fake_index = types.SimpleNamespace()
    def fake_search(vec, k):
        if vec[0, 0] > 0.5:
            return (
                np.array([[0.8, 0.2]], dtype="float32"),
                np.array([[1, 2]], dtype="int32"),
            )
        return (
            np.array([[0.8, 0.2]], dtype="float32"),
            np.array([[2, 1]], dtype="int32"),
        )

    fake_index.search = fake_search

    result = autopilot._match_keyword_candidates_direct(
        pd.DataFrame({
            "name": ["soil_moisture", "sample_date"],
            "description": ["soil moisture value", "sampling date"],
        }),
        fake_index,
        {
            1: {"concept_uri": "http://example.com/soil", "label": "soil", "source_id": "local_vocabularies:ansis.csv", "label_type": "prefLabel", "definition_en": "soil"},
            2: {"concept_uri": "http://example.com/date", "label": "date", "source_id": "local_vocabularies:ansis.csv", "label_type": "prefLabel", "definition_en": "date"},
        },
        FakeModel(),
        source_filter={"ansis.csv"},
        k_nearest=2,
    )

    assert result["soil_moisture"][0]["uri"] == "http://example.com/soil"
    assert result["sample_date"][0]["uri"] == "http://example.com/date"


def test_generate_keyword_matches_handles_dict_result_from_direct_matcher(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    def fake_direct_matcher(*args, **kwargs):
        return {
            "soil_moisture": [{"query": "soil_moisture", "uri": "http://example.com/soil", "source": "ansis", "Distance": 0.1}],
            "sample_date": [{"query": "sample_date", "uri": "http://example.com/date", "source": "ansis", "Distance": 0.2}],
        }

    monkeypatch.setattr(autopilot, "_match_keyword_candidates_direct", fake_direct_matcher)

    metadata = {
        "example_table": pd.DataFrame(
            {
                "name": ["soil_moisture", "sample_date"],
                "description": ["soil moisture value", "sampling date"],
            }
        )
    }

    result = autopilot._generate_keyword_matches_for_metadata(metadata, source_filter={"ansis"})

    assert result["example_table"]["soil_moisture"][0]["uri"] == "http://example.com/soil"
    assert result["example_table"]["sample_date"][0]["uri"] == "http://example.com/date"


def test_guess_bucket_from_name_detects_foi_id_with_real_patterns():
    autopilot = __import__("zenodo_autopilot")

    bucket = autopilot._guess_bucket_from_name("sample_id", "", "string")

    assert bucket == "Feature of Interest (FOI) - ID"


def test_heuristic_bucket_map_keeps_single_leftmost_foi_id():
    autopilot = __import__("zenodo_autopilot")

    df = pd.DataFrame(
        {
            "site_id": ["A1", "A2"],
            "sample_id": ["S1", "S2"],
            "temperature": [12.5, 13.0],
        }
    )

    buckets = autopilot._heuristic_bucket_map(df)

    assert buckets["site_id"] == "Feature of Interest (FOI) - ID"
    assert buckets["sample_id"] == "Unsorted"


def test_build_bucket_map_with_sorting_page_uses_uom_signals_from_llm_output(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    fake_context_module = types.SimpleNamespace()
    fake_context_module.st = types.SimpleNamespace(session_state={})
    fake_context_module.get_response_OpenAI = lambda prompt: "{}"
    fake_context_module.create_openai_client = lambda api_key: object()
    fake_context_module.generate_UoM_with_LLM = lambda **kwargs: {
        "LOI_method": {"UoM": "degrees Celsius", "UoM - Best guess": "degrees Celsius"},
        "LOI_perc": {"UoM": "percent", "UoM - Best guess": "percent"},
        "Dataset_name": {"UoM": "", "UoM - Best guess": ""},
        "ID": {"UoM": "", "UoM - Best guess": ""},
    }

    original_load_module = autopilot._load_module

    def _patched_load_module(module_name, file_path, *args, **kwargs):
        if file_path == autopilot.CONTEXT_PAGE_FILE:
            return fake_context_module
        return original_load_module(module_name, file_path, *args, **kwargs)

    monkeypatch.setattr(autopilot, "_load_module", _patched_load_module)

    df = pd.DataFrame(
        {
            "Dataset_name": ["example"],
            "LOI_method": [375],
            "LOI_perc": [12.5],
            "ID": ["sample-1"],
        }
    )

    llm_uom = autopilot._generate_llm_uom_for_metadata(
        {"example_table": df},
        api_key="dummy",
        context_text="test context",
    )

    bucket_map = autopilot._build_bucket_map_with_sorting_page(
        "example_table",
        df,
        description_rows=[
            {"name": "LOI_method", "description": "temperature used to measure Loss-on-ignition"},
            {"name": "LOI_perc", "description": "measured soil organic matter content"},
        ],
        uom_rows=llm_uom["example_table"],
        method_rows=[],
    )

    unit_map = {row["name"]: row["unit_symbol"] for row in llm_uom["example_table"]}
    assert unit_map["Dataset_name"] == ""
    assert unit_map["LOI_method"] == "degrees Celsius"
    assert unit_map["LOI_perc"] == "percent"
    assert unit_map["ID"] == ""
    assert bucket_map["Dataset_name"] == "FOI - Attribute"
    assert bucket_map["LOI_method"] == "Observed Property"
    assert bucket_map["LOI_perc"] == "Observed Property"
    assert bucket_map["ID"] == "Feature of Interest (FOI) - ID"


def test_collect_context_text_for_record_reuses_app_context_file_logic(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    fake_input_module = types.SimpleNamespace(
        get_record_id_from_Zenodo_url=lambda url: "123",
        get_metadata_from_Zenodo_id=lambda record_id: {"title": "Study", "description": "<p>Descriptive notes</p>"},
        get_files_URL_from_Zenodo_id=lambda record_id, extensions=None: ["https://example.com/context.pdf"],
        request_file_from_zenodo=lambda url: (types.SimpleNamespace(content=b"pdf-bytes", headers={"Content-Type": "application/pdf"}), "context.pdf", ".pdf"),
        description_to_plain_text=lambda raw: "Descriptive notes",
        filename_from_url=lambda url: "context.pdf",
    )
    fake_context_module = types.SimpleNamespace(
        ContextFile=lambda **kwargs: types.SimpleNamespace(**kwargs),
        read_context_file=lambda contextfile: f"Extracted text for {contextfile.name}",
    )

    monkeypatch.setattr(autopilot, "_load_module", lambda module_name, file_path: fake_input_module if "01_" in str(file_path) else fake_context_module)

    combined = autopilot._collect_context_text_for_record_url("https://zenodo.org/records/123")

    assert "Descriptive notes" in combined
    assert "Extracted text for context.pdf" in combined


def test_resolve_variable_description_fallback_is_empty_without_metadata_description():
    autopilot = __import__("zenodo_autopilot")
    df = pd.DataFrame(
        {
            "temperature_c": [12.5, 13.1],
        }
    )

    description = autopilot._resolve_variable_description(df, "temperature_c")

    assert isinstance(description, str)
    assert description == ""


def test_generate_llm_descriptions_records_prompt_feedback(monkeypatch):
    autopilot = __import__("zenodo_autopilot")

    fake_context_module = types.SimpleNamespace()
    fake_context_module.st = types.SimpleNamespace(session_state={})
    fake_context_module.get_response_OpenAI = lambda prompt: "{}"
    fake_context_module.create_openai_client = lambda api_key: object()
    fake_context_module.generate_descriptions_with_LLM = lambda **kwargs: {
        "temperature_c": "Measured air temperature in degrees Celsius"
    }

    monkeypatch.setattr(autopilot, "_load_module", lambda *args, **kwargs: fake_context_module)

    metadata = {
        "example_table": pd.DataFrame(
            {
                "name": ["temperature_c", "site_id"],
                "description": ["", ""],
            }
        )
    }
    feedback = {}
    rows = autopilot._generate_llm_descriptions_for_metadata(
        metadata,
        api_key="dummy",
        context_text="test context",
        debug_feedback=feedback,
    )

    row_map = {item["name"]: item["description"] for item in rows["example_table"]}
    assert row_map["temperature_c"] == "Measured air temperature in degrees Celsius"
    assert "example_table" in feedback
    assert feedback["example_table"]["llm_result_type"] == "dict"
    assert "temperature_c" in feedback["example_table"]["llm_result_keys"]
    assert "test context" in feedback["example_table"]["full_context_text"]
    resolved = {item["name"]: item["source"] for item in feedback["example_table"]["resolved_rows"]}
    assert resolved["temperature_c"] == "llm"


def test_context_page_loads_under_batch_streamlit_stub():
    autopilot = __import__("zenodo_autopilot")

    module = autopilot._load_module("dataannot_context_page_autopilot", autopilot.CONTEXT_PAGE_FILE)

    assert hasattr(module, "generate_descriptions_with_LLM")
    assert hasattr(module, "from_prompt_to_json")


def test_generate_llm_descriptions_keeps_debug_feedback_when_loading_fails(monkeypatch):
    autopilot = __import__("zenodo_autopilot")
    metadata = {
        "example_table": pd.DataFrame({
            "name": ["temperature_c", "site_id"],
            "description": ["", ""],
        })
    }
    feedback = {}

    monkeypatch.setattr(autopilot, "_load_module", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")))

    rows = autopilot._generate_llm_descriptions_for_metadata(
        metadata,
        api_key="dummy",
        context_text="test context",
        debug_feedback=feedback,
    )

    assert rows == {}
    assert "example_table" in feedback
    assert feedback["example_table"]["status"] == "skipped"
    assert feedback["example_table"]["reason"] == "context_module_load_failed"


