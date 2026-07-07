from scripts.hotpotqa_evaluate_recall import (
    compute_recall_at_k,
    parse_query_multiple_response,
)
from scripts.hotpotqa_import_subset import index_results_by_file_id
from scripts.evaluate_hotpotqa_retrieval import (
    build_relevance_maps,
    build_global_search_payload,
    candidate_mode_for_args,
    compute_official_metrics_fallback,
    compute_project_metrics,
    parse_retrieval_search_response,
)
from scripts.evaluate_hotpotqa_rag_answers import (
    answer_has_source_marker,
    build_global_rag_payload,
    compute_rag_answer_metrics,
    parse_rag_chat_global_response,
)
from scripts.evaluate_rag_refusals import (
    build_refusal_payload,
    compute_refusal_metrics,
    parse_refusal_response,
)
from scripts.hotpotqa_prepare_subset import (
    build_file_id,
    load_qrels,
    read_rows_by_ids_with_negatives,
    select_queries,
)


def test_build_file_id_sanitizes_path_separators():
    assert build_file_id("abc/def\\ghi", "hotpotqa-") == "hotpotqa-abc_def_ghi"


def test_load_qrels_supports_tsv_with_iteration_column(tmp_path):
    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text(
        "query-id\tcorpus-id\tscore\nq1\tdoc1\t1\nq1\tdoc2\t0\n",
        encoding="utf-8",
    )

    assert load_qrels(qrels_path) == {"q1": {"doc1": 1}}


def test_parse_query_multiple_response_extracts_file_ids():
    body = [
        [{"page_content": "a", "metadata": {"file_id": "hotpotqa-doc1"}}, 0.1],
        [{"page_content": "b", "metadata": {"file_id": "hotpotqa-doc2"}}, 0.2],
    ]

    assert parse_query_multiple_response(body) == ["hotpotqa-doc1", "hotpotqa-doc2"]


def test_parse_retrieval_search_response_extracts_file_ids_and_scores():
    body = {
        "results": [
            {
                "rank": 1,
                "file_id": "hotpotqa-doc1",
                "score": 0.1,
                "metadata": {"file_id": "hotpotqa-doc1"},
            },
            {
                "rank": 2,
                "score": 0.2,
                "metadata": {"file_id": "hotpotqa-doc2"},
            },
        ]
    }

    parsed = parse_retrieval_search_response(body)

    assert [item["file_id"] for item in parsed] == ["hotpotqa-doc1", "hotpotqa-doc2"]
    assert parsed[0]["score_for_eval"] == -0.1
    assert parsed[1]["score_for_eval"] == -0.2


def test_global_search_payload_can_include_file_id_prefix():
    payload = build_global_search_payload(
        "query text",
        k=10,
        entity_id="public",
        file_id_prefix="hotpotqa-day11-",
    )

    assert payload == {
        "query": "query text",
        "k": 10,
        "entity_id": "public",
        "file_id_prefix": "hotpotqa-day11-",
    }


def test_global_rag_payload_can_include_llm_and_file_id_prefix():
    payload = build_global_rag_payload(
        "query text",
        k=5,
        max_context_chars=1600,
        use_llm=True,
        entity_id="public",
        file_id_prefix="hotpotqa-day11-",
    )

    assert payload == {
        "query": "query text",
        "k": 5,
        "max_context_chars": 1600,
        "use_llm": True,
        "entity_id": "public",
        "file_id_prefix": "hotpotqa-day11-",
    }


def test_parse_rag_chat_global_response_extracts_source_file_ids_and_citations():
    body = {
        "answer": "Sergei Tokarev worked at Moscow State University. [source 1]",
        "refusal": False,
        "answer_strategy": "llm_grounded_context_v1",
        "sources": [
            {"metadata": {"file_id": "hotpotqa-day11-36722175"}},
            {"metadata": {"file_id": "hotpotqa-day11-36722175"}},
            {"metadata": {"file_id": "hotpotqa-day11-374544"}},
        ],
    }

    parsed = parse_rag_chat_global_response(body)

    assert parsed["answer_has_citation"] is True
    assert parsed["source_file_ids"] == [
        "hotpotqa-day11-36722175",
        "hotpotqa-day11-374544",
    ]
    assert parsed["source_count"] == 3


def test_answer_source_marker_detection_is_case_insensitive():
    assert answer_has_source_marker("Answer [Source 2]") is True
    assert answer_has_source_marker("Answer without marker") is False


def test_compute_rag_answer_metrics_counts_support_citations_and_strategies():
    details = [
        {
            "relevant_file_ids": ["doc1", "doc2"],
            "source_file_ids": ["doc1", "doc2", "docX"],
            "answer_has_citation": True,
            "refusal": False,
            "answer_strategy": "llm_grounded_context_v1",
            "source_count": 3,
        },
        {
            "relevant_file_ids": ["doc3", "doc4"],
            "source_file_ids": ["doc9"],
            "answer_has_citation": False,
            "refusal": True,
            "answer_strategy": "no_context_refusal",
            "source_count": 0,
        },
    ]

    metrics = compute_rag_answer_metrics(details)

    assert metrics["retrieved_support_hit_rate"] == 0.5
    assert metrics["all_support_in_sources_rate"] == 0.5
    assert metrics["citation_marker_rate"] == 0.5
    assert metrics["refusal_rate"] == 0.5
    assert metrics["llm_grounded_strategy_rate"] == 0.5
    assert metrics["no_context_refusal_rate"] == 0.5
    assert metrics["average_source_count"] == 1.5


def test_build_refusal_payload_uses_case_prefix_before_default_prefix():
    payload = build_refusal_payload(
        {
            "query": "What is the API key?",
            "file_id_prefix": "__no-context-",
        },
        k=5,
        max_context_chars=1800,
        use_llm=True,
        default_file_id_prefix="hotpotqa-day11-",
        entity_id="public",
    )

    assert payload == {
        "query": "What is the API key?",
        "k": 5,
        "max_context_chars": 1800,
        "use_llm": True,
        "file_id_prefix": "__no-context-",
        "entity_id": "public",
    }


def test_parse_refusal_response_extracts_strategy_sources_and_citation():
    parsed = parse_refusal_response(
        {
            "answer": "资料不足，无法回答。[source 1]",
            "refusal": True,
            "answer_strategy": "low_confidence_refusal",
            "sources": [
                {"metadata": {"file_id": "hotpotqa-day11-1"}},
                {"metadata": {"file_id": "hotpotqa-day11-1"}},
                {"metadata": {"file_id": "hotpotqa-day11-2"}},
            ],
        }
    )

    assert parsed["refusal"] is True
    assert parsed["answer_strategy"] == "low_confidence_refusal"
    assert parsed["source_count"] == 3
    assert parsed["source_file_ids"] == ["hotpotqa-day11-1", "hotpotqa-day11-2"]
    assert parsed["answer_has_citation"] is True


def test_compute_refusal_metrics_counts_success_and_unsafe_answers():
    details = [
        {
            "category": "no_context",
            "expected_refusal": True,
            "refusal": True,
            "answer_has_citation": False,
            "source_count": 0,
            "raw_response_error": None,
        },
        {
            "category": "out_of_scope",
            "expected_refusal": True,
            "refusal": False,
            "answer_has_citation": True,
            "source_count": 3,
            "raw_response_error": None,
        },
    ]

    metrics = compute_refusal_metrics(details)

    assert metrics["case_count"] == 2
    assert metrics["expected_refusal_count"] == 2
    assert metrics["refusal_success_rate"] == 0.5
    assert metrics["unsafe_answer_rate"] == 0.5
    assert metrics["citation_marker_rate"] == 0.5
    assert metrics["average_source_count"] == 1.5
    assert metrics["category_metrics"]["no_context"]["refusal_success_rate"] == 1.0
    assert metrics["category_metrics"]["out_of_scope"]["refusal_success_rate"] == 0.0


def test_candidate_mode_reports_prefix_scoped_global_search():
    class Args:
        global_search = True
        global_file_id_prefix = "hotpotqa-day11-"

    assert candidate_mode_for_args(Args()) == "global_vector_search_prefix"


def test_compute_recall_at_k_counts_query_level_hits():
    retrieved = {
        "q1": ["doc2", "doc1"],
        "q2": ["doc3"],
    }
    relevant = {
        "q1": {"doc1"},
        "q2": {"doc4"},
    }

    assert compute_recall_at_k(retrieved, relevant, [1, 2]) == {
        "recall@1": 0.0,
        "recall@2": 0.5,
    }


def test_prepare_subset_can_include_negative_distractors(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_path = tmp_path / "corpus.parquet"
    table = pa.table(
        {
            "_id": ["doc1", "doc2", "doc3", "doc4"],
            "title": ["relevant 1", "negative 1", "relevant 2", "negative 2"],
            "text": ["a", "b", "c", "d"],
        }
    )
    pq.write_table(table, corpus_path)

    relevant, negatives = read_rows_by_ids_with_negatives(
        corpus_path,
        {"doc1", "doc3"},
        negative_limit=1,
        negative_strategy="deterministic",
    )

    assert [row["_id"] for row in relevant] == ["doc1", "doc3"]
    assert [row["_id"] for row in negatives] == ["doc2"]


def test_prepare_subset_random_query_sampling_is_seeded():
    queries = [{"_id": f"q{i}", "text": str(i)} for i in range(10)]
    qrels = {f"q{i}": {"doc": 1} for i in range(10)}

    first = select_queries(queries, qrels, 4, "random", seed=7)
    second = select_queries(queries, qrels, 4, "random", seed=7)

    assert [row["_id"] for row in first] == [row["_id"] for row in second]
    assert [row["_id"] for row in first] != ["q0", "q1", "q2", "q3"]


def test_prepare_subset_random_negatives_use_reservoir_sampling(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_path = tmp_path / "corpus.parquet"
    table = pa.table(
        {
            "_id": [f"doc{i}" for i in range(20)],
            "title": [f"title {i}" for i in range(20)],
            "text": [f"text {i}" for i in range(20)],
        }
    )
    pq.write_table(table, corpus_path)

    relevant, negatives = read_rows_by_ids_with_negatives(
        corpus_path,
        {"doc1", "doc3"},
        negative_limit=5,
        negative_strategy="random",
        seed=42,
    )

    negative_ids = [row["_id"] for row in negatives]
    assert [row["_id"] for row in relevant] == ["doc1", "doc3"]
    assert len(negative_ids) == 5
    assert len(negative_ids) == len(set(negative_ids))
    assert not {"doc1", "doc3"}.intersection(negative_ids)


def test_project_metrics_separate_hit_recall_all_support_and_mrr():
    ranked = {
        "q1": ["doc1", "docX", "doc2"],
        "q2": ["docZ", "docY", "doc3"],
    }
    relevant = {
        "q1": {"doc1", "doc2"},
        "q2": {"doc3", "doc4"},
    }

    metrics = compute_project_metrics(ranked, relevant, [1, 3])

    assert metrics["hit@1"] == 0.5
    assert metrics["recall@1"] == 0.25
    assert metrics["all_support@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["recall@3"] == 0.75
    assert metrics["all_support@3"] == 0.5
    assert metrics["mrr@3"] == (1.0 + 1.0 / 3.0) / 2.0


def test_official_metrics_fallback_uses_file_id_qrels():
    qrels_rows = [
        {"query_id": "q1", "file_id": "doc1", "score": 1},
        {"query_id": "q1", "file_id": "doc2", "score": 1},
    ]
    qrels_by_query, _ = build_relevance_maps(qrels_rows)
    metrics = compute_official_metrics_fallback(
        qrels_by_query, {"q1": ["doc1", "docX", "doc2"]}, [1, 3]
    )

    assert metrics["P_1"] == 1.0
    assert metrics["recall_1"] == 0.5
    assert metrics["recall_3"] == 1.0
    assert 0.0 < metrics["ndcg_cut_3"] <= 1.0


def test_import_report_results_can_be_indexed_for_resume():
    report = {
        "results": [
            {"file_id": "doc1", "ok": True},
            {"file_id": "doc2", "ok": False},
            {"file_id": 3, "ok": True},
        ]
    }

    indexed = index_results_by_file_id(report)

    assert indexed["doc1"]["ok"] is True
    assert indexed["doc2"]["ok"] is False
    assert indexed["3"]["ok"] is True
