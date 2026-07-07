import logging

from app.models import RagAnswerStrategy, RagSource, RetrievalSearchResult
from app.services.llm import generate_grounded_answer, is_llm_enabled


logger = logging.getLogger(__name__)


RAG_REFUSAL_ANSWER = "资料不足，无法基于已上传文档回答该问题。"
RAG_ANSWER_STRATEGY = RagAnswerStrategy.extractive_context_v1


def build_rag_prompt(query: str, sources: list[RagSource]) -> str:
    context_blocks = []
    for source in sources:
        label = f"[source {source.rank}]"
        location = []
        if source.source_file:
            location.append(f"file={source.source_file}")
        if source.chunk_index is not None:
            location.append(f"chunk_index={source.chunk_index}")
        if source.page is not None:
            location.append(f"page={source.page}")
        location_text = ", ".join(location) if location else "unknown source"
        context_blocks.append(f"{label} {location_text}\n{source.content}")

    context = "\n\n".join(context_blocks)
    return (
        "你是一个严格基于资料回答的 RAG 助手。\n"
        "规则：\n"
        "1. 只能使用给定资料回答。\n"
        "2. 如果资料不足，必须回答资料不足，不能编造。\n"
        "3. 每个关键结论后使用 [source n] 标注来源。\n"
        "4. 不允许引用未提供的 source。\n"
        "5. 回答后必须能够对应到 sources。\n\n"
        f"用户问题：{query}\n\n"
        f"资料：\n{context}"
    )


def build_rag_sources(
    retrieval_results: list[RetrievalSearchResult],
    max_context_chars: int,
) -> list[RagSource]:
    sources = []
    remaining_chars = max(max_context_chars, 0)

    for result in retrieval_results:
        if remaining_chars == 0:
            break

        content = (result.content or "").rstrip()
        if len(content) > remaining_chars:
            content = content[:remaining_chars].rstrip()
        if not content:
            continue

        metadata = dict(result.metadata or {})
        chunk_id = metadata.get("chunk_id")
        if chunk_id is None:
            logger.warning("Missing chunk_id in metadata for source rank=%s", result.rank)
            chunk_id = metadata.get("id") or metadata.get("digest")
        sources.append(
            RagSource(
                rank=result.rank,
                source_file=result.source_file,
                chunk_index=result.chunk_index,
                chunk_id=chunk_id,
                page=result.page,
                score=result.score,
                score_type=result.score_type,
                content=content,
                metadata=metadata,
            )
        )
        remaining_chars -= len(content)

    return sources


def generate_extractive_answer(query: str, sources: list[RagSource]) -> tuple[str, bool]:
    if not sources:
        return RAG_REFUSAL_ANSWER, True

    evidence_blocks = [
        f"[source {source.rank}]\n{source.content}" for source in sources
    ]
    answer = (
        "根据已检索到的资料，相关依据如下：\n\n"
        + "\n\n".join(evidence_blocks)
        + "\n\n以上回答仅基于返回的 sources；如果需要更完整的自然语言总结，"
        "后续可以把同一份 context 交给 LLM 生成。"
    )
    return answer, False


def _looks_like_refusal(answer: str) -> bool:
    normalized_answer = answer.lower()
    refusal_markers = [
        "资料不足",
        "无法回答",
        "不能回答",
        "无法基于",
        "insufficient",
        "not enough information",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "cannot determine",
        "cannot be determined",
        "can't determine",
        "not provided",
        "do not include",
        "does not include",
        "do not contain",
        "does not contain",
    ]
    return any(marker in normalized_answer for marker in refusal_markers)


def build_rag_response(
    query: str,
    file_id: str,
    retrieval_results: list[RetrievalSearchResult],
    max_context_chars: int,
    use_llm: bool | None = None,
):
    sources = build_rag_sources(retrieval_results, max_context_chars)
    prompt = build_rag_prompt(query, sources)
    should_use_llm = use_llm if use_llm is not None else is_llm_enabled()

    if not sources:
        answer, refusal = generate_extractive_answer(query, sources)
        answer_strategy = RagAnswerStrategy.no_context_refusal
    elif should_use_llm:
        try:
            answer = generate_grounded_answer(prompt)
            refusal = _looks_like_refusal(answer)
            answer_strategy = (
                RagAnswerStrategy.low_confidence_refusal
                if refusal
                else RagAnswerStrategy.llm_grounded_context_v1
            )
        except Exception as exc:
            logger.warning("LLM grounded answer failed; falling back to extractive answer: %s", exc)
            answer, refusal = generate_extractive_answer(query, sources)
            answer_strategy = RAG_ANSWER_STRATEGY
    else:
        answer, refusal = generate_extractive_answer(query, sources)
        answer_strategy = RAG_ANSWER_STRATEGY

    return {
        "query": query,
        "file_id": file_id,
        "answer": answer,
        "refusal": refusal,
        "answer_strategy": answer_strategy,
        "used_context_count": len(sources),
        "sources": sources,
    }
