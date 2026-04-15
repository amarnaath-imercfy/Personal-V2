"""RAG tools for the Personal Tutor agent."""

import logging
from vertexai.preview import rag as vertex_rag
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

CORPUS_NAME = "projects/aitrack-29a9e/locations/us-east4/ragCorpora/2666130979403333632"
MAX_CHUNK_CHARS = 400  # truncate each chunk to keep tokens low


def query_lesson(query: str, tool_context: ToolContext) -> dict:
    """Searches the lesson file to answer a student question.

    Args:
        query: The student's question.

    Returns:
        dict with retrieved lesson content.
    """
    file_id = tool_context.state.get("file_id", "")
    resource = vertex_rag.RagResource(
        rag_corpus=CORPUS_NAME,
        rag_file_ids=[file_id] if file_id else [],
    )
    try:
        response = vertex_rag.retrieval_query(
            text=query,
            rag_resources=[resource],
            similarity_top_k=5,
        )
    except Exception as e:
        logger.error("query_lesson error: %s", e)
        return {"status": "error", "message": str(e)}

    if not response.contexts.contexts:
        return {"status": "no_results"}

    chunks = [c.text[:MAX_CHUNK_CHARS] for c in response.contexts.contexts]
    return {"status": "success", "content": chunks}


