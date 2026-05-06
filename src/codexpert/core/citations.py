from codexpert.core.types import Citation


def format_citation(citation: Citation) -> str:
    start_line = max(1, citation.start_line)
    end_line = max(start_line, citation.end_line)
    return f"[{citation.path}:{start_line}-{end_line}]"


def format_citations(citations: list[Citation]) -> list[str]:
    return [format_citation(citation) for citation in citations]
