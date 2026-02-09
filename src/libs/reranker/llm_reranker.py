import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM
from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class LLMReranker(BaseReranker):
    """
    Reranker implementation that uses an LLM to reorder candidates.
    It reads a prompt template from a file, injects the query and candidates,
    and parses the LLM's response to determine the new order.
    """

    def __init__(self, llm: BaseLLM, prompt_path: str = "config/prompts/rerank.txt"):
        """
        Initialize the LLM Reranker.

        Args:
            llm: An instance of BaseLLM to perform the reranking.
            prompt_path: Path to the prompt template file.
        """
        self.llm = llm
        self.prompt_path = prompt_path
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from file or return default fallback."""
        try:
            return Path(self.prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                f"Failed to load rerank prompt from {self.prompt_path}: {e}. Using default."
            )
            return (
                "Rank the following passages based on relevance to the query: {query}.\n"
                "Passages:\n{candidates}\n"
                "Return a JSON list of indices of the top passages, e.g. [0, 2, 1]."
            )

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        """
        Rerank candidates using LLM.

        Args:
            query: The search query.
            candidates: List of candidate objects.
            top_k: Number of results to return.
            trace: Trace context (unused in this implementation, passed for interface compatibility).

        Returns:
            Reordered list of candidates.
        """
        if not candidates:
            return []

        # Format candidates
        # Try to extract text content: 'page_content', 'text', or str()
        candidate_texts = []
        for i, cand in enumerate(candidates):
            # Safe text extraction
            text = ""
            if hasattr(cand, "page_content"):
                text = cand.page_content
            elif hasattr(cand, "text"):
                text = cand.text
            elif isinstance(cand, dict) and "text" in cand:
                text = cand["text"]
            else:
                text = str(cand)

            # Truncate text to avoid excessive token usage (heuristic)
            text = text[:300].replace("\n", " ")
            candidate_texts.append(f"[{i}] {text}")

        candidates_str = "\n".join(candidate_texts)

        # Construct prompt
        try:
            prompt = self._prompt_template.format(
                query=query, candidates=candidates_str
            )
        except KeyError as e:
            logger.error(
                f"Prompt template format error: {e}. Missing keys in template."
            )
            # Fallback to simple concatenation if template is broken
            prompt = f"Query: {query}\nCandidates:\n{candidates_str}\nRank them as JSON list of indices."

        try:
            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages)

            # Parse response
            ranked_indices = self._parse_indices(response)

            if not ranked_indices:
                logger.warning(
                    "LLM returned no valid indices. Returning original order."
                )
                if top_k:
                    return candidates[:top_k]
                return candidates

            # Reorder
            # Filter indices to ensure they are valid
            valid_indices = [i for i in ranked_indices if 0 <= i < len(candidates)]

            # Create result list
            ranked_results = [candidates[i] for i in valid_indices]

            # Append missing candidates at the end (if any) to preserve recall
            seen_indices = set(valid_indices)
            for i in range(len(candidates)):
                if i not in seen_indices:
                    ranked_results.append(candidates[i])

            if top_k:
                ranked_results = ranked_results[:top_k]

            return ranked_results

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Fallback: return original order
            if top_k:
                return candidates[:top_k]
            return candidates

    def _parse_indices(self, response: str) -> List[int]:
        """Parse JSON list of indices from LLM response."""
        try:
            # Clean up response (remove markdown code blocks if present)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]

            # Try to find something that looks like a JSON list
            match = re.search(r"\[[\d,\s]+\]", clean_response)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.debug(
                f"Failed to parse indices from response: {response[:100]}... Error: {e}"
            )
        return []
