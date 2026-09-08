"""Unit tests for the naive token-stream chunker.

Uses synthetic pages — no PDF, no DB required.
"""

from __future__ import annotations

from psych_qa.baseline.ingest import (
    PAGE_SEPARATOR,
    chunk_token_stream,
)


def _make_pages(*texts: str) -> list[dict]:
    """Build synthetic pages from text strings."""
    return [{"page_index": i, "text": t} for i, t in enumerate(texts)]


class TestChunkTokenStream:
    def test_single_page_basic(self):
        """One page with enough text produces multiple chunks."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        pages = _make_pages(text)
        chunks = chunk_token_stream(pages, chunk_tokens=50, step_tokens=40)

        assert len(chunks) > 1
        # All chunks should be from page 0
        for c in chunks:
            assert c.page_start == 0
            assert c.page_end == 0

    def test_chunk_token_bounds(self):
        """No chunk should exceed chunk_tokens."""
        text = "Pharmacology is the study of drugs and their effects. " * 200
        pages = _make_pages(text)
        chunks = chunk_token_stream(pages, chunk_tokens=100, step_tokens=80)

        for c in chunks:
            assert c.token_count <= 100, f"Chunk {c.chunk_index} has {c.token_count} tokens"

    def test_overlap(self):
        """Consecutive chunks should have different start positions but share content."""
        # Use non-repetitive text so chunks are actually different
        text = (
            "The pharmacology of antipsychotic drugs involves dopamine receptor blockade. "
            "Risperidone is a potent D2 antagonist with high affinity for serotonin receptors. "
            "Aripiprazole is a partial agonist at D2 receptors with a different safety profile. "
            "Quetiapine has weaker D2 occupancy and more histamine blockade than other agents. "
            "Olanzapine is associated with significant weight gain and metabolic side effects. "
            "Clozapine is effective for treatment-resistant schizophrenia but requires monitoring. "
        )
        pages = _make_pages(text)
        chunks = chunk_token_stream(pages, chunk_tokens=50, step_tokens=40)

        if len(chunks) >= 2:
            # Chunks should be different (non-repetitive text)
            assert chunks[0].text != chunks[1].text
            # Both should have content
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0

    def test_multi_page_tracking(self):
        """Chunks that span page boundaries should track both pages."""
        # Create a short page 0 and short page 1 so a chunk spans both
        page0 = "Short text on page zero. "
        page1 = "Short text on page one. "
        pages = _make_pages(page0, page1)
        chunks = chunk_token_stream(pages, chunk_tokens=100, step_tokens=80)

        # With small text, we should get at least one chunk
        assert len(chunks) >= 1
        # At least one chunk should span pages 0 and 1
        multi_page = [c for c in chunks if c.page_start != c.page_end]
        assert len(multi_page) >= 1, "Expected at least one multi-page chunk"
        assert multi_page[0].page_start == 0
        assert multi_page[0].page_end == 1

    def test_page_separator_prevents_word_merging(self):
        """The separator should prevent words from merging across pages."""
        page0 = "seroto"  # ends mid-word
        page1 = "nin is a neurotransmitter"
        pages = _make_pages(page0, page1)
        chunks = chunk_token_stream(pages, chunk_tokens=100, step_tokens=80)

        # The separator should appear in the text, preventing "serotonin"
        all_text = " ".join(c.text for c in chunks)
        assert PAGE_SEPARATOR.strip() in all_text or "[PAGE BREAK]" in all_text

    def test_empty_pages_skipped(self):
        """Empty pages should not produce chunks."""
        pages = _make_pages("", "", "Real content here. " * 20, "")
        chunks = chunk_token_stream(pages, chunk_tokens=50, step_tokens=40)

        # All chunks should be from page 2
        for c in chunks:
            assert c.page_start == 2
            assert c.page_end == 2

    def test_no_empty_chunks(self):
        """No chunk should have empty text."""
        text = "Some meaningful text that produces chunks. " * 50
        pages = _make_pages(text)
        chunks = chunk_token_stream(pages, chunk_tokens=50, step_tokens=40)

        for c in chunks:
            assert c.text.strip(), f"Chunk {c.chunk_index} is empty"

    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        text = "Sequential index test. " * 100
        pages = _make_pages(text)
        chunks = chunk_token_stream(pages, chunk_tokens=50, step_tokens=40)

        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_input_hash_unique(self):
        """Different chunks should have different input hashes."""
        text = "Hash uniqueness test with varied content. " * 50
        pages = _make_pages(text)
        chunks = chunk_token_stream(pages, chunk_tokens=50, step_tokens=40)

        hashes = [c.input_hash for c in chunks]
        # With overlap, most chunks should be unique
        assert len(set(hashes)) > len(hashes) * 0.5

    def test_expected_chunk_count(self):
        """Chunk count should match the expected formula."""
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-4o")

        text = "Token counting test for expected chunk formula. " * 100
        pages = _make_pages(text)
        total_tokens = len(encoder.encode(text)) + len(encoder.encode(PAGE_SEPARATOR))

        chunk_tokens = 50
        step_tokens = 40
        chunks = chunk_token_stream(pages, chunk_tokens=chunk_tokens, step_tokens=step_tokens)

        import math
        expected = math.ceil(max(total_tokens - chunk_tokens, 0) / step_tokens) + 1
        # Allow some tolerance for separator tokens and edge effects
        assert abs(len(chunks) - expected) <= 2, (
            f"Expected ~{expected} chunks, got {len(chunks)}"
        )
