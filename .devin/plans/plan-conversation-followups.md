# Plan: Conversation Follow-Up Questions (v2 — Simplified)

## Core Principle

**Conversation context resolves what the doctor means. Fresh evidence determines what the system says.**

Every turn has two separate phases:
1. **Resolution**: Use conversation history to understand the new question → produce a resolved question
2. **Answering**: Retrieve fresh evidence for the resolved question → generate a standalone answer

After resolution, the new answer stands on its own. No previous prose enters the answer prompt.

## What Changed from v1

- **Removed**: Previous Q&A in the answer-generation prompt (contamination risk)
- **Removed**: Heuristic-gated contextual parsing (always resolve if conversation exists)
- **Added**: `context_status` field: `standalone | resolved | ambiguous | missing_context`
- **Added**: Ambiguity → targeted clarification, not guessing
- **Simplified**: Service API — `answer_question(question, conversation_id=None)` loads context from DB itself
- **Simplified**: UI state — only `conversation_id` in session state, thread loaded from DB

## Design

### 1. Always-on context resolution

If a conversation exists, always give the resolver the last few turns. The resolver returns:

```json
{
  "raw_question": "What about its side effects?",
  "resolved_question": "What are the side effects of amisulpride?",
  "context_status": "resolved",
  "is_follow_up": true,
  "inherited_entities": {
    "drug_names": ["amisulpride"]
  },
  "question_type": "side_effects"
}
```

`context_status` values:
- `standalone` — no conversation context, or question doesn't reference prior turns
- `resolved` — context successfully applied
- `ambiguous` — cannot resolve (e.g., "the other one" with two drugs discussed) → ask clarification
- `missing_context` — references prior context but none available

### 2. No previous prose in answer generation

The answer generator receives:
- Resolved question
- Current structured understanding
- Freshly retrieved evidence
- Optionally: "This is turn N in a conversation." (non-medical instruction only)

**Invariant**: Every medical statement in turn N must be supported by evidence retrieved for turn N.

### 3. Standalone answers

Answers should NOT say "As discussed above..." — each answer is independently understandable, validatable, exportable, and reviewable.

### 4. Database changes

Add to `answer_traces`:
- `turn_number: int | None`
- `resolved_question: str | None`

Add unique constraint: `UNIQUE(conversation_id, turn_number)`

Store resolver result (context_status, inherited_entities, is_follow_up) in the existing `question_understanding` JSON.

No separate conversation-state table. State is reconstructed from previous traces.

### 5. Service API

```python
answer_question(
    question: str,
    conversation_id: int | None = None,
) -> dict
```

Behavior:
- If `conversation_id` is None → create a new conversation
- If present → load last 3 traces from DB for context
- Resolve the new question using context
- If `ambiguous` → return clarification result without retrieval
- Otherwise → retrieve fresh evidence, generate answer, save as next turn

### 6. Repository

```python
list_traces(conversation_id, limit=None) -> list[dict]
get_recent_context(conversation_id, max_turns=3) -> list[ConversationTurn]
```

`ConversationTurn` = `{question, resolved_question, understanding, answer_summary, drug_names}`

### 7. Streamlit state

Only `conversation_id` in session state. On every rerun, load the thread from PostgreSQL.

Flow:
- No conversation → first question creates one → store ID
- Follow-ups reuse that ID
- "New conversation" removes the ID → next question creates new one

### 8. Context for resolver

Last 3 turns, plus a small structured state:
```json
{
  "active_drugs": ["amisulpride"],
  "recent_question_types": ["mechanism", "side_effects"]
}
```

This is cheaper and more useful than sending full previous answers.

## Implementation Steps

### Step 1: DB schema
- `tables.py`: add `turn_number`, `resolved_question` to `AnswerTrace`, unique constraint
- `reset_db.py`: add columns to init

### Step 2: Context resolver
- `question_parser.py`: add `parse_question_with_context(question, recent_context)`
- Returns resolved question + context_status + inherited entities
- If ambiguous, returns `context_status: "ambiguous"` with a clarification question

### Step 3: Repository
- `conversations.py`: add `list_traces()`, `get_recent_context()`, update `save_answer_trace()` for turn_number + resolved_question

### Step 4: Answer service
- `answer_service.py`: update `answer_question()` to accept `conversation_id`
- Load context from DB, resolve, retrieve fresh, generate, save with turn_number
- If ambiguous → return clarification result (no retrieval)

### Step 5: Chat UI
- `1_Chat.py`: conversation_id in session state, thread loaded from DB, follow-up input, new conversation button

### Step 6: Tests (5 key tests)
1. **Pronoun resolution**: "What about its side effects?" inherits amisulpride
2. **Topic switch**: "What about aripiprazole?" switches active drug
3. **Ambiguity**: After two drugs, "Which one causes more weight gain?" → clarification
4. **Fresh retrieval**: Side-effects turn retrieves side-effect evidence, not mechanism evidence
5. **No contamination**: Incorrect statement in previous answer fixture is NOT repeated in new answer

## What Does NOT Change
- Evidence grounding, entailment, sufficiency, citations — all per-turn
- Feedback — per-trace
- Trust pipeline — runs independently per turn
