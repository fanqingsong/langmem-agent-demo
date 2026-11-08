from pydantic import BaseModel, Field

INITIAL_INSTRUCTIONS = """You are a helpful assistant.

Explain concepts clearly. Use what you know about the user when it helps.
Answer the question first, then add only the detail that is needed.
"""


class WorkingContext(BaseModel):
    """Active context for the current conversation thread."""

    goal: str = Field(
        ...,
        description="What the user is trying to accomplish in this thread.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints that apply only to the current task.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions still unresolved in this thread.",
    )
    recent_decisions: list[str] = Field(
        default_factory=list,
        description="Choices made so far in this thread.",
    )


class SemanticFact(BaseModel):
    """A durable fact about the user or their world."""

    content: str = Field(
        ...,
        description="The fact itself, stated so it can be recalled later.",
    )
    category: str = Field(
        ...,
        description="One of: identity, preference, knowledge, other.",
    )


class Episode(BaseModel):
    """Write the episode from the perspective of the agent within it.

    Use the benefit of hindsight to record the memory, saving the agent's key
    internal thought process so it can learn over time.
    """

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description=(
            "Internal reasoning process and observations of the agent in the episode "
            'that let it arrive at the correct action and result. "I ..."'
        ),
    )
    action: str = Field(
        ...,
        description=(
            "What was done, how, and in what format. Include whatever is salient "
            "to the success of the action. I ..."
        ),
    )
    result: str = Field(
        ...,
        description=(
            "Outcome and retrospective. What did you do well? What could you do "
            "better next time? I ..."
        ),
    )
