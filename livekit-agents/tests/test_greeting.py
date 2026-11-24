import pytest

from hn_agent.config import AgentConfig, DisclosureSettings, parse_agent_config
from hn_agent.greeting import (
    DEFAULT_PROMPT,
    DISCLOSURE_RULE,
    build_greeting,
    build_instructions,
    disclosure_line,
)

EL = "Είμαι ψηφιακός βοηθός."
EN = "I am a digital assistant."


@pytest.mark.parametrize(
    ("language", "first_words"),
    [("el", EL), ("en", EN), ("auto", f"{EL} {EN}")],
)
def test_default_greeting_starts_with_the_disclosure(language: str, first_words: str) -> None:
    config = parse_agent_config({"language": language}).config
    greeting = build_greeting(config)
    assert greeting.startswith(first_words)
    assert len(greeting) > len(first_words)


def test_custom_greeting_gets_the_disclosure_in_front() -> None:
    config = AgentConfig(language="el", greeting="Καλώς ήρθατε στο ιατρείο.")
    assert build_greeting(config) == f"{EL} Καλώς ήρθατε στο ιατρείο."


def test_disclosure_is_not_repeated_when_the_greeting_has_it() -> None:
    config = AgentConfig(language="el", greeting="Γεια σας! ΕΙΜΑΙ ψηφιακος βοηθος, πώς να βοηθήσω;")
    assert build_greeting(config) == config.greeting


def test_custom_disclosure_text() -> None:
    config = AgentConfig(
        language="en",
        greeting="How can I help?",
        ai_disclosure=DisclosureSettings(enabled=True, text="You are talking to an AI."),
    )
    assert build_greeting(config) == "You are talking to an AI. How can I help?"


def test_disclosure_off_leaves_the_greeting_alone() -> None:
    config = AgentConfig(
        language="en", greeting="Hi there!", ai_disclosure=DisclosureSettings(enabled=False)
    )
    assert disclosure_line(config) is None
    assert build_greeting(config) == "Hi there!"


def test_empty_greeting_still_says_the_disclosure() -> None:
    assert build_greeting(AgentConfig(language="en", greeting="")) == EN


def test_empty_greeting_without_disclosure_says_nothing() -> None:
    config = AgentConfig(greeting="", ai_disclosure=DisclosureSettings(enabled=False))
    assert build_greeting(config) == ""


def test_instructions_carry_language_and_disclosure_rules() -> None:
    text = build_instructions(AgentConfig(language="el"))
    assert text.startswith(DEFAULT_PROMPT)
    assert "Always answer in Greek" in text
    assert DISCLOSURE_RULE in text


def test_custom_prompt_without_disclosure_rule() -> None:
    config = AgentConfig(
        prompt="Be a pirate.", language="en", ai_disclosure=DisclosureSettings(enabled=False)
    )
    text = build_instructions(config)
    assert text.startswith("Be a pirate.")
    assert "Always answer in English" in text
    assert DISCLOSURE_RULE not in text
