import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.rag.router import QueryRouter, _detect_lang, _guess_query_type


def run(coro):
    return asyncio.run(coro)


def test_detect_lang_hindi():
    # Devanagari covers hi/mr/ne/sa — ambiguous by script, so None (STT or
    # full-pool search decides).
    assert _detect_lang("गंगा नदी कहाँ है?") is None


def test_detect_lang_tamil():
    assert _detect_lang("இந்தியாவின் தலைநகர் என்ன?") == "ta"


def test_detect_lang_bengali_block_none():
    # Bengali/Assamese share the U+0980-U+09FF block — ambiguous.
    assert _detect_lang("কৰ্পোৰেচন কি?") is None


def test_detect_lang_english_none():
    assert _detect_lang("Where is the Ganges river?") is None


def test_guess_query_type_number():
    assert _guess_query_type("How many states does India have?") == "NUMERIC"


def test_guess_query_type_person():
    assert _guess_query_type("Who founded the Mughal Empire?") == "PERSON"


def test_guess_query_type_location():
    assert _guess_query_type("Where is the Taj Mahal?") == "LOCATION"


def test_guess_query_type_entity():
    assert _guess_query_type("What is the capital of India?") == "ENTITY"


def test_route_graph_hint():
    router = QueryRouter(get_settings(), client=None)
    intent = run(router.route("What is the difference between TCP and UDP?"))
    assert intent.needs_graph is True
    assert intent.query_type in ("DESCRIPTION", "ENTITY")


def test_route_simple_description():
    router = QueryRouter(get_settings(), client=None)
    intent = run(router.route("Tell me about the monsoon"))
    assert intent.needs_graph is False