"""Name-matching edge cases.

These are the cases that decide whether a screening engine is usable. Each one
is a documented failure mode of naive exact or single-signal matching.
"""

from __future__ import annotations

import pytest

from src.config import ScreeningConfig
from src.sanctions.engine import (
    MatchDecision,
    ScreeningEngine,
    ScreeningSubject,
    WatchlistEntry,
)
from src.sanctions.normalize import normalize_name
from src.sanctions.phonetics import phonetic_keys, phonetic_similarity
from src.sanctions.similarity import compare_names


def _norm(text: str, config: ScreeningConfig):  # type: ignore[no-untyped-def]
    return normalize_name(text, config)


# --------------------------------------------------------------- normalization
def test_honorifics_are_stripped(screening_config: ScreeningConfig) -> None:
    plain = _norm("Mohammed Al-Hassan", screening_config)
    titled = _norm("Sheikh Dr. Mohammed Al-Hassan", screening_config)
    assert titled.tokens == plain.tokens
    assert "sheikh" in titled.removed_honorifics


def test_particles_are_removed_but_recorded(screening_config: ScreeningConfig) -> None:
    result = _norm("Ahmed bin Abdul Al-Rashid", screening_config)
    assert "bin" not in result.tokens
    assert "bin" in result.removed_particles


def test_hyphen_becomes_a_token_boundary(screening_config: ScreeningConfig) -> None:
    # Deleting the hyphen instead would yield the unmatched token "alhassan".
    assert "hassan" in _norm("Al-Hassan", screening_config).tokens


def test_corporate_suffix_removed_only_for_organizations(
    screening_config: ScreeningConfig,
) -> None:
    org = _norm("Crescent Shipping Ltd", screening_config)
    assert org.is_organization is True
    assert "ltd" not in org.tokens

    person = _norm("James Co Anderson", screening_config)
    assert "co" in person.tokens


def test_a_name_of_only_particles_is_not_discarded(screening_config: ScreeningConfig) -> None:
    result = _norm("Al Abu", screening_config)
    assert result.is_empty() is False


# --------------------------------------------------------- phonetic equivalence
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Mohammed", "Muhammad"),
        ("Mohamed", "Muhammed"),
        ("Hussein", "Husayn"),
        ("Qasim", "Kassim"),
        ("Yusuf", "Yousef"),
        ("Aleksandr", "Alexander"),
        ("Ivanov", "Ivanoff"),
        ("Smirnov", "Smirnoff"),
    ],
)
def test_transliteration_variants_share_a_phonetic_key(left: str, right: str) -> None:
    assert set(phonetic_keys(left).keys()) & set(phonetic_keys(right).keys())


def test_ivanov_and_ivanova_are_phonetically_equivalent_but_distinguishable(
    screening_config: ScreeningConfig,
) -> None:
    """The gendered surname problem.

    Ivanov and Ivanova key identically under Double Metaphone, which is correct
    — they are the same family name. They are different people, so the engine
    must not treat the phonetic agreement as sufficient on its own, and the
    corroboration layer is what separates them.
    """
    assert phonetic_keys("Ivanov").primary == phonetic_keys("Ivanova").primary

    signals = compare_names(
        _norm("Sergei Ivanov", screening_config), _norm("Sergei Ivanova", screening_config)
    )
    assert signals.phonetic == pytest.approx(1.0)
    assert signals.jaro_winkler < 1.0


# ------------------------------------------------------------- name-order swaps
def test_name_order_swap_scores_as_a_match(screening_config: ScreeningConfig) -> None:
    signals = compare_names(
        _norm("Sergei Ivanov", screening_config), _norm("Ivanov Sergei", screening_config)
    )
    assert signals.jaro_winkler == pytest.approx(1.0)
    assert signals.order_swapped is True


def test_identical_names_are_not_flagged_as_order_swapped(
    screening_config: ScreeningConfig,
) -> None:
    signals = compare_names(
        _norm("Sergei Ivanov", screening_config), _norm("Sergei Ivanov", screening_config)
    )
    assert signals.order_swapped is False


def test_missing_middle_name_is_penalised_not_ignored(
    screening_config: ScreeningConfig,
) -> None:
    full = _norm("Mohammed Ali Hassan", screening_config)
    partial = _norm("Ali Hassan", screening_config)
    assert phonetic_similarity(full.tokens, partial.tokens) == pytest.approx(2 / 3)


# ---------------------------------------------------------------- scoring rules
@pytest.fixture()
def engine(screening_config: ScreeningConfig) -> ScreeningEngine:
    engine = ScreeningEngine(config=screening_config)
    engine.index_watchlist(
        [
            WatchlistEntry(
                uid="SDN-000001",
                name="Muhammad Al-Hassan",
                program="SDGT",
                entity_type="INDIVIDUAL",
                dob="1975-04-12",
                nationality="SY",
                aliases=("Mohammed Hassan", "محمد الحسن"),
            ),
            WatchlistEntry(
                uid="SDN-000002",
                name="Sergei Ivanov",
                program="UKRAINE-EO13662",
                entity_type="INDIVIDUAL",
                dob="1968-11-02",
                nationality="RU",
            ),
            WatchlistEntry(
                uid="SDN-000003",
                name="Zhang Wei",
                program="CYBER2",
                entity_type="INDIVIDUAL",
                dob="1982",
                nationality="CN",
            ),
        ]
    )
    return engine


def test_romanization_variant_is_detected(engine: ScreeningEngine) -> None:
    matches = engine.screen(
        ScreeningSubject("C1", "Mohamed Al Hassan", dob="1975-04-12", nationality="SY")
    )
    assert matches
    assert matches[0].watchlist_uid == "SDN-000001"
    assert matches[0].decision is MatchDecision.ESCALATE


def test_native_script_subject_matches_latin_watchlist_entry(engine: ScreeningEngine) -> None:
    matches = engine.screen(ScreeningSubject("C2", "محمد الحسن", nationality="SY"))
    assert [m.watchlist_uid for m in matches] == ["SDN-000001"]


def test_han_order_swap_is_detected(engine: ScreeningEngine) -> None:
    matches = engine.screen(ScreeningSubject("C3", "Wei Zhang", nationality="CN"))
    assert [m.watchlist_uid for m in matches] == ["SDN-000003"]


def test_unrelated_name_produces_no_alert(engine: ScreeningEngine) -> None:
    assert engine.screen(ScreeningSubject("C4", "James Robert Anderson", dob="1990-01-01")) == []


def test_dob_mismatch_penalises_an_otherwise_perfect_name(engine: ScreeningEngine) -> None:
    same_dob = engine.screen(
        ScreeningSubject("C5", "Sergei Ivanov", dob="1968-11-02", nationality="RU")
    )[0]
    wrong_dob = engine.screen(
        ScreeningSubject("C6", "Sergei Ivanov", dob="1991-03-30", nationality="RU")
    )[0]
    assert same_dob.final_score > wrong_dob.final_score
    assert dict(wrong_dob.corroboration)["dob_mismatch"] < 0


def test_year_only_dob_gives_partial_credit(engine: ScreeningEngine) -> None:
    match = engine.screen(ScreeningSubject("C7", "Zhang Wei", dob="1982-07-19"))[0]
    assert "dob_year_only" in dict(match.corroboration)


def test_one_alert_per_entity_not_per_alias(engine: ScreeningEngine) -> None:
    # The entry carries three reference names; a hit must not triple-count.
    matches = engine.screen(ScreeningSubject("C8", "Mohammed Hassan", nationality="SY"))
    assert len({m.watchlist_uid for m in matches}) == len(matches)


def test_scoring_breakdown_is_serialisable_and_complete(engine: ScreeningEngine) -> None:
    match = engine.screen(
        ScreeningSubject("C9", "Mohamed Al Hassan", dob="1975-04-12", nationality="SY")
    )[0]
    breakdown = match.scoring_breakdown()
    for key in ("name_score", "corroboration_delta", "final_score", "rule_version",
                "signal_jaro_winkler", "signal_phonetic"):
        assert key in breakdown
    assert breakdown["rule_version"] == engine.config.rule_version
