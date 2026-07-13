import pytest

from drafter.ngram_drafter import NGramDrafter
from interface.abstract_drafter import AbstractDrafter
from interface.abstract_playback import AbstractPlayback
from interface.abstract_verifier import AbstractVerifier
from playback.speculative_playback import SpeculativePlayback
from verifier.greedy_verifier import GreedyVerifier


@pytest.mark.parametrize("abstract", [AbstractDrafter, AbstractVerifier, AbstractPlayback])
def test_abstracts_cannot_instantiate(abstract):
    with pytest.raises(TypeError):
        abstract()


def test_concrete_classes_implement_their_abc():
    assert isinstance(NGramDrafter(), AbstractDrafter)
    assert isinstance(GreedyVerifier(), AbstractVerifier)
    assert isinstance(SpeculativePlayback(), AbstractPlayback)
