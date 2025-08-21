from __future__ import annotations

from typing import TYPE_CHECKING

from spacy.lang.en import English

from .registry import BaseRegistry

if TYPE_CHECKING:
    from typing import Type


class SegmenterRegistry(BaseRegistry[Type["BaseSegmenter"]]):
    """A registry for segmenters."""


class BaseSegmenter:
    def __init__(self, segmenter_name_or_path: str, *args, **kwargs):
        super().__init__()

    def segment(self, text: str) -> list[str]:
        raise NotImplementedError()


@SegmenterRegistry.add("spacy")
class SpacySegmenter(BaseSegmenter):
    def __init__(self, segmenter_name_or_path: str, *args: object, **kwargs: object) -> None:
        assert segmenter_name_or_path == "spacy", "Only 'spacy' segmenter is supported"
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

    def segment(self, text: str) -> list[str]:
        return [sent.text_with_ws for sent in self.nlp(text).sents]
