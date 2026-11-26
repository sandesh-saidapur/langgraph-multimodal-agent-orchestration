"""
Model stand-ins for Phase 2.

These classes have the same methods a real model wrapper would have, so the
graph code does not change when you switch. They run offline.

To use a real model later, write a class with the same methods that calls the
API, and change the two lines in graph.py that create `vision_model` and
`router_model`.
"""

import os
import re


class MockVisionModel:
    """Pretends to look at an image by reading its file name.

    A real vision model would receive the image bytes and return a description
    of what it sees. The file name stands in for that here.
    """

    def describe(self, image_path: str) -> str:
        name = os.path.basename(image_path).lower()
        if "warning" in name:
            return "a dashboard with an orange engine warning light illuminated"
        return "an image with no recognisable automotive content"


class MockRouterModel:
    """Chooses 'calculator' or 'lookup' using simple rules.

    A real router would send the question (and image description) to a language
    model and ask it to pick a tool. Same output: 'calculator' or 'lookup'.
    """

    def choose(self, question: str, image_description: str | None) -> str:
        has_digits = bool(re.search(r"\d", question))
        if has_digits and any(op in question for op in ["+", "*", "/", "-", "calculate", "what is"]):
            return "calculator"
        return "lookup"
