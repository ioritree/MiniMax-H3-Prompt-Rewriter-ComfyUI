"""ComfyUI entry point for the MiniMax-H3 T2VA prompt rewriter."""

import importlib
import logging

from .minimax_h3_rewriter import routes as _routes  # registers HTTP routes
from .minimax_h3_rewriter.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

log = logging.getLogger(__name__)

#: The nodes built on the v3 node API: they use ``io.Autogrow`` for their input
#: groups, which an older ComfyUI does not have. Each is registered in its own
#: attempt so a failure in one -- an import error, a schema the running version
#: rejects -- costs that node alone and not the other.
_GROWING_NODES = (
    ("multi_caption", "Multi Reference Caption"),
    ("ref_writer", "Reference Writer (Ref2VA)"),
)

for _name, _label in _GROWING_NODES:
    try:
        _module = importlib.import_module(f".minimax_h3_rewriter.{_name}", __name__)
        NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        log.warning(
            "[minimax_h3_rewriter] '%s' needs a newer ComfyUI than this one, so it is not "
            "registered. Every other node in the pack is unaffected.",
            _label,
            exc_info=True,
        )

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
