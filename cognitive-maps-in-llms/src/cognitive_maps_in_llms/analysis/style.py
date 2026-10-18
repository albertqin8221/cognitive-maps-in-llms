from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPORT_RC_PARAMS: dict[str, Any] = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class MosaicPlotStyle:
    """Visual defaults and automatic axis policy for mosaic dot plots."""

    figsize: tuple[float, float] = (2.15, 1.25)
    font_family: str = "Helvetica"
    font_size: float = 8
    point_size: float = 4
    alpha: float = 0.4
    jitter: float = 0.1
    jitter_seed: int = 0
    offset_width: float = 0.125
    color_sets: tuple[tuple[str, ...], ...] = (
        ("tab:green",),
        ("tab:red", "#80558F"),
        ("tab:red", "tab:green", "tab:blue"),
    )
    legend: bool = False
    legend_loc: str = "upper right"
    legend_bbox_to_anchor: tuple[float, float] | None = (1.03, 1.03)
    legend_font_size: float = 6
    legend_frame_on: bool = False
    legend_marker_alpha: float = 0.8
    legend_marker_size: float = 20
    ratio_decades: int | None = None
    ratio_min_decades: int = 5
    ratio_decade_step: int = 5
    ratio_padding_decades: float = 0.25

    def __post_init__(self) -> None:
        if not self.color_sets or any(not colors for colors in self.color_sets):
            raise ValueError("color_sets must contain only nonempty palettes.")
        if self.legend_font_size <= 0:
            raise ValueError("legend_font_size must be positive.")
        if self.ratio_decades is not None and self.ratio_decades <= 0:
            raise ValueError("ratio_decades must be positive or None.")
        if self.ratio_min_decades <= 0:
            raise ValueError("ratio_min_decades must be positive.")
        if self.ratio_decade_step <= 0:
            raise ValueError("ratio_decade_step must be positive.")
        if self.ratio_padding_decades < 0:
            raise ValueError("ratio_padding_decades must be non-negative.")

    def rc_params(self) -> dict[str, Any]:
        return {
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size,
            "ytick.labelsize": self.font_size,
            "legend.fontsize": self.font_size,
            **EXPORT_RC_PARAMS,
        }


PAPER_MOSAIC_STYLE = MosaicPlotStyle()
