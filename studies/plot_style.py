"""Matplotlib style with Japanese font (IPAGothic) for the study figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
          "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": ["IPAPGothic", "IPAGothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
})
DLV, DLP = 9.0 / 22.0, 20.0 / 19.0


def species_lines(ax, label=True, y=None):
    ax.axvline(DLV, color="tab:green", ls="--", lw=1)
    ax.axvline(DLP, color="tab:purple", ls="--", lw=1)
    if label:
        yy = ax.get_ylim()[1] if y is None else y
        ax.text(DLV, yy, " ボルボックス", color="tab:green", va="top", fontsize=8)
        ax.text(DLP, yy, " プレオドリナ", color="tab:purple", va="top", fontsize=8)
