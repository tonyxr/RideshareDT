"""Build the publication-ready CRISP-Ride structural map.

The figure is sized for a two-column journal page and saved as a vector PDF.
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
QA_DIR = ROOT / "tmp" / "pdfs" / "crisp_ride_structure"
PDF_PATH = OUTPUT_DIR / "crisp_ride_structural_map.pdf"
PNG_PATH = QA_DIR / "crisp_ride_structural_map.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


NAVY = "#183B56"
TEXT = "#17212B"
MUTED = "#4D5B68"
LINE = "#36566F"
PALE_LINE = "#A9BAC7"
WHITE = "#FFFFFF"

BLUE = "#1F6FB2"
BLUE_LIGHT = "#EDF5FB"
BLUE_MID = "#D7EAF7"

TEAL = "#0A7C72"
TEAL_LIGHT = "#ECF7F5"
TEAL_MID = "#D4EEE9"

PURPLE = "#6B4BA1"
PURPLE_LIGHT = "#F5F1FA"
PURPLE_MID = "#E7DDF3"

ORANGE = "#C45B20"
ORANGE_LIGHT = "#FFF4EC"


def rounded_box(ax, x, y, w, h, face, edge, radius=0.9, lw=0.8, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.22,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def node(
    ax,
    x,
    y,
    w,
    h,
    title,
    body="",
    *,
    face=WHITE,
    edge=LINE,
    title_color=TEXT,
    body_color=MUTED,
    title_size=8.1,
    body_size=7.7,
    title_y=0.64,
    body_y=0.29,
    lw=0.75,
):
    rounded_box(ax, x, y, w, h, face, edge, radius=0.7, lw=lw)
    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        color=title_color,
        fontsize=title_size,
        fontweight="bold",
        linespacing=1.08,
        zorder=5,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * body_y,
            body,
            ha="center",
            va="center",
            color=body_color,
            fontsize=body_size,
            linespacing=1.08,
            zorder=5,
        )


def arrow(
    ax,
    start,
    end,
    *,
    color=LINE,
    lw=1.15,
    style="-",
    connection="arc3",
    mutation=8.5,
    z=2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, color=MUTED, size=7.4, rotation=0, face=WHITE):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=size,
        rotation=rotation,
        bbox={"boxstyle": "round,pad=0.13", "fc": face, "ec": "none", "alpha": 0.96},
        zorder=7,
    )


def stage_panel(ax, x, y, w, h, number, title, subtitle, color, light):
    rounded_box(ax, x, y, w, h, light, color, radius=1.25, lw=1.15, z=0)
    header_h = 7.0
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.22,rounding_size=1.2",
        linewidth=0,
        facecolor=color,
        zorder=1,
    )
    ax.add_patch(header)
    # Cover the lower rounded corners of the header so it reads as a clean band.
    ax.add_patch(
        plt.Rectangle(
            (x, y + h - header_h),
            w,
            1.3,
            linewidth=0,
            facecolor=color,
            zorder=1.1,
        )
    )
    ax.text(
        x + 1.4,
        y + h - 2.1,
        f"CONTRIBUTION {number}",
        ha="left",
        va="center",
        color=WHITE,
        fontsize=7.3,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        x + 1.4,
        y + h - 4.25,
        title,
        ha="left",
        va="center",
        color=WHITE,
        fontsize=9.3,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        x + 1.4,
        y + h - 6.0,
        subtitle,
        ha="left",
        va="center",
        color=WHITE,
        fontsize=7.1,
        zorder=6,
    )


def build_figure():
    fig, ax = plt.subplots(figsize=(7.15, 6.0))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.988, bottom=0.025)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 82)
    ax.set_aspect("auto")
    ax.axis("off")

    # Three contribution panels.
    stage_panel(
        ax,
        1.0,
        7.3,
        27.2,
        72.8,
        "1",
        "CrowdLLM population",
        "offline synthetic data construction",
        BLUE,
        BLUE_LIGHT,
    )
    stage_panel(
        ax,
        30.0,
        7.3,
        34.7,
        72.8,
        "2",
        "Interactive simulator",
        "riders, drivers, city, and platforms",
        TEAL,
        TEAL_LIGHT,
    )
    stage_panel(
        ax,
        66.5,
        7.3,
        32.5,
        72.8,
        "3",
        "Learnable POMDP",
        "observation, action, reward, and optimization",
        PURPLE,
        PURPLE_LIGHT,
    )

    # Contribution 1: offline construction and calibration.
    node(
        ax,
        3.0,
        62.1,
        23.2,
        8.3,
        "City datasets and\ndocumented priors",
        "trips | fares | weather\npopulation distributions",
        face=WHITE,
        edge=BLUE,
        body_size=7.45,
    )
    node(
        ax,
        3.0,
        50.0,
        10.8,
        8.1,
        "Calibration\nregistry " + r"$\boldsymbol{\psi}_c$",
        "city-specific\nparameters",
        face=BLUE_MID,
        edge=BLUE,
        title_size=8.0,
        body_size=7.65,
    )
    node(
        ax,
        15.4,
        50.0,
        10.8,
        8.1,
        "Profile and ride-\ncontext sampler",
        "persistent profiles\ncold-start trips",
        face=BLUE_MID,
        edge=BLUE,
        title_size=7.65,
        body_size=7.45,
    )
    node(
        ax,
        5.3,
        36.3,
        18.6,
        9.0,
        "Offline CrowdLLM\ninference",
        "bounded price-gap\nsalience parameters",
        face=WHITE,
        edge=BLUE,
        title_size=8.2,
        body_size=7.65,
    )
    node(
        ax,
        5.3,
        21.8,
        18.6,
        9.0,
        "Persistent rider\ncrowd " + r"$\mathcal{U}_c$",
        "cached before\npolicy learning",
        face=BLUE_MID,
        edge=BLUE,
        title_size=8.15,
        body_size=7.7,
    )
    rounded_box(ax, 5.3, 11.2, 18.6, 5.5, WHITE, PALE_LINE, radius=0.65, lw=0.65)
    ax.text(
        14.6,
        13.95,
        "Reproducible heterogeneity\nwithout an LLM in the control loop",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.35,
        linespacing=1.08,
        zorder=5,
    )

    arrow(ax, (14.6, 62.1), (8.4, 58.1), color=BLUE, connection="arc3,rad=0.13")
    arrow(ax, (14.6, 62.1), (20.8, 58.1), color=BLUE, connection="arc3,rad=-0.13")
    arrow(ax, (20.8, 50.0), (18.8, 45.3), color=BLUE, connection="arc3,rad=0.08")
    arrow(ax, (14.6, 36.3), (14.6, 30.8), color=BLUE)
    label(ax, 14.6, 33.55, "behavior parameters", color=BLUE, size=6.95, face=BLUE_LIGHT)

    # Contribution 2: multi-way market interaction.
    node(
        ax,
        32.0,
        62.1,
        14.3,
        8.3,
        "Urban context " + r"$\mathbf{z}_k$",
        "city time | weather\ndemand",
        face=TEAL_MID,
        edge=TEAL,
        title_size=8.2,
        body_size=7.3,
    )
    node(
        ax,
        48.3,
        62.1,
        14.3,
        8.3,
        "Competing\nplatforms",
        r"tariffs $\boldsymbol{\beta}_{f,k}$" + "\npublic quotes",
        face=TEAL_MID,
        edge=TEAL,
        title_size=8.15,
        body_size=7.6,
    )
    node(
        ax,
        32.0,
        47.2,
        14.3,
        9.2,
        "Riders",
        "outside option\nplatform choice",
        face=WHITE,
        edge=TEAL,
        title_size=8.5,
        body_size=7.45,
    )
    node(
        ax,
        48.3,
        47.2,
        14.3,
        9.2,
        "Drivers",
        "active supply\nplatform and\nride acceptance",
        face=WHITE,
        edge=TEAL,
        title_size=8.5,
        body_size=7.25,
    )
    node(
        ax,
        35.1,
        33.4,
        24.5,
        8.8,
        "Service matching and fulfillment",
        "pickup | cancellation\ncompleted service",
        face=TEAL_MID,
        edge=TEAL,
        title_size=8.25,
        body_size=7.45,
    )
    node(
        ax,
        35.1,
        19.5,
        24.5,
        8.8,
        "Realized market outcomes",
        "completed rides | share | wait\ncontribution profit",
        face=WHITE,
        edge=TEAL,
        title_size=8.25,
        body_size=7.6,
    )
    rounded_box(ax, 35.1, 10.9, 24.5, 4.8, TEAL_MID, TEAL, radius=0.65, lw=0.75)
    ax.text(
        47.35,
        13.3,
        "Endogenous next market state " + r"$x_{k+1}$",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=7.65,
        fontweight="bold",
        zorder=5,
    )

    arrow(ax, (39.15, 62.1), (39.15, 56.4), color=TEAL)
    arrow(ax, (55.45, 62.1), (43.1, 56.4), color=TEAL, connection="arc3,rad=0.10")
    arrow(ax, (55.45, 62.1), (55.45, 56.4), color=TEAL)
    label(ax, 48.4, 58.7, "quotes", color=TEAL, size=7.0, face=TEAL_LIGHT)
    arrow(ax, (39.15, 47.2), (42.3, 42.2), color=TEAL, connection="arc3,rad=-0.05")
    arrow(ax, (55.45, 47.2), (52.4, 42.2), color=TEAL, connection="arc3,rad=0.05")
    label(ax, 39.5, 44.1, "requests", color=TEAL, size=6.9, face=TEAL_LIGHT)
    label(ax, 55.2, 44.1, "supply", color=TEAL, size=6.9, face=TEAL_LIGHT)
    arrow(ax, (47.35, 33.4), (47.35, 28.3), color=TEAL)
    arrow(ax, (47.35, 19.5), (47.35, 15.7), color=TEAL)
    arrow(
        ax,
        (59.6, 13.3),
        (61.6, 62.1),
        color=TEAL,
        style=(0, (3, 2)),
        connection="arc3,rad=0.33",
        lw=1.0,
        z=1.5,
    )
    label(ax, 62.05, 38.4, "market feedback", color=TEAL, size=6.8, rotation=90, face=TEAL_LIGHT)

    # Contribution 3: simulator-derived POMDP and adapted optimization.
    node(
        ax,
        69.0,
        62.0,
        27.5,
        8.5,
        "Complete simulator state " + r"$x_k$",
        "rider | driver | market\nprivate competitor variables",
        face=PURPLE_MID,
        edge=PURPLE,
        title_size=8.2,
        body_size=7.55,
    )
    node(
        ax,
        69.0,
        50.5,
        27.5,
        7.4,
        "Information selection",
        "observability | causal coverage\nparsimony",
        face=WHITE,
        edge=PURPLE,
        title_size=8.15,
        body_size=7.3,
    )
    node(
        ax,
        69.0,
        39.0,
        27.5,
        7.4,
        "Actor observation window " + r"$O_n$",
        "public quotes | delayed operations\naction timing",
        face=PURPLE_MID,
        edge=PURPLE,
        title_size=8.15,
        body_size=7.3,
    )
    node(
        ax,
        69.0,
        27.0,
        13.0,
        8.0,
        r"Action $a_n$",
        "hold or bounded\ncity-relative\ntariff target",
        face=WHITE,
        edge=PURPLE,
        title_size=8.15,
        body_size=7.15,
        title_y=0.69,
        body_y=0.30,
    )
    node(
        ax,
        83.5,
        27.0,
        13.0,
        8.0,
        r"Reward $r^P$ or $r^C$",
        "normalized profit or\ncompetitiveness\noutcomes",
        face=WHITE,
        edge=PURPLE,
        title_size=7.85,
        body_size=7.05,
        title_y=0.69,
        body_y=0.30,
    )
    node(
        ax,
        69.0,
        14.8,
        27.5,
        8.2,
        "Adapted recurrent categorical PPO",
        "interval credit | duration discount\ncorrect truncation | staged opponents\nobjective-consistent checkpoint selection",
        face=PURPLE_MID,
        edge=PURPLE,
        title_size=8.05,
        body_size=7.05,
        title_y=0.69,
        body_y=0.29,
    )
    rounded_box(ax, 69.0, 9.3, 27.5, 3.4, ORANGE_LIGHT, ORANGE, radius=0.6, lw=0.85)
    ax.text(
        82.75,
        11.0,
        "Frozen profit or competitiveness\npricing policy",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=7.55,
        fontweight="bold",
        zorder=5,
    )

    arrow(ax, (82.75, 62.0), (82.75, 57.9), color=PURPLE)
    arrow(ax, (82.75, 50.5), (82.75, 46.4), color=PURPLE)
    arrow(ax, (82.75, 39.0), (78.0, 35.0), color=PURPLE, connection="arc3,rad=0.12")
    arrow(ax, (89.9, 27.0), (87.0, 23.0), color=PURPLE, connection="arc3,rad=-0.10")
    arrow(ax, (75.5, 27.0), (78.4, 23.0), color=PURPLE, connection="arc3,rad=0.10")
    arrow(ax, (82.75, 14.8), (82.75, 12.7), color=ORANGE, lw=1.25)

    # Cross-module data paths.
    arrow(
        ax,
        (13.8, 54.1),
        (32.0, 66.2),
        color=BLUE,
        connection="arc3,rad=-0.17",
        lw=1.15,
        z=2.2,
    )
    label(ax, 26.8, 59.1, "city calibration", color=BLUE, size=7.0, face=WHITE)
    arrow(
        ax,
        (23.9, 26.3),
        (32.0, 51.8),
        color=BLUE,
        connection="arc3,rad=-0.28",
        lw=1.15,
        z=2.2,
    )
    label(ax, 28.8, 35.1, "persistent crowd", color=BLUE, size=7.0, rotation=63, face=WHITE)

    arrow(ax, (64.7, 66.2), (69.0, 66.2), color=LINE, lw=1.3, z=2.4)

    # Route realized economic outcomes to the reward definition without
    # crossing the action or observation boxes.
    ax.plot(
        [59.6, 65.7, 65.7, 89.9],
        [23.9, 23.9, 37.0, 37.0],
        color=PURPLE,
        linewidth=1.05,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2.2,
    )
    arrow(ax, (89.9, 37.0), (89.9, 35.0), color=PURPLE, lw=1.05, z=2.2)
    label(ax, 74.5, 37.0, "economic outcomes", color=PURPLE, size=7.0, face=WHITE)

    # Closed-loop pricing action from the learning policy back to the platforms.
    ax.plot(
        [69.0, 65.75, 65.75, 63.4],
        [18.9, 18.9, 66.2, 66.2],
        color=ORANGE,
        linewidth=1.4,
        linestyle=(0, (4, 2.5)),
        dash_capstyle="round",
        dash_joinstyle="round",
        zorder=2.6,
    )
    arrow(
        ax,
        (63.4, 66.2),
        (62.6, 66.2),
        color=ORANGE,
        style=(0, (4, 2.5)),
        lw=1.4,
        mutation=9.5,
        z=2.6,
    )
    label(
        ax,
        65.75,
        43.0,
        r"bounded tariff action $a_n$",
        color=ORANGE,
        size=7.1,
        rotation=90,
        face=WHITE,
    )

    # Arrow-style key, kept compact so the figure is self-contained.
    arrow(ax, (2.0, 3.4), (7.3, 3.4), color=LINE, lw=1.15, mutation=8.0)
    ax.text(8.0, 3.4, "data / market transition", va="center", ha="left", fontsize=7.2, color=MUTED)
    arrow(
        ax,
        (34.5, 3.4),
        (39.8, 3.4),
        color=ORANGE,
        style=(0, (4, 2.5)),
        lw=1.35,
        mutation=8.0,
    )
    ax.text(40.5, 3.4, "pricing-action feedback", va="center", ha="left", fontsize=7.2, color=MUTED)
    ax.text(
        99.0,
        3.4,
        "offline initialization + closed-loop learning",
        va="center",
        ha="right",
        fontsize=7.35,
        color=NAVY,
        fontweight="bold",
    )

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(PDF_PATH, format="pdf", dpi=300, metadata={"Title": "CRISP-Ride structural map"})
    fig.savefig(PNG_PATH, format="png", dpi=300)
    plt.close(fig)
    print(PDF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
