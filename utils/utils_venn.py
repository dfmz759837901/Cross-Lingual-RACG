from typing import *
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
from matplotlib.colors import hex2color, to_hex
import numpy as np
from matplotlib.font_manager import FontProperties


def blend_colors(colors: List[str], alpha: float = 1.0) -> str:

    rgbs = [hex2color(c) for c in colors]

    blended_rgb = np.mean(rgbs, axis=0)

    return to_hex((*blended_rgb, alpha), keep_alpha=True)


def draw_venn(
    subsets: List[set],
    set_labels: List[str],
    save_path: str,
    title: str = "Venn Diagram",
    title_size=28,
    label_size=20,
    colors: Optional[List[str]] = None,
    alpha: float = 0.8
):
    assert len(subsets) == len(set_labels), "Subsets and labels count mismatch"
    assert (
        len(colors) == len(subsets) if colors is not None else True
    ), "Colors count mismatch"
    assert len(subsets) in (2, 3), "Only 2 or 3 sets supported"

    font_path = "utils/fonts/Times-Roman.ttf"
    title_font = FontProperties(fname=font_path, size=title_size)
    label_font = FontProperties(fname=font_path, size=label_size)

    _, ax = plt.subplots()
    if len(subsets) == 3:
        venn_diagram = venn3(
            subsets=subsets, set_labels=set_labels, ax=ax
        )
    else:
        venn_diagram = venn2(
            subsets=subsets, set_labels=set_labels, ax=ax
        )

    if colors is not None:

        region_config = {
            2: {
                '10': [0], '01': [1], '11': [0, 1]
            },
            3: {
                '100': [0], '010': [1], '001': [2],
                '110': [0, 1], '101': [0, 2], '011': [1, 2],
                '111': [0, 1, 2]
            }
        }

        regions = region_config[len(subsets)]
        for region_id, color_indices in regions.items():
            try:
                patch = venn_diagram.get_patch_by_id(region_id)
                if patch:

                    blend_colors_list = [colors[i] for i in color_indices]

                    blended_color = blend_colors(blend_colors_list, alpha)
                    patch.set_color(blended_color)
                    patch.set_edgecolor('none')
            except AttributeError:
                pass
    
    for text in venn_diagram.set_labels + venn_diagram.subset_labels:
        if text is not None:
            text.set_fontproperties(label_font)

    plt.title(title, fontproperties=title_font)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7}
    set_c = {5, 7, 8, 9, 1}

    draw_venn(
        subsets=[set_a, set_b],
        set_labels=["Set A", "Set B"],
        save_path="figures/temp_venn2.png",
        title="Two-Set Example",
    )

    draw_venn(
        subsets=[set_a, set_b, set_c],
        set_labels=["Set A", "Set B", "Set C"],
        save_path="figures/temp_venn3.png",
        title="",
        colors=['#0072B2', '#D55E00', '#CC79A7']
    )
