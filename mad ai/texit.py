import matplotlib
import matplotlib.pyplot as plt
import random

# Runtime Configuration Parameters
matplotlib.rcParams["mathtext.fontset"] = "cm"  # Font changed to Computer Modern


async def latex2image(
    latex_expression, image_name = "image.png", image_size_in=(50, 20), fontsize=20, dpi=200
):
    fig = plt.figure(figsize=image_size_in, dpi=dpi)
    text = fig.text(
        x=0.5,
        y=0.5,
        s=latex_expression,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )

    plt.savefig(image_name,bbox_inches="tight")

    return fig

