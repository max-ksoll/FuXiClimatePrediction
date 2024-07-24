import logging

import torch

logger = logging.getLogger(__name__)


def plot_average_difference_over_time(observation, model_out, variable_idx):
    # BATCH X AUTOREGRESSION X VARIABLES X LATITUDE X LONGITUDE
    # TODO
    difference = torch.mean(observation[:, 2:, :, :, :] - model_out)

    average_difference = difference.mean(dim="time").transpose("latitude", "longitude")
    average_difference[0] = 0
    average_difference[1] = 0
    average_difference[179] = 0
    average_difference[178] = 0

    # Karte plotten
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines()

    # Landflächen grau darstellen, wenn unter 850 hPa

    # Daten plotten. ,vmin=-getExtrem(variable), vmax=getExtrem(variable)
    im = average_difference.plot(
        ax=ax,
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
        cmap="bwr",
        vmin=-getExtrem(variable),
        vmax=getExtrem(variable),
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5).set_label(
        f"bias [{get_units(variable)}]"
    )
    ax.set_title(f"")
    plt.savefig(
        f"/content/drive/MyDrive/MLEarth/{variable}/FUXI{variable.title()}-{prediction_timedelta/4+0.25}days.png"
    )
    plt.close()
    print(
        f"finished: /content/drive/MyDrive/MLEarth/{variable}/FUXI{variable.title()}-{prediction_timedelta/4+0.25}days.png"
    )
