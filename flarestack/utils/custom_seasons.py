import numpy as np
from flarestack.core.time_pdf import TimePDF


def custom_dataset(dataset, sources, time_pdf):
    """For sources which only require a subset of available seasons,
    flarestack will run more quickly if only the relevant data is loaded. If
    the dataset (of all seasons) is provided alongside the source and time
    PDF, a custom dataset will be returned containing only relevant seasons.

    :param dataset: Dataset containing seasons
    :param sources: Source to be considered
    :param time_pdf: Time PDF
    :return: Custom dataset with only relevant seasons
    """

    relevant_seasons = dict()

    for (name, season) in dataset.seasons.items():
        time = TimePDF.create(time_pdf, season.get_time_pdf())

        overlap = np.sum([time.effective_injection_time(src) for src in sources])

        if overlap > 0:
            relevant_seasons[name] = season

    return relevant_seasons
