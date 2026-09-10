import pandas as pd
import warnings


def supress_warnings(
    warnings_to_surpress=[pd.errors.PerformanceWarning, FutureWarning, UserWarning]
):
    """
    Default warning suppressor that does nothing.
    """

    for warning in warnings_to_surpress:
        warnings.simplefilter(action="ignore", category=warning)
