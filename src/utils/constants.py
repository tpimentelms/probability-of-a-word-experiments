import numpy as np

POWER_RANGE = np.arange(0., 3, 0.25)
STRIDE = 200
MODELS = ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 
          'pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-14b', 'pythia-28b', 'pythia-69b', 'pythia-120b',]

DATASETS = ['brown', 'natural_stories', 'provo', 'dundee', ]

PREDICTOR_NAMES = {
    'surprisal_buggy': r'$h(w_{t})$',
    'prev_surprisal_buggy': r'$h(w_{t\!-\!1})$',
    'prev2_surprisal_buggy': r'$h(w_{t\!-\!2})$',
    'prev3_surprisal_buggy': r'$h(w_{t\!-\!3})$',

    'surprisal': r'$h_{f}(w_{t})$',
    'prev_surprisal': r'$h_{f}(w_{t\!-\!1})$',
    'prev2_surprisal': r'$h_{f}(w_{t\!-\!2})$',
    'prev3_surprisal': r'$h_{f}(w_{t\!-\!3})$',
}

PREDICTOR_ORDER = {
    'surprisal': 3,
    'prev_surprisal': 2,
    'prev2_surprisal': 1,
    'prev3_surprisal': 0,

    'surprisal_buggy': 3,
    'prev_surprisal_buggy': 2,
    'prev2_surprisal_buggy': 1,
    'prev3_surprisal_buggy': 0,
}
PREDICTOR_ORDER_HIGHER = {
    'surprisal': 6,
    'prev_surprisal': 6,
    'prev2_surprisal': 6,
    'prev3_surprisal': 6,

    'surprisal_buggy': 1,
    'prev_surprisal_buggy': 1,
    'prev2_surprisal_buggy': 1,
    'prev3_surprisal_buggy': 1,
}
DATASET_NAMES_FULL = {
    'natural_stories': 'Natural Stories',
    'brown': 'Brown',
    'provo': 'Provo (Progressive Gaze Duration, No Skipped)',
    'provo_skip2zero': 'Provo (Progressive Gaze Duration, Skipped Time=0)',
    'dundee': 'Dundee (Progressive Gaze Duration, No Skipped)',
    'dundee_skip2zero': 'Dundee (Progressive Gaze Duration, Skipped Time=0)',
}
DATASET_NAMES = {
    'natural_stories': 'Natural Stories',
    'brown': 'Brown',
    'provo': 'Provo',
    'provo_skip2zero': 'Provo',
    'dundee': 'Dundee',
    'dundee_skip2zero': 'Dundee',
    'meco': 'Meco',
}
DATASET_NAMES_PLOT = {
    'natural_stories': 'Natural Stories',
    'brown': 'Brown',
    'provo': 'Provo (No skip)',
    'provo_skip2zero': 'Provo (Skip=0)',
    'dundee': 'Dundee (No skip)',
    'dundee_skip2zero': 'Dundee (Skip=0)',
}
DATASET_ORDER = {
    'brown': 0,
    'natural_stories': 1,
    'provo': 4,
    'provo_skip2zero': 2,
    'dundee': 5,
    'dundee_skip2zero': 3,
}
