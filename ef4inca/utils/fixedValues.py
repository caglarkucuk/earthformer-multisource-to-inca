#### Normalization values of the dataset! Computed from the training set, no 2023 data is used for this. ###
NORM_DICT_TOTAL = {'ch05': {'mean': 234.55155594978726, 'std': 5.562155596657649}, 
    'ch06': {'mean': 246.52750706047374, 'std': 10.989032542074183}, 
    'ch07': {'mean': 264.19249641099907, 'std': 21.60488152494804}, 
    'ch09': {'mean': 264.9096449978718, 'std': 22.645564005913293}, 
    'radar': {'mean': 0.04819213929765177, 'std': 0.35574117560924823}, 
    'lght': {'mean': 0.0007089591366500517, 'std': 0.041719103179192224}, 
    'inca_precip': {'mean': 0.06588366464970631, 'std': 0.37410774996545293}, 
    'inca_cape': {'mean': 41.38775568677592, 'std': 55.79871680986862}, 
    'dem': {'mean': 773.2015428571428, 'std': 627.1002074757997}, 
    'lat': {'mean': 47.65664887073358, 'std': 1.038702266116927}, 
    'lon': {'mean': 12.933865015633783, 'std': 2.6925009993609623},
    }

#### Hand-picked events for best and worst performance from the test dataset
bestSamples = ['sampled_202305051455', 'sampled_202307130340', 'sampled_202307050335',
               'sampled_202308261550', 'sampled_202307041510']

worstSamples = ['sampled_202307111910', 'sampled_202307122015', 'sampled_202306271725',
               'sampled_202306231615', 'sampled_202308051200']

#### Randomly selected events from the remaining of the test dataset
randSamples = ['sampled_202308061645', 'sampled_202307211455', 'sampled_202307292200',
               'sampled_202307130125', 'sampled_202307151635', 'sampled_202307250800',
               'sampled_202308291420', 'sampled_202308021355', 'sampled_202308171200', 'sampled_202307040235']