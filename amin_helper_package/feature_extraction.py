import numpy as np
import pandas as pd
from tqdm import tqdm
from packaging import version
from sklearn.preprocessing import LabelEncoder


def device_os_extractor(s):
    """
    Extracts the device OS from the device_os field of the device_info field of the
    :param s: Device OS String
    :return: tuple of device OS and version in float
    """
    plist = s.split('/')[0].split()

    if plist[0].lower() == 'android':
        # Android
        device = 0
    else:
        # iOS/ iPhone
        device = 1

    try:
        v = version.parse(plist[-1])
        version_float = float(f"{v.major}.{v.minor}")
    except:
        version_float = np.nan

    return [device, version_float]


def fill_and_normalize_version(versions):
    mean = np.nanmean(versions)
    versions[np.where(np.isnan(versions))] = mean
    return versions / np.max(versions)


def device_os2_features(df):
    col = df['device_os_s']
    out = col.apply(device_os_extractor)
    out = np.array(list(out.values))

    android_ind = np.where(out[:, 0] == 0)
    ios_ind = np.where(out[:, 0] == 1)

    out[android_ind, 1] = fill_and_normalize_version(out[android_ind, 1])
    out[ios_ind, 1] = fill_and_normalize_version(out[ios_ind, 1])

    return out


def rearrange_categorial_features(df):
    categorical_features = [
        'platform_s',
        'device_mapped_s',
        'device_manufacturer_s',
        'device_gpu_s',
        'device_os_s',
        'device_model_s',
        'geo_s',
        'region_s',
        'lang_s'
    ]

    pbar = tqdm(total=5, desc='Rearranging categorical features')

    convert2numeric_features = [
        ('geo', 'geo_s'),
        ('region', 'region_s'),
        ('lang', 'lang_s')
    ]

    pbar.desc = 'Encoding categorical features'
    for new_feature, cfeature in convert2numeric_features:
        le = LabelEncoder()
        le.fit(df[cfeature])
        df[new_feature] = le.transform(df[cfeature])
        pbar.update(1)

    pbar.desc = 'Extracting mobile device features'
    out = device_os2_features(df)
    df['device_os'] = out[:, 0]
    df['device_os_version'] = out[:, 1]
    pbar.update(1)

    pbar.desc = 'Removing categorical features'
    df = df.drop(columns=categorical_features)
    pbar.update(1)
    pbar.close()

    return df
