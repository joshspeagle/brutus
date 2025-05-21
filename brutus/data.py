"""Data module."""

__all__ = ["strato"]

import pooch

_dois = {
    "MIST_1.2_iso_vvcrit0.0.h5": "10.7910/DVN/FZMFQY/BKAG41",
    "MIST_1.2_iso_vvcrit0.4.h5": "10.7910/DVN/FZMFQY/PRGJIP",
    "MIST_1.2_EEPtrk.h5": "10.7910/DVN/JV866N/FJ5NNO",
    "bayestar2019_v1.h5": "10.7910/DVN/G49MEI/Y9UZPG",
    "grid_mist_v9.h5": "10.7910/DVN/7BA4ZG/Z7MGA7",
    "grid_mist_v8.h5": "10.7910/DVN/7BA4ZG/NKVZFT",
    "grid_bayestar_v5.h5": "10.7910/DVN/7BA4ZG/LLZP0B",
    # "offsets_mist_v9.txt": "10.7910/DVN/L7D1FY/XXXXXX",
    "offsets_mist_v8.txt": "10.7910/DVN/L7D1FY/QTNKKN",
    "offsets_bs_v9.txt": "10.7910/DVN/L7D1FY/W4O6NJ",
    "nn_c3k.h5": "10.7910/DVN/MSCY2O/XHU1VJ",
}

strato = pooch.create(
    path=pooch.os_cache("astro-brutus"),
    base_url="https://dataverse.harvard.edu/api/access/datafile/",
    registry={
        "MIST_1.2_iso_vvcrit0.0.h5": "ac46048acb9c9c1c10f02ac1bd958a8c4dd80498923297907fd64c5f3d82cb57",
        "MIST_1.2_iso_vvcrit0.4.h5": "25d97db9760df5e4e3b65c686a04d5247cae5027c55683e892acb7d1a05c30f7",
        "MIST_1.2_EEPtrk.h5": "001558c1b32f4a85ea9acca3ad3f7332a565167da3f6164a565c3f3f05afc11b",
        "bayestar2019_v1.h5": "73064ab18f4d1d57b356f7bd8cbcc77be836f090f660cca6727da85ed973d1e6",
        "grid_mist_v9.h5": "7d128a5caded78ca9d1788a8e6551b4329aeed9ca74e7a265e531352ecb75288",
        "grid_mist_v8.h5": "b07d9c19e7ff5e475b1b061af6d1bb4ebd13e0e894fd0703160206964f1084e0",
        "grid_bayestar_v5.h5": "c5d195430393ebd6c8865a9352c8b0906b2c43ec56d3645bb9d5b80e6739fd0c",
        # "offsets_mist_v9.txt": None,
        "offsets_mist_v8.txt": "35425281b5d828431ca5ef93262cb7c6f406814b649d7e7ca4866b8203408e5f",
        "offsets_bs_v9.txt": "b5449c08eb7b894b6d9aa1449a351851ca800ef4ed461c987434a0c250cba386",
        "nn_c3k.h5": "bc86d4bf55b2173b97435d24337579a2f337e80ed050c73f1e31abcd04163259",
    },
    # The name of an environment variable that can overwrite the path
    env="ASTRO_BRUTUS_DATA_DIR",
    retry_if_failed=3,
)
# Need to customize the URLs since pooch doesn't know how to build the URL from the base path using the doi
strato.urls = {
    k: f"{strato.base_url}:persistentId?persistentId=doi:{v}" for k, v in _dois.items()
}
