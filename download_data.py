import requests
from synergy_dataset import iter_datasets
import json
import numpy as np
import pandas as pd
from rich import print

oa_batch_size = 100

def get_works(ids:list[str]):
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"ids.openalex:{"|".join(ids)}",
        "select": "id,publication_year",
        "per-page": oa_batch_size
    }
    r = requests.get(url, params=params)
    res = json.loads(r.content)
    return res["results"]

def main():
    for d in iter_datasets():
        df = d.to_frame()
        df["work_id"] = [x.split("/")[-1].replace("'", "") for x in df.index.values]
        pub_years = []

        n_splits = int(np.ceil(len(df) / oa_batch_size))

        batches = np.array_split(df, n_splits)

        for batch in batches:
            pub_years.extend(get_works(batch["work_id"].values))
            print(len(pub_years))

        pub_year_df = pd.DataFrame(pub_years)

        pub_year_df["work_id"] = [x.split("/")[-1].replace("'", "") for x in pub_year_df["id"].values]

        print(pub_year_df)

        df = df.merge(
            pub_year_df,
            left_on="work_id",
            right_on="work_id",
            how="left"
        ).drop(columns="id")

        print(f"found publication year for {round(100*df['publication_year'].value_counts().sum()/df.shape[0],2)}% records")
        df.to_csv(f"synergy_datasets/{d.name}.csv", index=False)


if __name__== "__main__":
    main()