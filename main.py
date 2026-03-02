# load all packages
import pandas as pd
from synergy_dataset import iter_datasets
import os
from buscarpy import calculate_h0
import time
import numpy as np
from pathlib import Path
# add the model specification
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import typer
from enum import Enum
from scipy.sparse import csr_matrix
from typing import cast
from loguru import logger
import sys
import matplotlib.pyplot as plt


class SimulationType(Enum):
    BASELINE = 1
    REORDERING = 2
    GAP_FILLING = 3

pd.set_option('display.max_columns', 20)

N_UPDATES = 3

logger.remove()
# Set this to DEBUG to see more verbose output, or INFO to see less verbose
#logger.add(sys.stderr, level="DEBUG")  
logger.add(sys.stderr, level="INFO")  

class ScreeningDataset:
    def __init__(
            self,
            df: pd.DataFrame,
            name: str,
            vectorizer_params: dict | None = None,
            batch_size: int = 100
    ):
        self.df = df.copy()
        self.vectorizer = TfidfVectorizer(**(vectorizer_params or {}))
        self.batch_size = batch_size
        self.sim_step = 0
        self.df_result = pd.DataFrame()
        self.name = name

    def vectorise(self):
        """
        Fit the vectorise on the dataset, and set dataset.x_train.
        """
        self.x_train = cast(csr_matrix, self.vectorizer.fit_transform(self.df["texts"]))

class ScreeningSimulation():
    """
    A screening simulation run for a dataset and review.

    Attributes
    ----------
    N : int
        Total number of documents considered in this simulation.
    
    """
    N: int
    df: pd.DataFrame
    batch_size: int
    current_review_update: int

    def __init__(
            self,
            dataset: ScreeningDataset,
            current_review_update: int,
            simulation_type: SimulationType,
            recall_target: float,
            alpha: float
    ):
        self.dataset = dataset
        self.current_review_update = current_review_update
        self.df = dataset.df.copy()
        self.df = self.df[self.df["review_update"] <= current_review_update]
        self.n_new_docs = (self.df["review_update"]==current_review_update).sum()
        self.N = self.df.shape[0]
        self.batch_size = int(max(100, min(0.1 * self.N, 500)))
        self.simulation_type = simulation_type
        self.recall_target = recall_target
        self.alpha=alpha

        self.relevant_documents = self.df["incl"].sum()

    @property
    def documents_screened(self) -> int:
        return (self.df['sim_seen'] == 1).sum()
    
    @property
    def relevant_documents_found(self) -> int:
        return self.df[self.df['sim_seen'] == 1]['incl'].sum()
    
    @property
    def prevalence(self) -> float:
        return self.relevant_documents_found / self.documents_screened
    
    @property
    def recall(self) -> float:
        return self.relevant_documents_found / self.relevant_documents
    
    @property
    def prop_screened(self) -> float:
        return self.documents_screened / self.N

    def simulate_sample(self, random_seed:int = 42) -> bool:
        print("Simulate screening of an initial random set")

        # create a random sample (N = 10% df.shape[0] but min=250 and max=500)
        N = int(max(100, min(0.1 * self.df.shape[0], 500)))

        # exclude datasets where we can't find enough included docs
        
        # Instead of drawing new samples, draw additional documents
        rs = 42
        idx = self.df.sample(n=N, random_state=rs).index
        while (self.df.loc[idx, 'incl'].nunique() <= 1) & ((self.df.loc[idx, 'incl'] == 1).sum() <= 1):
            N+=1
            idx = self.df.sample(n=N, random_state=rs).index
            if N >= self.df.shape[0]*0.5: 
                print(f"Skipping dataset {self.dataset.name} as no suitable sample found after even with half the dataset sampled.")
                return False
            
        self.df.loc[idx, 'sim_seen'] = 1 # we have seen the random sample
        self.df.loc[idx, 'sim_order'] = list(range(1, len(idx)+1)) # we saw them in this order
        self.df.loc[idx, "random_sample"] = 1

        print(f"Documents screened: {self.documents_screened}")
        print(f"Relevant documents identified: {self.relevant_documents_found}")
        print(f"Relevance rate across the dataset: {self.prevalence}")   

        return True    
    
    def prioritised_screening(
            self):
        while self.documents_screened < self.N:
            seen_index = self.df[self.df["sim_seen"] == 1].index
            unseen_index = self.df[self.df["sim_seen"]==0].index
            x_train_seen = self.dataset.x_train[seen_index]
            x_train_unseen = self.dataset.x_train[unseen_index]

            clf = SVC(C=0.25, probability=True, class_weight='balanced')
            # fmh: using standard values for SVM here
            start_time = time.time()
            clf.fit(x_train_seen, self.df.loc[seen_index, 'incl'])
            elapsed_time = time.time() - start_time

            logger.debug(f"used {seen_index.shape} seen docs for training")
            logger.debug(f"clf.fit took: {elapsed_time:.2f} seconds")

            # predict relevance of the remaining documents (from the unseen df_part dataset)
            start_time = time.time()
            y_pred = clf.predict_proba(x_train_unseen)[:,1]
            elapsed_time = time.time() - start_time
            
            logger.debug(f"clf.predict_proba took: {elapsed_time:.2f} seconds")

            self.df.loc[unseen_index,"prediction"] = y_pred

            batch_size = min(self.batch_size, y_pred.shape[0])

            top_idx = np.argpartition(-y_pred, batch_size - 1)[:batch_size]
            batch_idx = unseen_index[top_idx]

            self.df.loc[batch_idx, 'sim_order'] = list(range(
                self.documents_screened, self.documents_screened+batch_size
            ))
            self.df.loc[batch_idx,'sim_seen'] = 1

            if self.simulation_type==SimulationType.BASELINE or self.current_review_update==0:
                logger.debug("normal order")
                ordered_records = self.df[
                    (self.df["sim_seen"]==1)
                ].sort_values("sim_order")["incl"].to_numpy()
            elif self.simulation_type==SimulationType.REORDERING:
                logger.debug("Reordering!")
                random_idx = self.df["random_sample"] == 1
                self.df.loc[random_idx, "prediction"] = 1.0
                ordered_records = self.df[
                    (self.df["sim_seen"]==1)
                ].sort_values("prediction",ascending=False)["incl"].to_numpy()
            elif self.simulation_type==SimulationType.GAP_FILLING:
                simulation_within_simulation = ScreeningSimulation(
                    dataset=self.dataset,
                    current_review_update=self.current_review_update,
                    simulation_type=SimulationType.BASELINE,
                    recall_target=self.recall_target,
                    alpha=self.alpha
                )
                simulation_within_simulation.clear_dataset()
                simulation_within_simulation.prioritised_screening()
                self.df["level_2_order"] = simulation_within_simulation.df["sim_order"]
                ordered_records = self.df[
                    (self.df["sim_seen"]==1)
                ].sort_values("sim_order")["incl"].to_numpy()
            else:
                raise ValueError(f"Don't know how to order records when simulation type == {self.simulation_type}")

            p_val = calculate_h0(
                ordered_records,
                self.N,
                recall_target=self.recall_target
            )

            self.df.loc[batch_idx[-1], "p_score"] = p_val

            if p_val < self.alpha:
                logger.info(
                    f"reached p_value < {self.alpha}." \
                    f" Recall {self.recall}. " \
                    f"Proportion screened: {self.prop_screened:.0%}" \
                    f"({self.documents_screened} / {self.N})"
                )
                break

    def clear_dataset(self) -> None:
        random_sample_idx = self.df["random_sample"]==1
        self.df.loc[random_sample_idx, "sim_seen"] = 0
        self.df.loc[random_sample_idx, "prediction"] = None

def load_data(dataset_name: str):
    """
    Load the data for a synergy dataset with name: dataset_name.
    Do some cleaning: force titles and abstracts to strings.
    Split the data into parts.
    """
    df = pd.read_csv(
        Path("synergy_dataset") / f"{dataset_name}.csv"
    )
    df["sim_seen"] = 0
    df["random_sample"] = 0
    df['publication_year'] = np.random.rand(len(df))
    df = df.rename(columns={"label_included":"incl"})
    df['title'] = df['title'].apply(str)
    df['abstract'] = df['abstract'].apply(str)
    df['id'] = df.index.astype(str)

    df.sort_values('publication_year', inplace=True) # this is the inital dataset
    df.reset_index(inplace=True) 

    # split data into N_UPDATES equally sized subsets
    df['review_update'] = 0
    for y in range(0, N_UPDATES+1):
        part_startindex = y*round(df.shape[0]/N_UPDATES)
        part_endindex = (y+1)*round(df.shape[0]/N_UPDATES) - 1
        df.loc[part_startindex:part_endindex, 'review_update'] = y
    
    print("split data into the following parts:")
    print(df['review_update'].value_counts())

    df["texts"] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    
    return df


def main(testing: bool):
    if testing:
        dataset_name = "Wassenaar_2017"
        recall_target = 0.8
        alpha = 0.05
        df = load_data(dataset_name)

        result = simulate_screening(
            df,
            dataset_name=dataset_name,
            simulation_level=SimulationType.GAP_FILLING,
            recall_target=recall_target,
            alpha=alpha
        )

        # Save result

def simulate_screening(
        df: pd.DataFrame,
        dataset_name: str,
        simulation_level: SimulationType,
        recall_target: float,
        alpha: float,
        ) -> pd.DataFrame:
    
    dataset = ScreeningDataset(df=df, name=dataset_name)
    dataset.vectorise()

    for current_review_update in range(N_UPDATES):
        logger.info(f"Update: {current_review_update}")

        if current_review_update > 0:
            old_idx = df.index
            transfer_columns = [
                "sim_seen",
                "random_sample",
                "sim_order",
                "prediction"
            ]

            dataset.df.loc[df.index,transfer_columns] = df[transfer_columns]

        simulation = ScreeningSimulation(
            dataset=dataset,
            current_review_update=current_review_update,
            simulation_type=simulation_level,
            recall_target=recall_target,
            alpha=alpha
        )
        if current_review_update==0:
            sample_success = simulation.simulate_sample()
            if not sample_success:
                raise ValueError("Could not identify random sample")
            
            
        logger.info(
            f'Simulating stopping method with on dataset {simulation.dataset.name} with {simulation.N} documents available and batch size = {simulation.batch_size}')

        simulation.prioritised_screening()

        df = simulation.df

    return df
