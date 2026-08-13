# load all packages
from __future__ import annotations
import pandas as pd
from synergy_dataset import iter_datasets
import os
from buscarpy import calculate_h0
import time
import numpy as np
from pathlib import Path
# add the model specification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import typer
from enum import Enum
from scipy.sparse import csr_matrix
from typing import cast
from loguru import logger
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from copy import deepcopy

class SimulationType(Enum):
    BASELINE = 1
    REORDERING = 2
    GAP_FILLING = 3

pd.set_option('display.max_columns', 20)

N_UPDATES = 3
BATCH_NUMBER = "batch_number"
SIM_SEEN_IN_UPDATE = "sim_seen_in_update"

logger.remove()
# Set this to DEBUG to see more verbose output, or INFO to see less verbose
# logger.add(sys.stderr, level="DEBUG")  
logger.add(sys.stderr, level="INFO")  

class ScreeningDataset:
    def __init__(
            self,
            df: pd.DataFrame,
            name: str,
            vectorizer_params: dict | None = None,
    ):
        self.df = df.copy()
        self.vectorizer = TfidfVectorizer(**(vectorizer_params or {}))
        self.name = name

    def vectorise(self):
        """
        Fit the vectorise on the dataset, and set dataset.x_train.
        """
        self.x_train = cast(csr_matrix, self.vectorizer.fit_transform(self.df["texts"]))

    def update(self, previous_step: pd.DataFrame):
        """Update the dataset with data from the previous screening step."""
        logger.info(f"Before update {self.df.shape}")
        transfer_columns = [
            "sim_seen",
            "random_sample",
            "sim_order",
            "prediction",
            "p_score",
            SIM_SEEN_IN_UPDATE
        ]

        self.df.loc[previous_step.index,transfer_columns] = previous_step[transfer_columns]
        logger.info(f"After update {self.df.shape}")


class ScreeningSimulation():
    """
    A screening simulation run for a dataset and review.
    """

    def __init__(
            self,
            dataset: ScreeningDataset,
            current_review_update: int,
            simulation_type: SimulationType,
            recall_target: float,
            alpha: float,
            run_id: int
    ):
        self.dataset = dataset
        self.current_review_update = current_review_update
        self.simulation_type = simulation_type
        self.recall_target = recall_target
        self.alpha = alpha
        self.run_id = run_id

        # Filter the dataset for this review update
        self.df = self._prepare_df(dataset.df, current_review_update)
        self.N = self.df.shape[0]
        self.batch_size = int(max(100, min(0.1 * self.dataset.df.shape[0], 500)))
        self.batch_size = 200
        self.relevant_documents = self.df["incl"].sum()

    def _prepare_df(self, df: pd.DataFrame, current_update: int) -> pd.DataFrame:
        """Return a copy of the dataset filtered by review_update."""
        df_copy = df.copy()
        return df_copy[df_copy["review_update"] <= current_update]

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

    def simulate_sample(self) -> bool:
        logger.info("Simulate screening of an initial random set")

        # create a random sample (N = 10% df.shape[0] but min=250 and max=500)
        N = int(max(100, min(0.1 * self.df.shape[0], 500)))

        rng = np.random.default_rng(seed=self.run_id)  # reproducible
        shuffled_idx = rng.permutation(self.df.index.to_numpy())
        
        # Instead of drawing new samples, draw additional documents
        idx = shuffled_idx[:N]
        logger.info(f"RUN ID {self.run_id}")
        while (self.df.loc[idx, 'incl'].nunique() <= 1) & ((self.df.loc[idx, 'incl'] == 1).sum() <= 1):
            N+=1
            idx = shuffled_idx[:N]
            if N >= self.df.shape[0]*0.5: 
                logger.warning(f"Skipping dataset {self.dataset.name} as no suitable sample found after even with half the dataset sampled.")
                return False
            
        self.df.loc[idx, 'sim_seen'] = 1 # we have seen the random sample
        self.df.loc[idx, 'sim_order'] = list(range(1, len(idx)+1)) # we saw them in this order
        self.df.loc[idx, "random_sample"] = 1
        self.df.loc[idx, SIM_SEEN_IN_UPDATE] = 0

        logger.info(f"Documents screened: {self.documents_screened}")
        logger.info(f"Relevant documents identified: {self.relevant_documents_found}")
        logger.info(f"Relevance rate across the dataset: {self.prevalence}")   

        return True 

    def _fit_model(self, seen_index: pd.Index) -> SVC:
        clf = SVC(
            C=0.25, 
            probability=True, 
            class_weight='balanced',
            random_state=self.run_id
        )
        # fmh: using standard values for SVM here
        start_time = time.time()
        clf.fit(
            self.dataset.x_train[seen_index], 
            self.df.loc[seen_index, 'incl']
        )
        elapsed_time = time.time() - start_time

        logger.debug(f"used {seen_index.shape} seen docs for training")
        logger.debug(f"clf.fit took: {elapsed_time:.2f} seconds")
        return clf
    
    def _make_predictions(self, clf: SVC, unseen_index: pd.Index):
        start_time = time.time()
        y_pred = clf.predict_proba(self.dataset.x_train[unseen_index])[:,1]
        elapsed_time = time.time() - start_time
        
        logger.debug(f"clf.predict_proba took: {elapsed_time:.2f} seconds")

        self.df.loc[unseen_index,"prediction"] = y_pred
        return y_pred
    
    def _get_ordered_records(
            self, 
            inner_simulation: ScreeningSimulation | None  = None
        ) -> pd.DataFrame:
        """
        Get an ordered list of includes and excludes to pass to the stopping 
        criterion. This is dependent on the type of simulation.
        """
        df = self.df.copy()
        if self.simulation_type==SimulationType.BASELINE or self.current_review_update==0:
            logger.info("normal order")
            ordered_records = df[
                (df["sim_seen"]==1)
            ].sort_values("sim_order")
        elif self.simulation_type==SimulationType.REORDERING:
            logger.info("Reordering in descending order of prediction")
            random_idx = df["random_sample"] == 1
            df.loc[random_idx, "prediction"] = 1.0
            ordered_records = df[
                (df["sim_seen"]==1)
            ].sort_values("prediction", ascending=False)
        elif self.simulation_type==SimulationType.GAP_FILLING:
            if inner_simulation is None:
                raise ValueError("No baseline sim supplied")
            logger.info("Reordering in descending order of prediction generated withing inner simulation")
            df["level_2_order"] = inner_simulation.df["sim_order"]
            ordered_records = df[
                (df["sim_seen"]==1)
            ].sort_values("level_2_order")
        else:
            raise ValueError(f"Don't know how to order records when simulation type == {self.simulation_type}")
        
        ordered_records = ordered_records.reset_index(drop=True)
        ordered_records["sim_reordered_order"] = np.arange(ordered_records.shape[0])
        
        return ordered_records.reset_index(drop=True)
    
    def prioritised_screening(self):
        """
        Simulate prioritised screening on the dataset (up to current_review_update).

        - train a model on seen data
        - make predictions on unseen data 
        - "screen" a batch of the data with the highest predictions
        - calculate stopping criterion according to the simulation method
        - stop screening when criterion says it is safe to do so 
        """
        
        logger.info(
            f'Simulating stopping method with simulation type {self.simulation_type} on dataset {self.dataset.name} with {self.N} documents available, batch size = {self.batch_size}, and recall target = {self.recall_target}. {self.documents_screened} screened already.')
        
        if self.simulation_type==SimulationType.GAP_FILLING and self.current_review_update>0:
            simulation_within_simulation = ScreeningSimulation(
                dataset=deepcopy(self.dataset),
                current_review_update=self.current_review_update,
                simulation_type=SimulationType.BASELINE,
                recall_target=self.recall_target,
                alpha=0,
                run_id=self.run_id
            )
            logger.info("Running simulation within simulation.")
            simulation_within_simulation.clear_dataset()
            simulation_within_simulation.prioritised_screening()
        else:
            simulation_within_simulation = None

        while self.documents_screened < self.N:
            seen_index = self.df[self.df["sim_seen"] == 1].index
            unseen_index = self.df[self.df["sim_seen"]==0].index

            clf = self._fit_model(seen_index=seen_index)
            y_pred = self._make_predictions(clf=clf, unseen_index=unseen_index)

            batch_size = min(self.batch_size, y_pred.shape[0])

            top_idx = np.argpartition(-y_pred, batch_size - 1)[:batch_size] # does this mess with the simulated screening where it would be in truly descending order (argsort)
            batch_idx = unseen_index[top_idx]

            self.df.loc[batch_idx, 'sim_order'] = list(range(
                self.documents_screened, self.documents_screened+batch_size
            ))
            self.df.loc[batch_idx,'sim_seen'] = 1
            self.df.loc[batch_idx,SIM_SEEN_IN_UPDATE] = self.current_review_update

            ordered_records = self._get_ordered_records(
                inner_simulation=simulation_within_simulation
            )

            #self.save_interim_results("interim-screening-results", ordered_records)
            logger.info(f"Calculating p-value for {len(ordered_records)}, {self.N}")

            p_val = calculate_h0(
                ordered_records["incl"].to_numpy(),
                self.N,
                recall_target=self.recall_target
            )

            self.df.loc[batch_idx[-1], "p_score"] = p_val
            logger.info(f"Lowest p-value was {p_val} at index {batch_idx[-1]}")

            if p_val < self.alpha:
                logger.info(
                    f"reached p_value < {self.alpha}." \
                    f" Recall {self.recall}. " \
                    f"Proportion screened: {self.prop_screened:.0%}" \
                    f"({self.documents_screened} / {self.N})"
                )
                return
        logger.info("Finished screening all documents in dataset")

    def clear_dataset(self) -> None:
        """Clear out everything apart from the initial random sample."""
        nonrandom_sample_idx = self.df["random_sample"]==0
        self.df.loc[nonrandom_sample_idx, "sim_seen"] = 0
        self.df.loc[nonrandom_sample_idx, "sim_order"] = 0
        self.df["prediction"] = None

    def save_results(self, base_path: str) -> None:
        """Save results to a parquet table."""
        results_df = self.df.copy()
        results_df["dataset_name"] = self.dataset.name
        results_df["simulation_type"] = self.simulation_type.name
        results_df["run_id"] = self.run_id
        results_df["recall_target"] = self.recall_target
        results_df["alpha"] = self.alpha

        table = pa.Table.from_pandas(results_df)

        pq.write_to_dataset(
            table,
            root_path=base_path,
            partition_cols=[
                "dataset_name",
                "simulation_type",
                "alpa",
                "recall_target"
            ],
            existing_data_behavior="delete_matching"
        )

    def save_interim_results(self, base_path: str, ordered_records) -> None:
        """Save interim results to file"""
        interim_results = ordered_records.copy().drop(
            columns=["p_score"]
        )
        interim_results["dataset_name"] = self.dataset.name
        interim_results["simulation_type"] = self.simulation_type.name
        interim_results["review_update"] = self.current_review_update
        interim_results["run_id"] = self.run_id
        interim_results["recall_target"] = self.recall_target
        interim_results["alpha"] = self.alpha
        interim_results[BATCH_NUMBER] = self.documents_screened

        table = pa.Table.from_pandas(interim_results)

        pq.write_to_dataset(
            table,
            root_path=base_path,
            partition_cols=[
                "dataset_name",
                "simulation_type",
                BATCH_NUMBER
            ],
            #existing_data_behavior="delete_matching"
        )

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
    df[SIM_SEEN_IN_UPDATE] = None
    df["random_sample"] = 0
    df["p_score"] = None
    df[BATCH_NUMBER] = None
    #np.random.seed(42)
    #df['publication_year'] = np.random.rand(len(df))
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
    
    logger.info("split data into the following parts:")
    logger.info(df['review_update'].value_counts())

    df["texts"] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    
    return df

def get_run_id() -> int:
    """Get the SLURM ID from os.environ or set run_id to 1."""
    try:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        return 1

class SampleError(Exception):
    pass

def main(testing: bool = True, workdir: str= "screening_results/"):

    run_id = get_run_id()

    logger.info(f"Initialising living prioritised screening simulation with run_id {run_id}")

    if testing:
        dataset_names = ["Hall_2012"]
        recall_targets = [0.99, 0.95, 0.9, 0.8]
        alphas = [0.05,0.10]
        simulation_types = list(SimulationType)

    else:
        recall_targets = [0.99, 0.95, 0.9, 0.8] 
        alphas = [0.05,0.10] 
        dataset_names = [d.name for d in iter_datasets()]
        simulation_types = list(SimulationType)

    for dataset_name in dataset_names:
        try:
            for recall_target in recall_targets:
                for alpha in alphas:
                    for simulation_level in simulation_types:

                        df = load_data(dataset_name)

                        dataset = ScreeningDataset(df=df, name=dataset_name)
                        dataset.vectorise()
                        for current_review_update in range(N_UPDATES):
                            
                            logger.info(f"Update: {current_review_update}")
                            
                            if current_review_update > 0:
                                dataset.update(df)
                                

                            simulation = ScreeningSimulation(
                                dataset=deepcopy(dataset),
                                current_review_update=current_review_update,
                                simulation_type=simulation_level,
                                recall_target=recall_target,
                                alpha=alpha,
                                run_id=run_id
                            )
                            if current_review_update==0:
                                simulation.clear_dataset()
                                sample_success = simulation.simulate_sample()
                                logger.info(f"{simulation.df["random_sample"].sum()}")
                                
                                if not sample_success:
                                    raise SampleError("Could not identify random sample") 
                                
                            simulation.prioritised_screening()

                            df = simulation.df

                            if current_review_update == N_UPDATES-1:
                                simulation.save_results(f"{workdir}/screening_results/")
        except SampleError as e:
            logger.info(str(e))
            continue

if __name__=="__main__":
    typer.run(main)
