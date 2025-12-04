# load all packages
import pandas as pd
from synergy_dataset import iter_datasets
import matplotlib.pyplot as plt
import os
from buscarpy import calculate_h0
import timeit
import time
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 20)

# add the model specification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.exceptions import UndefinedMetricWarning

def assign_batch(order,n):
    # Batch sizes of n increment, adjust as needed
    return ((order) // n + 1) * n

# here we define the whole ML pipeline starting from creating features to training a classifier
pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SVC(probability=True,  class_weight="balanced")),
])

# now we define the parameter space we want to search. This gets computationally expensive the
# more parameters we search, as we fit a model for every unique combination of parameters (there are fancier and smarter ways to do this)
# decreased the number of parameters to simplify computation

parameters = [
    {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__min_df': (5, 10, 15),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__kernel': ['rbf'],
        # because only the rbf kernel has the gamma parameters, we do this separately
        'clf__gamma': [1e-3, 1e-4], 
        'clf__C': [1, 1e1, 1e2]
    },
    {
        'vect__max_df': (0.5,),
        'vect__min_df': (5, 10, 15),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__kernel': ['linear'],
        'clf__C': [1e2, 1e3]
    }
]

# 2. simulate ML-assisted prioritised screening for time *t*, *t+1*, and *t+2* without level 2 simulation
# update 'sim_order'  based on the order we are screening the documents in (level 1 simulation)
# fmh: took out recall target = 1
recall_targets = [0.99, 0.95, 0.9, 0.8,]
col_names = ["buscar_p_r099", "buscar_p_r095", "buscar_p_r090", "buscar_p_r080"]


# function to simulate the screening of the random sample in the beginning

def simulate_screening_random_sample(df_part, data_name):
        
    # create a random sample (N = 10% df.shape[0] but min=250 and max=500)
    N = int(max(100, min(0.1 * df_part.shape[0], 500)))

    # exclude datasets where we can't find enough included docs
    for rs in range(1000):
        idx = df_part.sample(n=N, random_state=rs).index
        # check if more than one unique 'incl' value AND at least 2 docs where incl == 1
        # fmh: his effectively excludes all dataset where either less than 2 docs have incl == 1 or all docs have incl == 1
        if (df_part.loc[idx, 'incl'].nunique() > 1) & ((df_part.loc[idx, 'incl'] == 1).sum() > 1):
            df_part.loc[idx, 'sim_seen'] = 1 # we have seen the random sample
            df_part.loc[idx, 'sim_order'] = range(1, len(idx)+1) # we saw them in this order
            print(f"Used random_state = {rs}")
            break
        else:
            print("Couldn't find a sample withmore than one unique 'incl' value AND at least 2 docs where incl == 1.")
    else:
        # if we get here, no break occurred,no suitable sample found after 1000 tries
        print(f"Skipping dataset {data_name} as no suitable sample found after 1000 attempts.")
        return 0

    #df.loc[df.sample(n=N, random_state=50824).index, 'sim_seen'] = 1
    print(f"Documents screened: {df_part[df_part['sim_seen'] == 1]['sim_seen'].value_counts()[1]}")
    print(f"Relevant documents identified: {df_part[df_part['sim_seen'] == 1]['incl'].sum()}")
    print(f"Relevance rate across the dataset: {df_part[df_part['sim_seen'] == 1]['incl'].sum()/len(df_part[df_part['sim_seen'] == 1]['incl'])}")

    return df_part


def order_unseen_by_pred_relevance(df_part, seen_df, x_train, n):
    
    seen_index = df_part[df_part['sim_seen'] == 1].index
    x_train_seen = x_train[seen_index]
    unseen_index = df_part[df_part['sim_seen'] == 0].index
    x_train_unseen = x_train[unseen_index]

    retrain = False
    if retrain:
        clf.fit(seen_df.dropna(subset=['text','incl'])['text'], seen_df.dropna(subset=['text','incl'])['incl'])
        clf.best_estimator_
        results = pd.DataFrame(clf.cv_results_)
        results.head()
        f1_scores = [results[f"split{x}_test_score"].values[0] for x in range(5)]
        print(np.mean(f1_scores))

        fig, ax = plt.subplots()

        ax.boxplot(f1_scores)
        ax.set_ylim([0,1])
        ax.grid(axis="y")
        ax.set_xticklabels(["Best performing model"])
        ax.set_ylabel('F1 score')

    else:
        clf = SVC(C=0.25, probability=True, class_weight='balanced')
        # fmh: what about the other parameters? Are they just set to standard values here?
        part1_time = timeit.timeit(stmt=lambda:
        # no time to retrain, we'll just take the pipeline with the best performance from last time
        # fmh: If I understand the code correctly, the model is trained on the newly seen data,
        # but no parameter tuning is done? (side note: timeit is for measuring fast routines)
        clf.fit(x_train_seen, seen_df_part.dropna(subset=['text','incl'])['incl']), number=1)
        print(f"clf.fit took: {part1_time:.2f} seconds")

        # predict relevance of the remaining documents (from the entire dataset)
        start_time = time.time()
        y_pred = clf.predict_proba(x_train_unseen)
        elapsed_time = time.time() - start_time
        print(f"clf.predict_proba took: {elapsed_time:.2f} seconds")

        # put these predictions in the "prediction" column of the dataframe
        df_part.loc[unseen_index,'prediction'] = y_pred[:,1]
        pdf_part = df_part.loc[unseen_index].sort_values('prediction', ascending=False)

        # get an n batch with highest_predictions
        batch_idx = pdf_part[:n].index
        valid_batch_idx = [idx for idx in batch_idx if idx in df_part.loc[unseen_index].index]
        df_part.loc[valid_batch_idx,'sim_seen'] = 1
        pdf_part.loc[valid_batch_idx,'sim_seen'] = 1
        # seen_df = df[df['seen'] == 1]
        seen_df_part = pd.concat([seen_df_part, pdf_part[:n]], ignore_index=True) # does it make a difference whether we use df_part or pdf_part here?
                                                                                    # fmh: yes, pdf is the sorted view on df_part, it is important to use this one
        # they will be added in a simulated order
        print(seen_df_part.shape[0])

        return seen_df_part


# main function to run a single simulation
def run_simulation(df, data_name, recall_target, col_name):

    print(f"starting experiment 2-v1 on dataset {data_name} ({df.shape[0]}) with recall target: {recall_target}")

    #prepare the directory and filename
    results_file = f"{data_name}-{recall_target}-results-2-v1.csv"
    results_folder = Path("simulated_screening-2-v1")
    os.makedirs(results_folder,exist_ok=True)
    if (results_folder / results_file).exists():
        print(f"Skipping {results_folder / results_file}—file already exists.")  
        return 0 

    # initiate the default columns 'sim_seen', 'model', and 'id'
    # force string type on 'title' and 'abstract'
    df['sim_seen'] = 0
    df['title'] = df['title'].apply(str)
    df['abstract'] = df['abstract'].apply(str)
    df['id'] = df.index.astype(str)
    df['model'] = "SVC(C=0.25, probability=True, class_weight='balanced'), retrain=False"
    # create a 'text' column from 'title' and 'abstract'
    df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    seen_df = df[df['sim_seen'] == 1]
    x = df.shape[0]

    # create a TF-IDF vectorizer object that will transform text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(df['text'].fillna(''))

    # do loops as long as our seen df is smaller than our labelled df
    df = df.sort_values('publication_year') # this is the inital dataset
    df_result = pd.DataFrame() # this is where we'll be writing the results
    print(df_result.shape[0], df.shape[0])

    # introduce an explicit split of the data into 3 parts (assuming there are 3 reviews
    # spread out in time according to the literature growth)

    # fmh: here, one could introduce different possibilities to allocate the data to slices of review_updates
    df['review_update'] = 0
    for y in range(0, 4):
        part_startindex = y*round(df.shape[0]/3)
        part_endindex = (y+1)*round(df.shape[0]/3)
        df.loc[part_startindex:part_endindex,'review_update'] = y

    for current_review_update in range(0, max(df['review_update']+1)):
        # get the slice of data for each review update
        df_part = df[df['review_update'] <= current_review_update]
        n_new_docs = (df['review_update'] == current_review_update).sum()
        # fmh: x is the length of the current review and previous review updates
        x = df_part.shape[0]
        n_seen = df_part[df_part['sim_seen']==1].shape[0]
        print(f"This review update {current_review_update} will cover a total of {x} documents,")
        print(f"of which {n_seen} have already been seen,")
        print(f"and {n_new_docs} new documents")

        # for the baseline review, find a random sample
        if current_review_update == 0:

            df_part = simulate_screening_random_sample(df_part, data_name)

        # simulate screening effort within this partial review
        n = int(max(100, min(0.1 * df_part.shape[0], 500)))
        print(f'Simulating stopping methods with batch size = {n}')
        seen_df_part = df_part[df_part['sim_seen'] == 1] # started with a random sample
        df_result = seen_df_part
        # seen_df_part['sim_order'] = range(1*y, len(seen_df_part) + 1*y) 
        # this way, we'll avoid having multiple documents with the same simulated order of screening
        print(seen_df_part.shape[0])

        # simulate screening effort within this review update
        while seen_df_part.shape[0] < x:

            seen_df_part = order_unseen_by_pred_relevance(df_part, seen_df, x_train)

            # fmh: the old structure of the code simulated that each of the documents is seen and then
            # calculates the probability of the criterion retrospectively.
            # I would implement this differently: After each iteration above, the criterion is calculated and
            # if it meets the stopping criterion, then the current order (according to predicted probabilities) is preserved
            df_result = pd.concat([df_result, seen_df_part], ignore_index=True)
            
            # calculate p-values for a recall target 
            seen_documents = df_result['incl']

            # fmh: using calculate_h0 instead of retrospective
            res = calculate_h0(seen_documents, df_result.shape[0], recall_target=recall_target)

            # fmh: this needs to be adapted accordingly
            res_df = pd.DataFrame(res)
            res_df = res_df.rename(columns={'p': col_name})
            res_df = res_df.astype({'batch_sizes':'int32'})

            df_result['batch_size_group'] = [assign_batch(idx, n) for idx in df_result.index]
            df_result = df_result.merge(
                res_df[[col_name, 'batch_sizes']], 
                left_on='batch_size_group', right_on='batch_sizes', how='left')
            
                # drop the helper column to avoid conflict on next iteratio
            df_result.drop(columns=['batch_sizes','batch_size_group'], inplace=True)
            print(f"merged p-values for {recall_target}")

            # this way we can store p-values for different reviews (updates) within the living review 
            col_name_y = col_name + '_' + str(y)
            df_result = df_result.rename(columns={col_name: col_name_y})
            # assume all documents where batch > stopping_batch sim_seen = 0 in df_result
            # how do we avoid premature cutoff for all future iterations? (i.e. if the desirable p-value had been reached in a prior iteration)? 
            # seems like for some reviews with very low relevance rate, the stopping method can be triggered prematurely too 
            # which is something we would like to keep and show but need to built in a workaround for this error
            
            # we need to first reset the sim_seen column
            df_result.loc[df_result[col_name_y] >= 0.05, 'sim_seen'] = 1
            df_result.loc[df_result[col_name_y] < 0.05, 'sim_seen'] = 0
            # was the recall target and desired p-value reached at all? 
            print(f"The lowest p-value for the recall target {recall_target} was {min(df_result[col_name_y])}")

            ### ADDED BECAUSE THE SAME RECORD WAS SEEN / NOT SEEN IN DIFFERENT UPDATES WHICH CAUSED MULTIPLE ENTRIES PER ID -- NEED TO RESOLVE ### 
            # Identify the 'id's present in seen_df_part
            seen_ids = set(df_result.loc[df_result['sim_seen'] == 1, 'id'])
            # Update 'sim_seen' to 1 where 'id' is in seen_ids
            df.loc[df['id'].isin(seen_ids), 'sim_seen'] = 1


    df_result.to_csv(results_folder / results_file, index=False)
    print(f"saved a csv of simulated screening for dataset {data_name} (recall target: {recall_target})")

    return df_result


if __name__ == "__main__":

    testing = True

    if testing:
        dataset_name = 'Wolters_2018'
        df = pd.read_csv(f"synergy_dataset/{dataset_name}.csv")
        print(df.columns)
        run_simulation(df, dataset_name, 0.8, 'buscar_p_r080')

    else:
        # run through all simulations
        for d in iter_datasets():

            df = pd.read_csv(f"synergy_datasets/{d.name}.csv")
            if df.shape[0] < 3000: #or df['incl'].sum()/df.shape[0] < 0.01:
                continue

            for recall_target, col_name in zip(recall_targets,col_names):
                run_simulation(df, d.name, recall_target, col_name)

        print('end of iteration over all datasets')
