# load all packages
import pandas as pd
from synergy_dataset import iter_datasets
import matplotlib.pyplot as plt
import os
from buscarpy import retrospective_h0
import timeit
import time
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 20)

# load all synergy datasets incl. PY
for d in iter_datasets():
    globals()[d.name] = pd.read_csv(f"synergy_datasets/{d.name}.csv") # I guess there is another script somewhere that gets these and loads the publication year? Would be good to commit this
    globals()[d.name] = globals()[d.name].rename(columns={"label_included":"incl"})
    print(f"have loaded {d.name} dataset")

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
recall_targets = [1.0, 0.99, 0.95, 0.9, 0.8,]
col_names = ["buscar_p_r100", "buscar_p_r099", "buscar_p_r095", "buscar_p_r090", "buscar_p_r080"]

for d in iter_datasets():

    df = globals()[d.name]
    if df.shape[0] < 3000: #or df['incl'].sum()/df.shape[0] < 0.01:
        continue

    for recall_target, col_name in zip(recall_targets,col_names):
        print(f"starting experiment 2-v1 on dataset {d.name} ({df.shape[0]}) with recall target: {recall_target}")

        #prepare the directory and filename
        results_file = f"{d.name}-{recall_target}-results-2-v1.csv"
        results_folder = Path("simulated_screening-2-v1")
        os.makedirs(results_folder,exist_ok=True)
        if (results_folder / results_file).exists():
            print(f"Skipping {results_folder / results_file}—file already exists.")  
            continue 

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
        y = 1

        # introduce an explicit split of the data into 3 parts (assuming there are 3 reviews
        # spread out in time according to the literature growth)

        while y <= 3:
            # get the slice of data for each of the 3 reviews
            part_endindex = y*round(df.shape[0]/3)
            df_part = df[:part_endindex]
            x = df_part.shape[0]
            n_seen = df_part[df_part['sim_seen']==1].shape[0]
            print(f"this review number {y} will cover {x} documents, of which {n_seen} have already been seen")

            # for the 1st review (baseline review), find a random sample
            if y == 1:
                # create a random sample (N = 10% df.shape[0] but min=250 and max=500)
                N = int(max(100, min(0.1 * df_part.shape[0], 500)))

                # Instead of doing this, I think it would make more sense to keep drawing random records until the conditions are met
                # Otherwise, for difficult datasets with low prevalence, we only use them with nice draws, where in real life we would
                # not be able to simply generate a new sample, we'd only be able to keep drawing new documents
                # exclude datasets where we can't find enough included docs
                for rs in range(1000):
                    idx = df_part.sample(n=N, random_state=rs).index
                    # check if more than one unique 'incl' value AND at least 2 docs where incl == 1
                    if (df_part.loc[idx, 'incl'].nunique() > 1) & ((df_part.loc[idx, 'incl'] == 1).sum() > 1):
                        df_part.loc[idx, 'sim_seen'] = 1 # we have seen the random sample
                        df_part.loc[idx, 'sim_order'] = range(1, len(idx)+1) # we saw them in this order
                        print(f"Used random_state = {rs}")
                        break
                    else:
                        print("Couldn't find a sample withmore than one unique 'incl' value AND at least 2 docs where incl == 1.")
                else:
                    # if we get here, no break occurred,no suitable sample found after 1000 tries
                    print(f"Skipping dataset {d.name} as no suitable sample found after 1000 attempts.")
                    continue

                #df.loc[df.sample(n=N, random_state=50824).index, 'sim_seen'] = 1
                print(f"Documents screened: {df_part[df_part['sim_seen'] == 1]['sim_seen'].value_counts()[1]}")
                print(f"Relevant documents identified: {df_part[df_part['sim_seen'] == 1]['incl'].sum()}")
                print(f"Relevance rate across the dataset: {df_part[df_part['sim_seen'] == 1]['incl'].sum()/len(df_part[df_part['sim_seen'] == 1]['incl'])}")

            # simulate screening effort within this partial review
            n = int(max(100, min(0.1 * df_part.shape[0], 500)))
            print(f'Simulating stopping methods with batch size = {n}')
            seen_df_part = df_part[df_part['sim_seen'] == 1] # started with a random sample
            df_result = seen_df_part
            # seen_df_part['sim_order'] = range(1*y, len(seen_df_part) + 1*y) 
            # this way, we'll avoid having multiple documents with the same simulated order of screening
            print(seen_df_part.shape[0])

            # simulate screening effort within this partial review
            while seen_df_part.shape[0] < x:
                seen_index = df_part[df_part['sim_seen'] == 1].index
                x_train_seen = x_train[seen_index]
                full_index = df_part.index
                x_train_full = x_train[full_index]
                unseen_index = df_part[df_part['sim_seen'] == 0].index
                x_train_unseen = x_train[unseen_index]

                # fit the model on the seen documents
                # either retrain or load best-performing
                # if we retrain here it takes very LONG

                retrain = False
                if retrain:
                    clf = GridSearchCV(pipeline, parameters)
                    clf.fit(
                        seen_df.dropna(subset=['text','incl'])['text'], 
                        seen_df.dropna(subset=['text','incl'])['incl'],
                    )
                    clf.best_estimator_
                    results = pd.DataFrame(clf.cv_results_)
                    results.head()
                    f1_scores = [results[f"split{x}_test_score"].values[0] for x in range(5)]
                    print(np.mean(f1_scores))

                    fig, ax = plt.subplots()

                    ax.boxplot(f1_scores)
                    ax.set_ylim((0,1))
                    ax.grid(axis="y")
                    ax.set_xticklabels(["Best performing model"])
                    ax.set_ylabel('F1 score')

                else:
                    clf = SVC(C=0.25, probability=True, class_weight='balanced')
                    part1_time = timeit.timeit(stmt=lambda:
                    # no time to retrain, we'll just take the pipeline with the best performance from last time
                    clf.fit(x_train_seen, seen_df_part.dropna(subset=['text','incl'])['incl']), number=1) # We should probably drop these NAs at the beginning?
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
                    # Yes, df_part contains all documents, if I understand correctly, and pdf_part is just the latest batch
                    # they will be added in a simulated order
                    print(seen_df_part.shape[0])

            df_result = pd.concat([df_result, seen_df_part], ignore_index=True)
            
            # calculate p-values for a recall target 
            seen_documents = df_result['incl']
            res = retrospective_h0(seen_documents, df_result.shape[0], batch_size=n, recall_target=recall_target)

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

            y +=1
        df_result.to_csv(results_folder / results_file, index=False)
        print(f"saved a csv of simulated screening for dataset {d.name} (recall target: {recall_target})")    

        # seems like it's mostly working but we need to only plot the last iteration, i suppose? cause at the moment we "recycle" the seen documents, hence the final dataset has more rows than the real data

"""clf.fit(x_train_seen, seen_df_part.dropna(subset=['text','incl'])['incl']), number=1)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dianada/hpctest/venv/lib/python3.12/site-packages/sklearn/base.py", line 1389, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dianada/hpctest/venv/lib/python3.12/site-packages/sklearn/svm/_base.py", line 197, in fit
    X, y = validate_data(
           ^^^^^^^^^^^^^^
  File "/home/dianada/hpctest/venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2961, in validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dianada/hpctest/venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 1389, in check_X_y
    check_consistent_length(X, y)
  File "/home/dianada/hpctest/venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 475, in check_consistent_length
    raise ValueError(
ValueError: Found input variables with inconsistent numbers of samples: [3252, 4715] """