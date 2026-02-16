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

alpha = 0.05   # 1 - confidence level
recall_targets = [0.99, 0.95, 0.9, 0.8,]
col_names = ["buscar_p_r099", "buscar_p_r095", "buscar_p_r090", "buscar_p_r080"]

# TODO: Let's consider whether we want to include a random sample in the beginning of each review update?
# function to simulate the screening of the random sample in the beginning

def simulate_screening_random_sample(df_part, data_name): 
    
    print("Simulate screening of an initial random set")

    # create a random sample (N = 10% df.shape[0] but min=250 and max=500)
    N = int(max(100, min(0.1 * df_part.shape[0], 500)))

    # exclude datasets where we can't find enough included docs
    
    # Instead of drawing new samples, draw additional documents
    rs = 42
    idx = df_part.sample(n=N, random_state=rs).index
    while (df_part.loc[idx, 'incl'].nunique() <= 1) & ((df_part.loc[idx, 'incl'] == 1).sum() <= 1):
        N+=1
        idx = df_part.sample(n=N, random_state=rs).index
        if N >= df_part.shape[0]*0.5: 
            print(f"Skipping dataset {data_name} as no suitable sample found after even with half the dataset sampled.")
            return 0
        
    df_part.loc[idx, 'sim_seen'] = 1 # we have seen the random sample
    df_part.loc[idx, 'sim_order'] = range(1, len(idx)+1) # we saw them in this order


    # for rs in range(1000):
    #     idx = df_part.sample(n=N, random_state=rs).index
    #     # check if more than one unique 'incl' value AND at least 2 docs where incl == 1
    #     # fmh: this effectively excludes all dataset where either less than 2 docs have incl == 1 or all docs have incl == 1
    #     if (df_part.loc[idx, 'incl'].nunique() > 1) & ((df_part.loc[idx, 'incl'] == 1).sum() > 1):
    #         df_part.loc[idx, 'sim_seen'] = 1 # we have seen the random sample
    #         df_part.loc[idx, 'sim_order'] = range(1, len(idx)+1) # we saw them in this order
    #         random_sample = df_part.loc[idx]
    #         print(f"Used random_state = {rs}")
    #         break
    #     else:
    #         print("Couldn't find a sample withmore than one unique 'incl' value AND at least 2 docs where incl == 1.")
    # else:
    #     # if we get here, no break occurred,no suitable sample found after 1000 tries
    #     print(f"Skipping dataset {data_name} as no suitable sample found after 1000 attempts.")
    #     return 0

    #df.loc[df.sample(n=N, random_state=50824).index, 'sim_seen'] = 1
    print(f"Documents screened: {df_part[df_part['sim_seen'] == 1]['sim_seen'].value_counts()[1]}")
    print(f"Relevant documents identified: {df_part[df_part['sim_seen'] == 1]['incl'].sum()}")
    print(f"Relevance rate across the dataset: {df_part[df_part['sim_seen'] == 1]['incl'].sum()/len(df_part[df_part['sim_seen'] == 1]['incl'])}")

    return df_part

# function to order unseen documents by predicted relevance and "screen" the top n documents

def order_unseen_by_pred_relevance(df_part, x_train, n, sim_step, verbosity=0):
    
    seen_index = df_part[df_part['sim_seen'] == 1].index
    x_train_seen = x_train[seen_index]
    unseen_index = df_part[df_part['sim_seen'] == 0].index
    x_train_unseen = x_train[unseen_index]

    clf = SVC(C=0.25, probability=True, class_weight='balanced')
    # fmh: using standard values for SVM here
    start_time = time.time()
    seen_df_part = df_part[df_part['sim_seen'] == 1]
    clf.fit(x_train_seen, seen_df_part['incl'])
    elapsed_time = time.time() - start_time
    if verbosity > 1:
        print(f"used {len(seen_df_part)} seen docs for training")
        print(f"clf.fit took: {elapsed_time:.2f} seconds")

    # predict relevance of the remaining documents (from the unseen df_part dataset)
    start_time = time.time()
    y_pred = clf.predict_proba(x_train_unseen)
    elapsed_time = time.time() - start_time
    if verbosity > 1:
        print(f"clf.predict_proba took: {elapsed_time:.2f} seconds")

    # put these predictions in the "prediction" column of the dataframe
    # fmh: I think we do not need to store these predictions in the dataframe
    # MC: We do! so we can reorder later
    df_part.loc[unseen_index,'prediction'] = y_pred[:,1]

    # get an n batch with highest_predictions
    #print(np.argsort(np.array(y_pred[:,1]))[:n])
    batch_idx = unseen_index.values[np.argsort(np.array(y_pred[:,1]))[::-1][:n]]  # this gets the indices that would order the prediction values and uses them to select the batch ids
    
    if verbosity > 2:
        print(batch_idx[:10])
        print(y_pred[:,1][np.argsort(np.array(y_pred[:,1]))[::-1][:10]])
    
    df_part.loc[batch_idx,'sim_seen'] = 1
    df_part.loc[batch_idx, 'sim_order'] = range(sim_step+1, sim_step+len(batch_idx)+1)
    current_batch_df = df_part.loc[batch_idx]
    return df_part, current_batch_df

# DD: not sure if we need this or can just add a conditional statement to the previous function
def order_unseen_by_pred_relevance_lev2(df_part, x_train, n, sim_step_lev2, verbosity=0):
    
    # DD: we want to load values only for the random sample and then imitate normal ordering/screening
    # sim_lev2_seen == 1 only for the random sample
    seen_index = df_part[df_part['sim_lev2_seen'] == 1].index 
    x_train_seen = x_train[seen_index]
    unseen_index = df_part[df_part['sim_lev2_seen'] == 0].index
    x_train_unseen = x_train[unseen_index]

    clf = SVC(C=0.25, probability=True, class_weight='balanced')
    # fmh: using standard values for SVM here
    start_time = time.time()
    seen_df_part = df_part[df_part['sim_lev2_seen'] == 1]
    clf.fit(x_train_seen, seen_df_part['incl'])
    elapsed_time = time.time() - start_time

    if verbosity > 1:
        print(f"used {len(seen_df_part)} seen docs for training")
        print(f"clf.fit took: {elapsed_time:.2f} seconds")

    # predict relevance of the remaining documents (from the unseen df_part dataset)
    start_time = time.time()
    y_pred = clf.predict_proba(x_train_unseen)
    elapsed_time = time.time() - start_time
    if verbosity > 1:
        print(f"clf.predict_proba took: {elapsed_time:.2f} seconds")

    # get an n batch with highest_predictions
    #print(np.argsort(np.array(y_pred[:,1]))[:n])
    batch_idx = unseen_index.values[np.argsort(np.array(y_pred[:,1]))[::-1][:n]]  # this gets the indices that would order the prediction values and uses them to select the batch ids
    
    if verbosity > 2:
        print(batch_idx[:10])
        print(y_pred[:,1][np.argsort(np.array(y_pred[:,1]))[::-1][:10]])
    
    # update these values directly in df_part, they will be reset before the next time
    df_part.loc[batch_idx,'sim_lev2_seen'] = 1
    df_part.loc[batch_idx, 'sim_lev2_order'] = range(sim_step_lev2+1, sim_step_lev2+len(batch_idx)+1)
    current_batch_df_lev2 = df_part.loc[batch_idx]
    return df_part, current_batch_df_lev2


# main function to run a single simulation
def run_simulation(df, data_name, recall_target, col_name, verbosity=0):

    print(f"starting experiment 1 on dataset {data_name} ({df.shape[0]}) with recall target: {recall_target}")

    # initiate the default columns 'sim_seen', 'model', and 'id'
    # force string type on 'title' and 'abstract'
    df['sim_seen'] = 0
    df['title'] = df['title'].apply(str)
    df['abstract'] = df['abstract'].apply(str)
    df['id'] = df.index.astype(str)
    # fmh: I would try to store this info differently (if running with different types of models at all)
    #df['model'] = "SVC(C=0.25, probability=True, class_weight='balanced'), retrain=False"

    df.sort_values('publication_year', inplace=True) # this is the inital dataset
    df.reset_index(inplace=True) # ValueError: cannot insert level_0, already exists (worked with a dataset and recall_target 0.99 and 0.95 but then broke)

    # use 'title' and 'abstract' for classification
    texts = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    # create a TF-IDF vectorizer object that will transform text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(texts.fillna(''))

    sim_step = 0 # tracks the current step (in terms of data points) of the simulation

    # introduce an explicit split of the data into 3 parts (assuming there are 3 reviews
    # spread out in time according to the literature growth)

    # fmh: here, one could introduce different possibilities to allocate the data to slices of review_updates

    # split data into 3 equally sized subsets
    df['review_update'] = 0
    for y in range(0, 4):
        part_startindex = y*round(df.shape[0]/3)
        part_endindex = (y+1)*round(df.shape[0]/3) - 1
        df.loc[part_startindex:part_endindex, 'review_update'] = y
    
    print("split data into the following parts:")
    print(df['review_update'].value_counts())

    df_result = pd.DataFrame() # this is where we'll be writing the results

    for current_review_update in range(0, max(df['review_update']+1)):
        # get the slice of data for each review update
        df_part = df[df['review_update'] <= current_review_update].copy()
        n_new_docs = (df['review_update'] == current_review_update).sum()
        # fmh: x is the length of the current review update plus all previous review updates
        x = df_part.shape[0]
        n_seen = df_part[df_part['sim_seen']==1].shape[0]
        print(f"\nReview update #{current_review_update}: This will cover a total of {x} documents,")
        print(f"with {n_seen} already seen and {n_new_docs} new documents")

        # for the baseline review, find a random sample
        if current_review_update == 0:

            df_part = simulate_screening_random_sample(df_part, data_name)
            
            # DD: interrupt simulation and move on to next dataset if we couldn't find a random sample 
            if isinstance(df_part, int):
                raise TypeError("Skipping this dataset as no suitable random sample could be found.")
            df_result = df_part[df_part['sim_seen'] == 1].sort_values('sim_order')
            sim_step = len(df_result)

        # simulate screening effort within this partial review
        batch_size = int(max(100, min(0.1 * df_part.shape[0], 500)))
        print(f'Simulating stopping method with batch size = {batch_size}')
        # seen_df['sim_order'] = range(1*y, len(seen_df) + 1*y) 
        # this way, we'll avoid having multiple documents with the same simulated order of screening

        # simulate screening effort within this review update
        while df_result.shape[0] < x:
            
            df_part, current_batch_df = order_unseen_by_pred_relevance(df_part, x_train, batch_size, sim_step, verbosity=verbosity)
            sim_step += len(current_batch_df)
            print("length of current batch:", len(current_batch_df))
            print("current sim_step:", sim_step)

            # fmh: the old structure of the code simulated that each of the documents is seen and then
            # calculates the probability of the criterion retrospectively.
            # I would implement this differently: After each iteration above, the criterion is calculated and
            # if it meets the stopping criterion, then the current order (according to predicted probabilities) is preserved

            df_result = pd.concat([df_result, current_batch_df])
            
            # calculate p-values for a recall target
            # fmh: using calculate_h0 instead of retrospective
            p_val = calculate_h0(df_result['incl'], df_part.shape[0], recall_target=recall_target)

            if len(df_result) > len(df_part):
                print('something went wrong ...')
                break

            df_result.loc[current_batch_df.index, col_name] = p_val

            if verbosity > 1:
                print("sim step:", sim_step)
                print("p_val", p_val)
                print("len df_result:", df_result.shape[0])

            if p_val < alpha:
                print('reached p_value < 1 - confidence level')
                break

        # write all recorded values in df
        df.update(df_part)

        # check recall
        n_rel_seen = df_part[df_part['sim_seen'] == 1]['incl'].sum()
        n_rel_tot = df_part['incl'].sum()
        recall = n_rel_seen / n_rel_tot

        if verbosity > 0:
            print(f"Relevant seen: {n_rel_seen} of {n_rel_tot}")
            print("Recall at stopping:", recall)
            print("Work saved:", df_part['sim_seen'].sum() / len(df_part))

    # add all unseen data to results set
    df_result = pd.concat([df_result, df[df['sim_seen'] == 0]])

    return df_result

# function to run a simulation of review update > 0 
# incl. reordering based on predicted relevance (simulation level 2)
# dd: we re-run the simulation as we might have actually made different 
# decisions to stop in review updates

def run_simulation_with_reordering(df, data_name, recall_target, col_name, verbosity=0):
    
    print(f"starting experiment 2 (simulation with reordering) on dataset {data_name} ({df.shape[0]}) with recall target: {recall_target}")

    # initiate the default columns 'sim_seen', 'model', and 'id'
    # force string type on 'title' and 'abstract'
    df['sim_seen'] = 0
    df['title'] = df['title'].apply(str)
    df['abstract'] = df['abstract'].apply(str)
    df['id'] = df.index.astype(str)

    # sort by publication year and reset index
    df.sort_values('publication_year', inplace=True) 
    df.reset_index(inplace=True) 
    
    # use 'title' and 'abstract' for classification
    texts = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    # create a TF-IDF vectorizer object that will transform text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(texts.fillna(''))

    sim_step = 0 # tracks the current step (in terms of data points) of the simulation

    # introduce an explicit split of the data into 3 parts (assuming there are 3 reviews
    # spread out in time according to the literature growth)

    # fmh: here, one could introduce different possibilities to allocate the data to slices of review_updates

    # split data into 3 equally sized subsets
    df['review_update'] = 0
    for y in range(0, 4):
        part_startindex = y*round(df.shape[0]/3)
        part_endindex = (y+1)*round(df.shape[0]/3) - 1
        df.loc[part_startindex:part_endindex, 'review_update'] = y
    
    print("split data into the following parts:")
    print(df['review_update'].value_counts())

    df_result = pd.DataFrame() # this is where we'll be writing the results

    # for current_review_update in range(1, max(df['review_update']+1))
 
    for current_review_update in range(0, max(df['review_update']+1)):
        # get the slice of data for each review update
        df_part = df[df['review_update'] <= current_review_update].copy()
        n_new_docs = (df['review_update'] == current_review_update).sum()
        # fmh: x is the length of the current review update plus all previous review updates
        x = df_part.shape[0]
        n_seen = df_part[df_part['sim_seen']==1].shape[0]
        print(f"\nReview update #{current_review_update}: This will cover a total of {x} documents,")
        print(f"with {n_seen} already seen and {n_new_docs} new documents")

        # for the baseline review, find a random sample
        if current_review_update == 0:

            df_part = simulate_screening_random_sample(df_part, data_name)
            # DD: interrupt simulation and move on to next dataset if we couldn't find a random sample 
            if isinstance(df_part, int):
                raise TypeError("Skipping this dataset as no suitable random sample could be found.")
            df_result = df_part[df_part['sim_seen'] == 1].sort_values('sim_order')
            sim_step = len(df_result)

            batch_size = int(max(100, min(0.1 * df_part.shape[0], 500)))
            print(f'Simulating stopping method with batch size = {batch_size}')

            # simulate screening effort within this review update
            while df_result.shape[0] < x:
                
                df_part, current_batch_df = order_unseen_by_pred_relevance(df_part, x_train, batch_size, sim_step, verbosity=verbosity)
                sim_step += len(current_batch_df)
                # print("length of current batch:", len(current_batch_df))
                print("current sim_step:", sim_step)

                # fmh: the old structure of the code simulated that each of the documents is seen and then
                # calculates the probability of the criterion retrospectively.
                # I would implement this differently: After each iteration above, the criterion is calculated and
                # if it meets the stopping criterion, then the current order (according to predicted probabilities) is preserved

                df_result = pd.concat([df_result, current_batch_df])
                
                # calculate p-values for a recall target
                # fmh: using calculate_h0 instead of retrospective
                p_val = calculate_h0(df_result['incl'], df_part.shape[0], recall_target=recall_target)

                if len(df_result) > len(df_part):
                    print('something went wrong ...')
                    break

                df_result.loc[current_batch_df.index, col_name] = p_val

                if verbosity > 1:
                    print("sim step:", sim_step)
                    print("p_val", p_val)
                    print("len df_result:", df_result.shape[0])

                if p_val < alpha:
                    print('reached p_value < 1 - confidence level')
                    break

            # write all recorded values in df
            df.update(df_part)

            # check recall
            n_rel_seen = df_part[df_part['sim_seen'] == 1]['incl'].sum()
            n_rel_tot = df_part['incl'].sum()
            recall = n_rel_seen / n_rel_tot

            if verbosity > 0:
                print(f"Relevant seen: {n_rel_seen} of {n_rel_tot}")
                print("Recall at stopping:", recall)
                print("Work saved:", df_part['sim_seen'].sum() / len(df_part))
        
        else:

            batch_size = int(max(100, min(0.1 * df_part.shape[0], 500)))
            print(f'Simulating stopping method with batch size = {batch_size}')

            # simulate screening effort within this review update
            while df_result.shape[0] < x:
                
                df_part, current_batch_df = order_unseen_by_pred_relevance(df_part, x_train, batch_size, sim_step, verbosity=verbosity)
                sim_step += len(current_batch_df)
                # print("length of current batch:", len(current_batch_df))
                print("current sim_step:", sim_step)

                df_result = pd.concat([df_result, current_batch_df])
                
                # calculate p-values for a recall target
                # fmh: using calculate_h0 instead of retrospective
                # dd: this is where we add reordering 
                # rows where sim_order != 1: keep original order
                
                # TODO: Let's consider whether we need to hold these constant 
                # Actually, these don't have a prediction, so we should keep them at the front
                # But not all sample docs have sim_order == 1
                mask = (df_result['sim_order'] == 1 ) | (pd.isna(df_result['prediction']))
                df_keep = df_result[mask]

                # rows where sim_order != 1: sort by predicted relevance (scores from the model)
                # dd: does this really work or do we have to add a full simulation level 2? 
                # Ah, this was supposed to be ascending = False!
                df_sort = df_result[~mask].sort_values('prediction', ascending=False)

                # combine: kept rows first (unchanged), then sorted rows
                df_reordered = pd.concat([df_keep, df_sort], ignore_index=False)  

                p_val = calculate_h0(df_reordered['incl'], df_part.shape[0], recall_target=recall_target)

                if len(df_reordered) > len(df_part):
                    print('something went wrong ...')
                    break
                
                # dd: does this still work as expected?
                df_result.loc[current_batch_df.index, col_name] = p_val
                df_result.loc[df_reordered.index, "sim_reordered"] = np.arange(df_reordered.shape[0])

                if verbosity > 1:
                    print("sim step:", sim_step)
                    print("p_val", p_val)
                    print("len df_result:", df_result.shape[0])

                if p_val < alpha:
                    print('reached p_value < 1 - confidence level')
                    break

            # write all recorded values in df
            # dd: this way, we store the original, step-wise order of screening
            # and only calculate the p-values based on the re-ordered dataframe
            df.update(df_part)

            # check recall
            n_rel_seen = df_part[df_part['sim_seen'] == 1]['incl'].sum()
            n_rel_tot = df_part['incl'].sum()
            recall = n_rel_seen / n_rel_tot

            if verbosity > 0:
                print(f"Relevant seen: {n_rel_seen} of {n_rel_tot}")
                print("Recall at stopping:", recall)
                print("Work saved:", df_part['sim_seen'].sum() / len(df_part))

    # add all unseen data to results set
    df_result = pd.concat([df_result, df[df['sim_seen'] == 0]])

    return df_result

def run_simulation_with_level2(df, data_name, recall_target, col_name, verbosity=0):
    
    print(f"starting experiment 3 (simulation with level2) on dataset {data_name} ({df.shape[0]}) with recall target: {recall_target}")

    # initiate the default columns 'sim_seen', 'model', and 'id'
    # force string type on 'title' and 'abstract'
    df['sim_seen'] = 0
    df['sim_lev2_seen'] = 0
    df['title'] = df['title'].apply(str)
    df['abstract'] = df['abstract'].apply(str)
    df['id'] = df.index.astype(str)

    # sort by publication year and reset index
    df.sort_values('publication_year', inplace=True) 
    df.reset_index(inplace=True) 
    
    # use 'title' and 'abstract' for classification
    texts = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    # create a TF-IDF vectorizer object that will transform text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(texts.fillna(''))

    sim_step = 0 # tracks the current step (in terms of data points) of the simulation

    # introduce an explicit split of the data into 3 parts (assuming there are 3 reviews
    # spread out in time according to the literature growth)

    # fmh: here, one could introduce different possibilities to allocate the data to slices of review_updates

    # split data into 3 equally sized subsets
    df['review_update'] = 0
    for y in range(0, 4):
        part_startindex = y*round(df.shape[0]/3)
        part_endindex = (y+1)*round(df.shape[0]/3) - 1
        df.loc[part_startindex:part_endindex, 'review_update'] = y
    
    print("split data into the following parts:")
    print(df['review_update'].value_counts())

    df_result = pd.DataFrame() # this is where we'll be writing the results
 
    for current_review_update in range(0, max(df['review_update']+1)):
        # get the slice of data for each review update
        df_part = df[df['review_update'] <= current_review_update].copy()
        n_new_docs = (df['review_update'] == current_review_update).sum()
        # fmh: x is the length of the current review update plus all previous review updates
        x = df_part.shape[0]
        n_seen = df_part[df_part['sim_seen']==1].shape[0]
        print(f"\nReview update #{current_review_update}: This will cover a total of {x} documents,")
        print(f"with {n_seen} already seen and {n_new_docs} new documents")

        # for the baseline review, find a random sample
        if current_review_update == 0:

            df_part = simulate_screening_random_sample(df_part, data_name)

            # DD: interrupt simulation and move on to next dataset if we couldn't find a random sample
            if isinstance(df_part, int):
                raise TypeError("Skipping this dataset as no suitable random sample could be found.")

            # keep the random sample the same in level 2 simulation
            df_part['sim_lev2_seen'] = df_part['sim_seen']
            df_part['sim_lev2_order'] = df_part['sim_order']
            df_result = df_part[df_part['sim_seen'] == 1].sort_values('sim_order')
            sim_step = len(df_result)

            batch_size = int(max(100, min(0.1 * df_part.shape[0], 500)))
            print(f'Simulating stopping method with batch size = {batch_size}')

            # simulate screening effort within this review update
            while df_result.shape[0] < x:
                
                df_part, current_batch_df = order_unseen_by_pred_relevance(df_part, x_train, batch_size, sim_step, verbosity=verbosity)
                sim_step += len(current_batch_df)
                # print("length of current batch:", len(current_batch_df))
                print("current sim_step:", sim_step)

                df_result = pd.concat([df_result, current_batch_df])
                
                # calculate p-values for a recall target
                p_val = calculate_h0(df_result['incl'], df_part.shape[0], recall_target=recall_target)

                if len(df_result) > len(df_part):
                    print('something went wrong ...')
                    break

                df_result.loc[current_batch_df.index, col_name] = p_val

                if verbosity > 1:
                    print("sim step:", sim_step)
                    print("p_val", p_val)
                    print("len df_result:", df_result.shape[0])

                if p_val < alpha:
                    print('reached p_value < 1 - confidence level')
                    break

            # write all recorded values in df
            df.update(df_part)

            # check recall
            n_rel_seen = df_part[df_part['sim_seen'] == 1]['incl'].sum()
            n_rel_tot = df_part['incl'].sum()
            recall = n_rel_seen / n_rel_tot

            if verbosity > 0:
                print(f"Relevant seen: {n_rel_seen} of {n_rel_tot}")
                print("Recall at stopping:", recall)
                print("Work saved:", df_part['sim_seen'].sum() / len(df_part))
        
        # for the updates, simulate with level2
        else:

            batch_size = int(max(100, min(0.1 * df_part.shape[0], 500)))
            print(f'Simulating stopping method with batch size = {batch_size}')

            # keep the random sample the same in level 2 simulation
            # TODO: copy sim_seen and sim_order for the random sample only
            # TODO: fix KeyError: 'sim_order'
            # 626-627, 647-649
            df_part['sim_lev2_seen'] = df_part['sim_seen']
            df_part['sim_lev2_order'] = df_part['sim_order']

            # simulate screening effort within this review update
            while df_result.shape[0] < x:
                
                df_part, current_batch_df = order_unseen_by_pred_relevance(df_part, x_train, batch_size, sim_step, verbosity=verbosity)
                sim_step += len(current_batch_df)
                # print("length of current batch:", len(current_batch_df))
                print("current sim_step:", sim_step)

                df_result = pd.concat([df_result, current_batch_df])
                
                # calculate p-values for a recall target
                # re-order documents by running a sim level2 and comparing sim_level2_seen with sim_seen (gaps)
                # order_unseen_by_pred_relevance without splitting data into rev 0 and 1
                df_result_lev2 = pd.DataFrame() # this is where we'll be writing the results of sim level 2
                sim_step_lev2 = 0
                
                # DD: reset sim_lev2_seen and sim_lev_order every time apart from for the random sample
                # TODO: check this is working as intended
                mask = (df_part['sim_order'] == 1) | (pd.isna(df_part['prediction']))
                df_part['sim_lev2_seen'] = df_part['sim_lev2_seen'].mask(mask, 0)
                df_part['sim_lev2_order'] = df_part['sim_lev2_order'].mask(mask, 0)

                while df_result_lev2.shape[0] < x: 

                    # write screening results in sim_lev2_seen and sim_lev2_order
                    # use existing values from df_part, where random sample has correct sim_lev2_seen and sim_lev2_order values
                    df_part, current_batch_df_lev2 = order_unseen_by_pred_relevance_lev2(df_part, x_train, batch_size, sim_step_lev2, verbosity=verbosity)
                    sim_step_lev2 += len(current_batch_df_lev2)
                    # print("length of current batch:", len(current_batch_df))
                    print("current sim_step_lev2:", sim_step_lev2)

                    df_result_lev2 = pd.concat([df_result_lev2, current_batch_df_lev2])
                
                # re-order documents by sim_lev2_order
                # df_result = df_result.sort_values(by='sim_lev2_order')
                p_val = calculate_h0(df_result_lev2['incl'], df_part.shape[0], recall_target=recall_target)

                if len(df_result_lev2) > len(df_part):
                    print('something went wrong ...')
                    break

                # TODO: check that this is working as intended 
                df_result.loc[current_batch_df_lev2.index, col_name] = p_val
                df_result.loc[current_batch_df_lev2.index, 'sim_lev2_seen'] = 1
                df_result.loc[current_batch_df_lev2.index, 'sim_lev2_order'] = df_result_lev2.loc[current_batch_df_lev2.index, 'sim_lev2_order']

                if verbosity > 1:
                    print("sim step:", sim_step)
                    print("p_val", p_val)
                    print("len df_result:", df_result.shape[0])

                if p_val < alpha:
                    print('reached p_value < 1 - confidence level')
                    break

            # write all recorded values in df
            
            df.update(df_part)

            # check recall
            # add the checks described above in here
            n_rel_seen = df_part[df_part['sim_seen'] == 1]['incl'].sum()
            # TODO: check that this is working, returns different numbers of total includes somehow from run to run
            n_rel_tot = df_part['incl'].sum()
            recall = n_rel_seen / n_rel_tot

            if verbosity > 0:
                print(f"Relevant seen: {n_rel_seen} of {n_rel_tot}")
                print("Recall at stopping:", recall)
                print("Work saved:", df_part['sim_seen'].sum() / len(df_part))

    # add all unseen data to results set
    df_result = pd.concat([df_result, df[df['sim_seen'] == 0]])

    return df_result

if __name__ == "__main__":

    testing = True
    sim_type = '3' # '1', '2', '3'
    SIM_FUNCS = {
    1: run_simulation,
    2: run_simulation_with_reordering,
    3: run_simulation_with_level2,
    }

    if testing:

        dataset_name = 'Wassenaar_2017'
        recall_target = 0.8

        results_file = f"{dataset_name}-{recall_target}-results-{sim_type}.csv"
        results_folder = Path("simulated_screening")
        os.makedirs(results_folder,exist_ok=True)
    
        df = pd.read_csv(f"synergy_dataset/{dataset_name}.csv")
        df['publication_year'] = np.random.rand(len(df))
        df = df.rename(columns={"label_included":"incl"})
        
        sim_time = time.time()
        sim_func = SIM_FUNCS[int(sim_type)]    
        df_res = sim_func(df.copy(), dataset_name, recall_target, 'buscar_p_r080', verbosity=2)
        sim_time = time.time() - sim_time

        print(f"Simulation took: {sim_time:.2f} seconds")

        df_res.to_csv(results_folder / results_file, index=False)
        print(f"Saved a csv of simulated screening ({sim_type}) for dataset {dataset_name} (recall target: {recall_target})")

    else:
        # run through all simulations
        for d in iter_datasets(): # somehow, starts with Bos_2018 rather than Appenzeller-Herzog_2019

            df = pd.read_csv(f"synergy_dataset/{d.name}.csv")
            if df.shape[0] < 3000: #or df['incl'].sum()/df.shape[0] < 0.01:
                continue

            for recall_target, col_name in zip(recall_targets,col_names):

                # MC: I just commented out this block for the time being to focus on the other bits.
                # #prepare the directory and filename
                # results_file = f"{d.name}-{recall_target}-results-{sim_type}.csv"
                # results_folder = Path("simulated_screening")
                # os.makedirs(results_folder,exist_ok=True)
                # if (results_folder / results_file).exists():
                #     print(f"Skipping {results_folder / results_file}—file already exists.")  
                #     continue

                # df = df.rename(columns={"label_included":"incl"})
                # df_res = run_simulation(df.copy(), d.name, recall_target, col_name)

                # df_res.to_csv(results_folder / results_file, index=False)
                # print(f"saved a csv of simulated screening ({sim_type}) for dataset {d.name} (recall target: {recall_target})")

                #prepare the directory and filename
                results_file = f"{d.name}-{recall_target}-results-{sim_type}.csv"
                results_folder = Path("simulated_screening_reordering")
                os.makedirs(results_folder,exist_ok=True)
                if (results_folder / results_file).exists():
                    print(f"Skipping {results_folder / results_file}—file already exists.")  
                    continue

                df = df.rename(columns={"label_included":"incl"})

                # DD: add if returned 0, interrupt the simulation and move on to the next dataset
                try: 
                    sim_func = SIM_FUNCS[int(sim_type)] 
                    df_res = sim_func(df.copy(), d.name, recall_target, col_name)
                    df_res.to_csv(results_folder / results_file, index=False)
                    print(f"saved a csv of simulated screening ({sim_type}) for dataset {d.name} (recall target: {recall_target})")
                except TypeError:
                    print(f"Skipping dataset {d.name} due to error in simulation.")
                    continue

        print('end of iteration over all datasets')
