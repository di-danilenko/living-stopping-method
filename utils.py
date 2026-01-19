### TO-DO ###
# current fix of clf.fit results in that we dont go over all the N documents
# restrict to only the last iteration of each result dataframe?
# adjust column names where needed
# add a loop for dataset, review iteration, recall target, confidence level, random state
# add a dynamic recall iteration trigger instead of a fixed number of reviews for all datasets
# add reordering based on predictions from the model

def plot_screeningprogress(dataset_name="Bos_2018", title="Cumulative included documents"):
    import matplotlib.pyplot as plt
    import pandas as pd
    from buscarpy import retrospective_h0
    df = pd.read_csv(f"simulated_screening-2-v1/{dataset_name}-0.8-results-2-v1.csv")
    # df = df.sort_values(by=['sim_order'])
    # sim_order repeats values across different updates of the review
    unique_id_count = df['id'].nunique()
    print("Unique id count:", unique_id_count)
    # TO ADD drop duplicates, keep the last iteration only
    df = df.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)
    print("Dataframe shape after dropping duplicates:", df.shape)
    # We want to somehow display screening effort over different iterations of the review too, though DISCUSS what a good visualisation would be
    print(df['sim_seen'].value_counts())
    # Print summary stats for buscar_p_r_080_1, buscar_p_r_080_2, buscar_p_r_080_3 if present
    for col in ['buscar_p_r080_1', 'buscar_p_r080_2', 'buscar_p_r080_3']:
        if col in df.columns:
            print(f"Summary stats for {col}:")
            print(df[col].describe())
            print()

    df.incl.cumsum().plot()
    plt.savefig(f'simulated_screening-2-v1/{dataset_name}-0.95-results-2-v1_incl_cumsum.png')

    seen_documents = df.loc[df['sim_seen'] == 1, 'incl']

    fig, ax = plt.subplots()
    seen_documents.cumsum().plot()
    ax.set_xlim(xmax=df.shape[0]+50)
    plt.savefig(f'simulated_screening-2-v1/{dataset_name}-0.95-results-2-v1_seen_documents_cumsum.png')

# calculate true recall and work savings
def calculate_true_recall(seen_df_part, df):
    true_recall = seen_df_part.loc[seen_df_part['seen'] == 1]['label_included'].sum()/df['label_included'].sum() 
    return true_recall

def calculate_missed_includes(seen_df_part, df):
    missed_includes = df['label_included'].sum() - seen_df_part.loc[seen_df_part['seen'] == 1]['label_included'].sum()
    return missed_includes

def calculate_work_savings(seen_df_part, df):
    work_savings = 1 - seen_df_part.loc[seen_df_part['seen'] == 1].shape[0] / df.shape[0]
    return work_savings

def plot_screeningprogress_with_stoppingpoints(seen_documents, df, n, dataset_name="Bos_2018"):
    import matplotlib.pyplot as plt
    import os
    from buscarpy import retrospective_h0

    # add different possible recall targets to the graph
    h0_99 = retrospective_h0(seen_documents, df.shape[0], batch_size=5*n, recall_target=0.99, plot=False)
    h0_95 = retrospective_h0(seen_documents, df.shape[0], batch_size=5*n, recall_target=0.95, plot=False)
    h0_9 = retrospective_h0(seen_documents, df.shape[0], batch_size=5*n, recall_target=0.9, plot=False)
    h0_75 = retrospective_h0(seen_documents, df.shape[0], batch_size=5*n, recall_target=0.75, plot=False)

    fig, ax = plt.subplots()

    ax.plot(seen_documents.cumsum())
    ax2 = ax.twinx()

    ax2.scatter(h0_99['batch_sizes'], h0_99['p'], label='recall==0.99')
    ax2.scatter(h0_95['batch_sizes'], h0_95['p'], label='recall==0.95')
    ax2.scatter(h0_9['batch_sizes'], h0_9['p'], label='recall==0.9')
    ax2.scatter(h0_75['batch_sizes'], h0_75['p'], label='recall==0.75')

    # Function to find the minimum batch size where p < 0.05
    def get_min_batch_size(data):
        # Find indices where p < 0.05
        indices = [i for i, value in enumerate(data['p']) if value < 0.05]
        if indices:  # Check if there are any indices found
            # Return the minimum batch size at those indices
            return min(data['batch_sizes'][i] for i in indices)
        return None

    # Get the minimum batch sizes for each recall target
    min_batch_size_99 = get_min_batch_size(h0_99)
    min_batch_size_95 = get_min_batch_size(h0_95)
    min_batch_size_9 = get_min_batch_size(h0_9)
    min_batch_size_75 = get_min_batch_size(h0_75)

    # Add vertical lines for each recall target where p < 0.05
    if min_batch_size_99 is not None:
        ax.axvline(x=min_batch_size_99, color = 'steelblue',linestyle='--', label='p < 0.05 (Recall 0.99)')
    if min_batch_size_95 is not None:
        ax.axvline(x=min_batch_size_95, color= 'orange', linestyle='--', label='p < 0.05 (Recall 0.95)')
    if min_batch_size_9 is not None:
        ax.axvline(x=min_batch_size_9, color= 'forestgreen', linestyle='--', label='p < 0.05 (Recall 0.90)')
    if min_batch_size_75 is not None:
        ax.axvline(x=min_batch_size_75, color= 'firebrick', linestyle='--', label='p < 0.05 (Recall 0.75)')

    ax.set_xlabel('Documents screened')
    ax.set_ylabel('Relevant documents identified')
    ax2.set_ylabel('p score for H0 that recall target missed')

    ax2.legend()

    path = str(os.getcwd()) + "/" + "simulated-screening-curve-p-line-" + dataset_name + ".png"
    plt.savefig(path)

