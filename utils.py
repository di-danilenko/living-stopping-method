### TO-DO ###
# check how we handle the multiple iterations in the simulated screening results (i.e. seen in iteration 1-3, p-value, etc.)
# add a dynamic review iteration trigger instead of a fixed number of reviews for all datasets
# add reordering based on predictions from the model (experiment 3)

import pandas as pd
dataset_name = "Bos_2018"
RECALL_TARGET = 0.99
df = pd.read_csv(f"simulated_screening/{dataset_name}-{RECALL_TARGET}-results-2-v1.csv")
seen_df = df.loc[df['sim_seen'] == 1]
seen_documents = df.loc[df['sim_seen'] == 1, 'incl']

def plot_screeningprogress(df, seen_documents, title="Cumulative included documents", recall_target=0.8):
    import matplotlib.pyplot as plt
    import pandas as pd
    from buscarpy import retrospective_h0


    df.incl.cumsum().plot()
    plt.savefig(f'simulated_screening/{dataset_name}-{recall_target}-2-v1_incl_cumsum.png')

    fig, ax = plt.subplots()
    seen_documents.cumsum().plot()
    ax.set_xlim(xmax=df.shape[0]+50)
    plt.savefig(f'simulated_screening/{dataset_name}-{recall_target}-2-v1_seen_documents_cumsum.png')

# calculate true recall and work savings
# how do we get seen_df_part?
def calculate_true_recall(seen_df, df):
    true_recall = seen_df.loc[seen_df['sim_seen'] == 1]['incl'].sum()/df['incl'].sum() 
    return true_recall

def calculate_missed_includes(seen_df, df):
    missed_includes = df['incl'].sum() - seen_df.loc[seen_df['sim_seen'] == 1]['incl'].sum()
    return missed_includes

def calculate_work_savings(seen_df, df):
    work_savings = 1 - seen_df.loc[seen_df['sim_seen'] == 1].shape[0] / df.shape[0]
    return work_savings

def plot_screeningprogress_with_stoppingpoints(df, seen_documents, recall_target=0.8, n=100):
    import matplotlib.pyplot as plt
    import os
    from buscarpy import retrospective_h0
    import pandas as pd

    df = pd.read_csv(f"simulated_screening/{dataset_name}-{recall_target}-results-2-v1.csv")
    seen_documents = df.loc[df['sim_seen'] == 1, 'incl']
    
    # batch_size = int(max(100, min(0.1 * df_part.shape[0], 500))) 

    # add different possible recall targets to the graph
    h0 = retrospective_h0(seen_documents, df.shape[0], batch_size=5*n, recall_target=recall_target, plot=False)

    fig, ax = plt.subplots()
    ax.plot(seen_documents.cumsum())
    ax2 = ax.twinx()

    ax2.scatter(h0['batch_sizes'], h0['p'], label=f'recall=={recall_target}')

    # Find the minimum batch size where p < 0.05
    def get_min_batch_size(data): 
        # check whether this also gets us ALL the local minima (points in the screening curve where we'd have stopped in all iterations)
        # Find indices where p < 0.05
        indices = [i for i, value in enumerate(data['p']) if value < 0.05]
        if indices:  # Check if there are any indices found
            # Return the minimum batch size at those indices
            return min(data['batch_sizes'][i] for i in indices)
        return None

    # Get the minimum batch size for the recall target
    min_batch_size = get_min_batch_size(h0)
    if min_batch_size is not None:
        ax.axvline(x=min_batch_size, color = 'steelblue',linestyle='--', label=f'p < 0.05 (Recall {recall_target})')

    ax.set_xlabel('Documents screened')
    ax.set_ylabel('Relevant documents identified')
    ax2.set_ylabel('p score for H0 that recall target missed')

    ax2.legend()

    path = str(os.getcwd()) + "/simulated_screening/" + dataset_name + "-" + str(recall_target) + "-2-v1_simulated-screening-curve-p-line-" + ".png"
    plt.savefig(path)

plot_screeningprogress(df, seen_documents, title="Cumulative included documents", recall_target=RECALL_TARGET)
plot_screeningprogress_with_stoppingpoints(df, seen_documents, recall_target=RECALL_TARGET, n=100)
calculate_missed_includes(seen_df, df)
calculate_true_recall(seen_df, df)
calculate_work_savings(seen_df, df)
