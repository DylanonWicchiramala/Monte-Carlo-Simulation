import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def monticalosim(winrate:float, riskrewardratio:float, riskpertrade:float, initbalance:float|int, numtrade:int, tiralnum:int):
    
    def rebalance(prebalance:float, win:bool, riskrewardratio:float=riskrewardratio, riskpertrade:float=riskpertrade):
        if prebalance<=0:
            return 0
        if win:
            balance = prebalance * (1 + riskpertrade * riskrewardratio)
        else:
            balance = prebalance * (1 - riskpertrade)
        return max(balance, 0)
        

    w_l = np.random.binomial(1, winrate, (tiralnum, numtrade))


    tradepaths = np.zeros(w_l.shape, dtype=np.float64)
    for ti in range(len(w_l)):
        for i in range(len(w_l[ti])):
            if i==0:
                prebalance = initbalance
            else:
                prebalance = tradepaths[ti, i-1]
            tradepaths[ti, i] = rebalance(prebalance, w_l[ti, i-1])
            
    return tradepaths


def drawdown(a):
    acc_max = np.maximum.accumulate(a, axis=-1)
    return -1+(a/acc_max)


def plot_old(tradepaths, n_sample=10, logscale=True):
    sns.set_style("whitegrid")
    g = sns.lineplot(tradepaths[:n_sample,:].T, dashes=0)
    if logscale:
        g.set(yscale='log')
    g.axhline(y=0)
    g.get_legend().set_visible(False)
    plt.show()
    

def plot(tradepaths, n_sample=10, logscale=True):
    # Convert the tradepaths array to a DataFrame for easier manipulation with plotly express
    df = pd.DataFrame(tradepaths[:n_sample, :].T)
    
    # Create a long-form DataFrame for plotly express
    df_long = df.reset_index().melt(id_vars=['index'], var_name='path', value_name='value')
    
    # Create the plot using plotly express
    fig = px.line(df_long, x='index', y='value', color='path', line_dash='path')
    
    # Set log scale if specified
    if logscale:
        fig.update_yaxes(type='log')
    
    # Add a horizontal line at y=0
    fig.add_shape(type="line", x0=df_long['index'].min(), x1=df_long['index'].max(), 
                  y0=0, y1=0, line=dict(color="Black", width=1))
    
    # Update layout to hide the legend if desired
    fig.update_layout(showlegend=False)
    
    fig.show()


initbalance = 10000
tradepaths = monticalosim(
    winrate = .50,
    riskrewardratio = 1.5,
    riskpertrade = 0.1,
    initbalance = initbalance,
    numtrade = 10*12*1,
    tiralnum = 1000
)   

dd = drawdown(tradepaths)
endbalances = tradepaths[:, -1]
n_margincall = sum(endbalances<0.1)

fm = "{:,.2f}".format

print(
    f"num margincal: {n_margincall}",
    f"mean: {fm(endbalances.mean())}",
    f"1%: {fm(np.percentile(endbalances, 1))}  ({fm(np.percentile(endbalances, 1)-initbalance)})",
    f"5%: {fm(np.percentile(endbalances, 5))}  ({fm(np.percentile(endbalances, 5)-initbalance)})",
    f"10%: {fm(np.percentile(endbalances, 10))}  ({fm(np.percentile(endbalances, 10)-initbalance)})",
    f"20%: {fm(np.percentile(endbalances, 20))}  ({fm(np.percentile(endbalances, 20)-initbalance)})",
    f"50%: {fm(np.percentile(endbalances, 50))}  ({fm(np.percentile(endbalances, 50)-initbalance)})",
    f"max drawdown: {fm(np.percentile(dd, 1)*100)}%",
      sep="\n")

    
# plot(tradepaths, 5,logscale=0)