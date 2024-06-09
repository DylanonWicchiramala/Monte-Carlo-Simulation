import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot(tradepaths, n_sample=10, logscale=True):
    sns.set_style("whitegrid")
    g = sns.lineplot(tradepaths[:n_sample,:].T, dashes=0)
    if logscale:
        g.set(yscale='log')
    g.axhline(y=0)
    g.get_legend().set_visible(False)
    plt.show()
    

initbalance = 5000
tradepaths = monticalosim(
    winrate = .56,
    riskrewardratio = 1.45,
    riskpertrade = 0.1,
    initbalance = initbalance,
    numtrade = 18*8,
    tiralnum = 600
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
    f"70%: {fm(np.percentile(endbalances, 70))}  ({fm(np.percentile(endbalances, 70)-initbalance)})",
    f"max drawdown: {fm(np.percentile(dd, 0)*100)}%",
      sep="\n")

    
# plot(tradepaths, 20,logscale=1)