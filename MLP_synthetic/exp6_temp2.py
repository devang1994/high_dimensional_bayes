__author__ = 'da368'

import matplotlib as mpl
mpl.use('Agg')
from mlp_rand_proj import mlp_synthetic_proj

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

epochs = 1000
refError = 0.729677179036


if __name__ == "__main__":

    hidden_width=10
    mini_batchsize=5
    #proj_width=50
    eval_pts = [10,20,50,100,200, 300, 500, 700, 1000, 1300, 1500, 1800, 2000]
    for proj_width in [80]:
        for i in range(-5,1):
            L2reg=pow(10,i)
            test_costs = []
            for numTP in eval_pts:
                print L2reg,'  ', numTP, ' ', proj_width
                fin_cost_train, fin_cost_test = mlp_synthetic_proj(L2reg=L2reg, hidden_width=hidden_width, numTrainPoints=numTP,
                                                          mini_batchsize=mini_batchsize,proj_width=proj_width)
                test_costs.append(fin_cost_test)
            plt.plot(eval_pts, test_costs, label='L2reg:{}'.format(L2reg))

        tArray = np.ones(2000) * refError
        plt.plot(range(2000), tArray, label='Reference', color='black', linewidth=2.0)
        plt.legend()
        plt.xlabel('Num Training Points')
        plt.ylabel('Error')
        plt.savefig('logs/exp6b{}ep1000.png'.format(proj_width), dpi=400)
        plt.close()




