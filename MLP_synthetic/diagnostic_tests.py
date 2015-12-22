__author__ = 'da368'

from load_synthetic import load_synthetic as load
import numpy as np

X_train, X_test, y_train, y_test = load()

print np.std(y_test)

print np.var(y_test)


#print np.std(y_train)
print np.mean(y_test)

print np.shape(y_test)[0]
print y_test.size


    # best_test=10
    # bestL2=0
    # bestWi=0
    # for i in range(-6,4):
    #     for j in [5,10,20,50,100,200]:
    #         L2=pow(10,i)
    #         try:
    #             fin_cost_train,fin_cost_test=run_test(L2reg=L2, hidden_width=j)
    #         except:
    #             print 'some err'
    #
    #         if fin_cost_test < best_test:
    #             best_test=fin_cost_test
    #             bestL2=L2
    #             bestWi=j
    #             print best_test
    # print best_test,bestL2,bestWi