import numpy as np
import random as rnd
import time as tm


# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def next_cord(d,t):
    #return d[rnd.randint(0,d.shape[0]-1)]
    return (t-1)%d




################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
    (n, d) = X.shape
    t = 0
    totTime = 0
    
    # w is the model vector and will get returned once timeout happens
    w = np.zeros( (d,) )
    tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

    # You may reinitialize w to your liking here
    # You may also define new variables here e.g. step_length, mini-batch size etc
    w = np.zeros_like(w)
    z_calc= np.ones_like(w)
    for a in range(d):
        z_calc[a]=np.dot(X[:,a],X[:,a])
    y_pred=X@w
    d_active=np.arange(d)
################################
# Non Editable Region Starting #
################################
    while True:
        t = t + 1
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                print(t)
                return (w, totTime)
            else:
                tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
        #for a in range(d):
#        if(t%(5*d)==0):
 #           d_active=np.arange(d)
        a=next_cord(d,t)
        #y_pred=np.delete(X, a, axis=1).dot(np.delete(w, a))
        #if(t%10==1):
            #print(f'error:{np.linalg.norm(y-y_pred,1)}')
                #return(w,totTime)
        #y_pred=X@w
        #y_pred= linalg.blas.dgemm(1,X,w).reshape(n)-w[a]*X[:,a]
        old_w=w[a]
        old_w_x=old_w*X[:,a]
        p=np.dot(X[:,a],(y-y_pred+old_w_x))
        z=z_calc[a]
        #new_w=max((p-1*np.sign(p)*0.5)/z,0)
        if (p<-0.5):
            new_w=(0.5+p)/z
        elif(p>0.5):
            new_w=(p-0.5)/z
        else:
            new_w=0
        if(new_w!=old_w):
            y_pred=y_pred-old_w_x+new_w*X[:,a]
            w[a]=new_w
           # if(new_w==0):
            #    d_active=d_active[d_active!=a]
        
        # Write all code to perform your method updates here within the infinite while loop
        # The infinite loop will terminate once timeout is reached
        # Do not try to bypass the timer check e.g. by using continue
        # It is very easy for us to detect such bypasses which will be strictly penalized
        
        # Please note that once timeout is reached, the code will simply return w
        # Thus, if you wish to return the average model (as is sometimes done for GD),
        # you need to make sure that w stores the average at all times
        # One way to do so is to define a "running" variable w_run
        # Make all GD updates to w_run e.g. w_run = w_run - step * delw
        # Then use a running average formula to update w
        # w = (w * (t-1) + w_run)/t
        # This way, w will always store the average and can be returned at any time
        # In this scheme, w plays the role of the "cumulative" variable in the course module optLib
        # w_run on the other hand, plays the role of the "theta" variable in the course module optLib
        
    return (w, totTime) # This return statement will never be reached