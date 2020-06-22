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

    


################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
    #np.errstate(all="raise")
    (n, d) = X.shape
    t = 0
    totTime = 0
    
    # w is the model vector and will get returned once timeout happens
    w = np.zeros( (d,) )
    tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

    XT=2*X.T
    
    x_2=np.dot(XT,X)
    x_y =np.dot(XT,y)
    #Deter=np.linalg.norm(xt_x,2)
    #print(1/Deter)
    #step=1/Deter
    step=0.04
    #w=np.ones(d,)
    #w= np.random.rand(d,)
    
    #v=0
    

    # You may reinitialize w to your liking here
    # You may also define new variables here e.g. step_length, mini-batch size etc
################################
# Non Editable Region Starting #
################################
    while True:
        t = t + 1
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                print(t,np.linalg.norm(w),np.linalg.norm(g),theta)
                #return (w, totTime)
                break
            else:
                tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
        g=np.dot(x_2,w) - x_y
    '''  
        theta=2/(t+1)
        beta=(1-theta)*w-theta*v
        u=beta-step*g
        
        w_new=np.sign(u)*np.maximum(np.abs(u)-1,0)
        v=w+(w_new-w)/theta
        w=w_new
    '''
        u=w-step*g
        w_new=np.sign(u)*np.maximum(np.abs(u)-step,0)
        d_w=w_new-w
        dt=(t-2)/(t+1)
        w = w_new + d_w*dt
        
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