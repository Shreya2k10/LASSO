import numpy as np
from submitProxShreya import solver

# Find out how much loss is the learnt model incurring?
def getObjValue( X, y, wHat ):
	lassoLoss = np.linalg.norm( wHat, 1 ) + pow( np.linalg.norm( X.dot( wHat ) - y, 2 ), 2 )
	return lassoLoss

# Find out how far is the learnt model from the true one in terms of Euclidean distance
def getModelError( wHat, wAst ):
	return np.linalg.norm( wHat - wAst, 2 )

# Force the learnt model to become sparse and then see how well it approximates the true model
def getSupportError( wHat, wAst, k ):
	# Find the k coordinates where the true model has non-zero values
	idxAst = np.abs( wAst ).argsort()[::-1][:k]
	# Find the k coordinates with largest values (absolute terms) in the learnt model
	idxHat = np.abs( wHat ).argsort()[::-1][:k]
	
	# Set up indicator arrays to find the diff between the two
	# Could have used Python's set difference function here as well
	a = np.zeros_like( wAst )
	a[idxAst] = 1
	b = np.zeros_like( wAst )
	b[idxHat] = 1
	return np.linalg.norm( a - b, 1 )//2

Z = np.loadtxt( "train" )
wAst = np.loadtxt( "wAstTrain" )
k = 20

y = Z[:,0]
X = Z[:,1:]

# To avoid unlucky outcomes try running the code several times
numTrials = 5

# Try various timeouts - the timeouts are in seconds
timeouts = np.array( [0.1, 1, 2, 5] )

# Try checking for timeout every 10 iterations
spacing = 10

result = np.zeros( (len( timeouts ), 4) )

for i in range( len( timeouts ) ):
	to = timeouts[i]
	avgObj = 0
	avgDist = 0
	avgSupp = 0
	avgTime = 0
	for t in range( numTrials ):
		(w, totTime) = solver( X, y, to, spacing )
		avgObj = avgObj + getObjValue( X, y, w )
		avgDist = avgDist + getModelError( w, wAst )
		avgSupp = avgSupp + getSupportError( w, wAst, k )
		avgTime = avgTime + totTime
	result[i, 0] = avgObj/numTrials
	result[i, 1] = avgDist/numTrials
	result[i, 2] = avgSupp/numTrials
	result[i, 3] = avgTime/numTrials

np.savetxt( "result2", result, fmt = "%.6f" )