import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def result(mean,cov,X):

    pdf = np.divide(np.exp(np.divide(np.dot( np.dot(np.transpose(np.subtract(X,mean)),inv(cov)),np.subtract(X,mean)),
            -2) ),np.multiply(np.power(det(cov),0.5) ,(44/7)) )


    return pdf

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix



    d = np.shape(X)[1]
    nc = int(np.max(y))
    
    means = np.empty((d, nc));

    covmat = np.zeros(d)
    for i in range (1, nc + 1):
        a = np.where(y==i)[0]
        trainData = X[a,:]
        means[:, i-1] = np.mean(trainData, axis=0).transpose()
        covmat = covmat + (np.shape(trainData)[0]-1) * np.cov(np.transpose(trainData))


    covmat = (1.0/(np.shape(X)[0] - nc)) * covmat;

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    

    nc = int(np.max(y))
    d = np.shape(X)[1]
    means = np.empty((d, nc));

    covmats = []
    for i in range (1, nc+1):
        c = np.where(y==i)[0]
        train_data = X[c,:]
        means[:, i-1] = np.mean(train_data, axis=0).transpose()
        n=np.transpose(train_data)
        m=np.cov(n)
        covmats.append(m)

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    ypred = np.empty([Xtest.shape[0], 1])

    for a in range(Xtest.shape[0]):
        predict = 0
        classnum = 0
        for index in range(means.shape[1]):
            p = result(means[:,index],covmat,Xtest[a])
            if p > predict:
                predict = p
                classnum = index
        ypred[a,0] = (classnum +1)

    correct = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            correct = correct + 1;
    acc = correct

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    ypred = np.empty([Xtest.shape[0], 1])

    for a in range(Xtest.shape[0]):
        predict = 0
        classnum = 0
        for index in range(means.shape[1]):
            p = result(means[:,index],covmats[index],Xtest[a])
            if p > predict:
                predict = p
                classnum = index
        ypred[a,0] = (classnum +1)

    correct = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            correct = correct + 1;
    acc = correct

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # w = (inverse(transpose(X) * X)) * transpose(X) * y
    
    w = np.zeros((X.shape[1], 1))
    transposeX_X = np.dot(X.T, X)
    invTransposeX_X = inv(transposeX_X)
    transposeX_Y = np.dot(X.T, y)

    w = np.dot(invTransposeX_X, transposeX_Y)
                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1 
  
    # w =  inverse((λI + transpose(X) * X)) * transpose(X) * y
    
    transposeX_X = np.dot(X.T, X)
    lambdaIdentity = np.dot(lambd, np.identity(X.shape[1]))
    temp = lambdaIdentity + transposeX_X
    invTemp = inv(temp)
    transposeX_Y = np.dot(X.T, y)
    
    w = np.dot(invTemp, transposeX_Y)

    return w


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # mse = (1/N)(Summation(i:1,N)(Yi - transpose(w)*(Xi))^2)
    
    transposeW_Xi = np.dot(Xtest,w)
    diff = np.subtract(ytest, transposeW_Xi)
    squaredDiff = np.multiply(diff, diff)
    summation = np.sum(squaredDiff)
    
    N = Xtest.shape[0]
    
    mse = np.sum(summation)/N

    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda  
    
    #Formula to implement
    # J(w) =  (1/2)(Summation(i:1,N)(Yi - transpose(w)*(Xi))^2) + (1/2)λ(w*transpose(w));
    # Gradient = transpose(X)(Xw−y) + λw

    error = 0
    
    transposew_X = np.dot(w.T, X.T)

    diff = np.subtract(y.T, transposew_X)
    squaredDiff = np.multiply(diff, diff)
    summationN = np.sum(squaredDiff)
    transposeW_W = np.dot(w.T, w)
    lambdaTransposeW_W = np.multiply(lambd, transposeW_W)

    error = (summationN / 2) + (lambdaTransposeW_W / 2)

    transposeX_X = np.dot(X.T, X)
    transposeY_X = np.dot(y.T, X)
    transposeX_XW =  np.dot(w.T, transposeX_X)
    lambdaW = np.multiply(lambd, w)
    sub = np.subtract(lambdaW, transposeY_X)
    error_grad = np.add(sub, transposeX_XW)


    error = error.flatten()
    error_grad = error_grad.flatten()

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    
    N = x.shape[0]
    Xd = np.ones((N, p + 1));
    for i in range(1, p + 1):
        Xd[:, i] = x ** i;
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 50}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses4)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()