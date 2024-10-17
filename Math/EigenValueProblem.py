import numpy as np
from scipy.stats import ortho_group
from scipy import sparse
print()

SMALLEST,LARGEST = 0,1

# Ax = ax, return a,x
def EigenValue(A,type):
    if type not in [SMALLEST,LARGEST]:
        print('Type must be SMALLEST or LARGEST')
        return
    n = A.shape[0]
    x = np.ones(n)
    while np.linalg.norm(A@x-(x.T@A@x)*x) > 1e-10:
        if type == LARGEST:     x = A@x
        elif type == SMALLEST:  x = np.linalg.solve(A,x)
        x /= np.sqrt(x.T@x)
    return x.T@A@x,x

def TestEigenValue(A):
    value, vector = EigenValue(A, SMALLEST)
    print('smallest eigen value: ', value)
    value, vector = EigenValue(A, LARGEST)
    print('largest eigen value: ', value)


def GenerateTestMatrixA(n):
    eigenValues = np.sort(np.random.uniform(1, 10, size=n))
    print('eigen values are ', eigenValues)
    Sigma = np.eye(n)
    for i in range(n):Sigma[i,i] = eigenValues[i]
    U = ortho_group.rvs(n)
    return U@Sigma@U.T

def GeneralizedEigenValue(A,B,type):
    if type not in [SMALLEST,LARGEST]:
        print('Type must be SMALLEST or LARGEST')
        return
    n = A.shape[0]
    x = np.ones(n)
    while np.linalg.norm(A@x-(x.T@A@x)*B@x) > 1e-10:
        if type == LARGEST:     x = np.linalg.solve(B,A@x)
        elif type == SMALLEST:  x = np.linalg.solve(A,B@x)
        x /= np.sqrt(x.T@B@x)
    return x.T@A@x,x

def TestGeneralizedEigenValue(A,B):
    values, _ = np.linalg.eig(np.linalg.inv(B)@A)
    print('eigen values are ', values)
    value, vector = GeneralizedEigenValue(A, B, SMALLEST)
    print('smallest eigen value: ', value)
    value, vector = GeneralizedEigenValue(A, B, LARGEST)
    print('largest eigen value: ', value)

def TestScipySparseEigenValue():
    values = np.arange(1,100).astype(np.float64)
    np.random.shuffle(values)
    row,col,data = [],[],[]
    for i,v in enumerate(values):
        row.append(i)
        col.append(i)
        data.append(v)
    A = sparse.coo_matrix((data,(row,col)),shape=(len(values),len(values)))
    largest, _ = sparse.linalg.eigsh(A,1,which='LM')
    smallest,_ = sparse.linalg.eigsh(A,1,which='SM')
    print(smallest,largest)

TestScipySparseEigenValue()

def Main():
    n = 2

    A = GenerateTestMatrixA(n)
    print('testing A -------- ')
    TestEigenValue(A)

    B = np.random.randn(n,n)
    B = B.T@B
    C = np.linalg.inv(B)@A
    print('testing B_inv@A -------- ')
    TestEigenValue(C)
    print('testing A,B -------- ')
    TestGeneralizedEigenValue(A,B)

Main()