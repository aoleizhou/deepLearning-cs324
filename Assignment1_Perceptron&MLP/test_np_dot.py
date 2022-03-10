import numpy as np

# in = 2, out = 3: W=out*in=3*2, x(in*1)
# W=np.array( [[1,2],
#     [1,2],
#     [1,2]])
# x=np.array([1,1])
# W2=np.array([[1,1,1],[2,2,2]])
# x2=np.array([1,1])
# print(np.dot( x,W))
# print(np.dot(x2, W2))

# x=np.array([-1,2,-3,4])
# y=[1,1,1,1]
# print(np.where(x<0, 0,y))

# x=[1,2,3]
# y=[3,4,7]
# print(np.sum(x*np.log(y)))
# print(np.dot(x,np.log(y)))

# x = [6,7,8]
# for i in range(len(x)-1, -1, -1):
#     print(x[i])
# print(round(3.1415926))

# x=np.array([0.12123,0.50001,0.49999])
# y=np.array([0,1,0])
# print((np.around(x)==y).all())
# a=np.array([3,6,9])
# print(a/3)
# a=np.array([[-1,2],[3,4]])
# print(np.maximum(0,a))
#
# from sklearn import datasets
# x, t = datasets.make_moons(1000)
# print(x[:5])
# print(t[:5])

# print([2,4,6]/[2,2,3])
# print(np.zeros((3,1)))

# x= [1,2,3]
# print(-7.13771696e-02/800)#-8.92214620e-05

x=[[3,4,5]]
y=[[1,1,1]]
print(np.subtract(x,y))
# for i in range(len(x)-1, -1, -1):
#     print(i)