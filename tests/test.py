A = [0,2,3,4,6]
B = [4,3,6,1,8]
print(min(zip(A,B), key=lambda x: x[1])[0])