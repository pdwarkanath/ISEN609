import numpy as np
import matplotlib.pyplot as plt


alpha1 = 0.7
alpha2 = 0.6
B1 = 3
B2 = 4

varTup = (B1, B2, alpha1, alpha2)


def getStateSpace(B1,B2):
	return(list(range(-B2,B1+1)))

S = getStateSpace(B1,B2)
print("The state space is " + str(S))

def getPMatrix(B1, B2, alpha1, alpha2):
	P = [[0]*(B1+B2+1)]
	P[0][0] =1-alpha1
	P[0][1] = alpha1


	for i in range(B1+B2-1):
		P.append([0]*(B1+B2+1))
		P[i+1][i] = (1-alpha1)*alpha2
		P[i+1][i+1] = alpha1*alpha2 + (1-alpha1)*(1-alpha2)
		P[i+1][i+2] = alpha1*(1-alpha2)

	P.append([0]*(B1+B2+1))
	P[B1+B2][B1+B2-1] = alpha2
	P[B1+B2][B1+B2] =1-alpha2
	return(np.matrix(P))



def getPi(P):
	I = np.identity(P.shape[0])	
	Q = I-P
	Q[:,P.shape[0]-1] = 1
	a = [0]*(P.shape[0]-1)
	a.append(1)
	pi = np.matrix([a])*Q.getI()
	return(pi.round(4))


def getBin1Components(B1, B2, alpha1, alpha2):
	P = getPMatrix(B1, B2, alpha1, alpha2)
	pi = getPi(P)
	result = 0
	for i in range(1,B1+1):
		result += i*pi.item(B2 + i)
	return(round(result,4))

def getBin2Components(B1, B2, alpha1, alpha2):
	P = getPMatrix(B1, B2, alpha1, alpha2)
	pi = getPi(P)
	result = 0
	for i in range(B2):
		result += (B2-i)*pi.item(i)
	return(round(result,4))


print(getBin1Components(B1, B2, alpha1, alpha2))
print(getBin2Components(B1, B2, alpha1, alpha2))

def getItems(B1, B2, alpha1, alpha2):
	P = getPMatrix(B1, B2, alpha1, alpha2)
	pi = getPi(P)
	items = 0

	for i in range(B2):
		items += pi.item(i)*alpha1
	
	items += pi.item(B2)*alpha1*alpha2

	for i in range(B1):
		items += pi.item(B2+i+1)*alpha2
	return(round(items,4))

print(getItems(B1, B2, alpha1, alpha2))


# changing B1

Bin1Comp = []
Bin2Comp = []
ShippedProds = []
B1range = list(range(1,11))

for i in B1range:
	Bin1Comp.append(getBin1Components(i, B2, alpha1, alpha2))
	Bin2Comp.append(getBin2Components(i, B2, alpha1, alpha2))
	ShippedProds.append(getItems(i, B2, alpha1, alpha2))

def drawPlots(x, xname):
	plt.plot(x, Bin1Comp, 'r^', label = 'Bin 1')
	plt.plot(x, Bin2Comp, 'bx', label = 'Bin 2')
	plt.legend(loc="upper left")
	plt.ylim(0,10)
	plt.ylabel("Average Components")
	plt.xlim(0,max(x)+x[1]-x[0])
	plt.xlabel(xname)
	plt.show()

	plt.plot(x, ShippedProds, 'go')
	plt.ylim(0,1)
	plt.xlim(0,max(x)+x[1]-x[0])
	plt.ylabel("Average Shipped Products")

	plt.xlabel(xname)
	plt.show()

drawPlots(B1range, "Bin 1 Capacity")

# Changing B2

Bin1Comp = []
Bin2Comp = []
ShippedProds = []
B2range = list(range(1,11))
for i in B2range:
	Bin1Comp.append(getBin1Components(B1, i, alpha1, alpha2))
	Bin2Comp.append(getBin2Components(B1, i, alpha1, alpha2))
	ShippedProds.append(getItems(B1, i, alpha1, alpha2))
print(Bin1Comp)
drawPlots(B2range, "Bin 2 Capacity")


# Changing alpha1

Bin1Comp = []
Bin2Comp = []
ShippedProds = []
alpha1range = np.arange(0.1, 1.0, 0.1)
for i in alpha1range:
	Bin1Comp.append(getBin1Components(B1, B2, i, alpha2))
	Bin2Comp.append(getBin2Components(B1, B2, i, alpha2))
	ShippedProds.append(getItems(B1, B2, i, alpha2))

drawPlots(alpha1range, "Machine 1 success rate")

# Changing alpha2

Bin1Comp = []
Bin2Comp = []
ShippedProds = []
alpha2range = np.arange(0.1, 1.0, 0.1)
for i in alpha2range:
	Bin1Comp.append(getBin1Components(B1, B2, alpha1, i))
	Bin2Comp.append(getBin2Components(B1, B2, alpha1, i))
	ShippedProds.append(getItems(B1, B2, alpha1, i))

drawPlots(alpha2range, "Machine 2 success rate")