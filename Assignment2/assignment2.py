import matplotlib.pyplot as plt

N = 3
Lambda = 4
mu = 5
alpha = 60

k = 1 # Placeholder value
M = 40 # Large number of states. Truncated here since value of k starts to converge after this


def calculateP0j(j):
	return(k)

def calculateP0Nplusj(alpha, j):
	return(k*(Lambda/(Lambda+alpha))**(j+1))

def calculateP1j(j):
	result = 0
	for i in range(j):
		result += (Lambda**(j-i-1))*(mu**(i))
	result *= k*Lambda/(mu**j)
	return(result)

def calculateP1Nplusj(N, alpha, j):
	if j == 0:
		return(calculateP1j(N))
	else:
		return((((Lambda/(Lambda+alpha))**(j-1))*Lambda - ((Lambda/(Lambda+alpha))**(j))*alpha)*k/mu + Lambda*calculateP1Nplusj(N, alpha, j-1)/mu)




def calculateP00(N, alpha):
	result = 0
	for i in range(N):
		result += calculateP0j(i)
	for i in range(M):
		result += calculateP0Nplusj(alpha, i)
	for i in range(N):
		result += calculateP1j(i)
	for i in range(M):
		result += calculateP1Nplusj(N, alpha, i)
	return(k/result)

def calculateJobs(N, alpha):
	result = 0
	p00 = calculateP00(N,alpha)
	for i in range(N):
		result += calculateP0j(i)*p00*i/k
	for i in range(M):
		result += calculateP0Nplusj(alpha, i)*p00*(N+i)/k
	for i in range(N):
		result += calculateP1j(i)*p00*i/k
	for i in range(M):
		result += calculateP1Nplusj(N, alpha, i)*p00*(N+i)/k
	return(result)

print(calculateJobs(N, alpha))

def calculateUtilization(N, alpha):
	result = 0
	p00 = calculateP00(N,alpha)
	for i in range(N):
		result += calculateP1j(i)*p00/k
	for i in range(M):
		result += calculateP1Nplusj(N, alpha, i)*p00/k
	return(result)

print(calculateUtilization(N, alpha))

utilNDict = {}

for i in range(10):
	utilNDict[i+1] = calculateUtilization(i+1, alpha)

utilalphaDict = {}

for i in range(10,110,10):
	utilalphaDict[i] = calculateUtilization(N, i)

jobsNDict = {}

for i in range(10):
	jobsNDict[i+1] = calculateJobs(i+1, alpha)

jobsalphaDict = {}

for i in range(10,110,10):
	jobsalphaDict[i] = calculateJobs(N, i)


def getDictLabels(dict):
	labels ={}
	if dict == utilalphaDict:
		labels["x"] = "alpha"
		labels["y"] = "Utilization"
		labels["title"] = "Utilization vs. alpha"
	if dict == utilNDict:
		labels["x"] = "N"
		labels["y"] = "Utilization"
		labels["title"] = "Utilization vs. N"
	if dict == jobsalphaDict:
		labels["x"] = "alpha"
		labels["y"] = "Jobs"
		labels["title"] = "Average no. of jobs vs. alpha"
	if dict == jobsNDict:
		labels["x"] = "N"
		labels["y"] = "Jobs"
		labels["title"] = "Average no. of jobs vs. N"
	return(labels)

def getLims(dict):
	lims = []
	if dict == utilalphaDict or dict == utilNDict:
		lims.append(0)
		lims.append(1)
	if dict == jobsalphaDict or dict == jobsNDict:
		lims.append(0)
		lims.append(10)
	return(lims)


def drawPlot(dict):
	plt.plot(list(dict.keys()), list(dict.values()), 'bo')
	plt.xlabel(getDictLabels(dict)["x"])
	plt.ylabel(getDictLabels(dict)["y"])
	plt.title(getDictLabels(dict)["title"])
	plt.ylim((getLims(dict)))
	plt.show()


drawPlot(jobsNDict)
drawPlot(jobsalphaDict)
drawPlot(utilNDict)
drawPlot(utilalphaDict)