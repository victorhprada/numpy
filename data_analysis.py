import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/alura-cursos/numpy/dados/apples_ts.csv'

try:
    data = np.loadtxt(url, delimiter=',', usecols=np.arange(1, 88, 1))
    print("Data loaded successfully!")
    print(f"Shape of the data: {data.shape}")
except FileNotFoundError:
    print("Error: Could not find the file in the link'")
except Exception as e:
    print(f"An error occurred: {str(e)}")

print(data.ndim)
print(data.size)
print(data.shape)
print(data.T)

data_transport = data.T
datas = data_transport[:,0]
prices = data_transport[:,1:6]

datas = np.arange(1,88,1)

plt.plot(datas, prices[:,0])
plt.show()

Moscow = prices[:,0]
kaliningrad = prices[:,1]
Petersburg = prices[:,2]
Krasnodar = prices[:,3]
Ekaterinburg = prices[:,4]


Moscow_ano1 = Moscow[0:12] 
Moscow_ano2 = Moscow[12:24]
Moscow_ano3 = Moscow[24:36]
Moscow_ano4 = Moscow[36:48]

plt.plot(np.arange(1,13, 1), Moscow_ano1)
plt.plot(np.arange(1,13, 1), Moscow_ano2)
plt.plot(np.arange(1,13, 1), Moscow_ano3)
plt.plot(np.arange(1,13, 1), Moscow_ano4)
plt.legend(['Ano 1', 'Ano 2', 'Ano 3', 'Ano 4'])
plt.show()

print(np.array_equal(Moscow_ano3, Moscow_ano4))
print(np.allclose(Moscow_ano3, Moscow_ano4, 10))

x = datas
y = 2*x+80
plt.plot(datas, Moscow)
plt.plot(x,y)
plt.show()

Moscow_at_square = np.sqrt(np.sum(np.power(Moscow-y, 2)))
print(Moscow_at_square)

y = 0.52*x+80
plt.plot(datas, Moscow)
plt.plot(x,y)
plt.show()

Moscow_at_square = np.sqrt(np.sum(np.power(Moscow-y, 2)))
print(Moscow_at_square)

print(np.linalg.norm(Moscow-y))

Y = Moscow
X = datas
n = np.size(Moscow)

a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y)) / (n*np.sum(X**2) - np.sum(X)**2)

print(a)

b = np.mean(Y) - a*np.mean(X)
print(b)

y = a*X+b
print(np.linalg.norm(Moscow-y))

# Kaliningrad yearly patterns
kaliningrad_ano1 = kaliningrad[0:12]
kaliningrad_ano2 = kaliningrad[12:24]
kaliningrad_ano3 = kaliningrad[24:36]
kaliningrad_ano4 = kaliningrad[36:48]

plt.plot(kaliningrad)
plt.show()

print(sum(np.isnan(kaliningrad)))
print(np.mean([kaliningrad[3], kaliningrad[5]]))
kaliningrad[4] = np.mean([kaliningrad[3], kaliningrad[5]])

plt.plot(kaliningrad)
plt.show()

# Petersburg yearly patterns
Petersburg_ano1 = Petersburg[0:12]
Petersburg_ano2 = Petersburg[12:24]
Petersburg_ano3 = Petersburg[24:36]
Petersburg_ano4 = Petersburg[36:48]

# Krasnodar yearly patterns
Krasnodar_ano1 = Krasnodar[0:12]
Krasnodar_ano2 = Krasnodar[12:24]
Krasnodar_ano3 = Krasnodar[24:36]
Krasnodar_ano4 = Krasnodar[36:48]

# Ekaterinburg yearly patterns
Ekaterinburg_ano1 = Ekaterinburg[0:12]
Ekaterinburg_ano2 = Ekaterinburg[12:24]
Ekaterinburg_ano3 = Ekaterinburg[24:36]
Ekaterinburg_ano4 = Ekaterinburg[36:48]


