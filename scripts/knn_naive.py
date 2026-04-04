from math import sqrt 

A = (0.2, 0.8, 0.5)
B = (0.9, 0.1, 0.4)
C = (0.4, 0.6, 0.9)
D = (0.7, 0.3, 0.2)
E = (0.1, 0.5, 0.7)

data = [B, C, D, E, A]


Q = (0.3, 0.7, 0.6)

def delta(ref, inp):
    agg = 0;
    agg += (ref[0] - inp[0]) ** 2
    agg += (ref[1] - inp[1]) ** 2
    agg += (ref[2] - inp[2]) ** 2

    return sqrt(agg)

def nearest_one(data: list, inp):
    nearest = data[0]

    for e in data:
        if delta(e, inp) < delta(nearest, inp):
            nearest = e

    return nearest

print(nearest_one(data, Q))

for i, e in enumerate(data):
    print(f"[{i}]={e}, d={delta(e, Q)}") 
