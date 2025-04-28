
def fibonnaci(n):
    a, b = 0, 1
    for _ in range(n):
        a,b = b, a + b
        return a

print(fibonnaci(10))

def fibonnaci(n):
    a, b = 0, 1
    for i in range(n):
        a,b = b, a + b
        return a

print(fibonnaci(10))