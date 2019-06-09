def foo():
    print("starting...")
    res = '5'
    while True:
        res += 'a'
        yield res
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(next(g))
print("*"*20)
print(next(g))
print("*"*20)
print(next(g))