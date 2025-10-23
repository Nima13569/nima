lis_mantheg = [(1,3),(3,1)]
w= 0
for i in range(60):
    jame = 0
    for i in lis_mantheg:
        jame +=i[0]*(i[1]-(w*i[0]))
    hamash = -2*(jame)
    new_w= w-(0.01*hamash)
    w=new_w
    print(w)
    if new_w==0:
        print("this the end")