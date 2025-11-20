import math 
z=float(input("enter number 1:"))
x=float(input("enter number 2:"))
y=float(input("enter number 3:"))

i=0
for i in range(100):
    print("i =",i)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||")
    q=i/10
    w=q*z+x
    f=math.log(1+math.e**w)
    javab1=-2*(y-f)*(math.e**f)+(math.e**f)*(q)
    javab2=-2*(y-f)*(math.e**f)+(math.e**f)*(1)
    print(q)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||")
    print("ln(1+e**z) =",f)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||")
    print("javab1 =",javab1)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||")
    print("javab2 =",javab2)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||")
