from sklearn.preprocessing import StandardScaler

# def rmfield( a, *fieldnames_to_remove ):
#     return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]
def rmfield(a, b,c,d):
    return a

def normalize(ins, outs):
    print(type(ins))
    ins = rmfield(ins, 'X', 'Y', 'Z')
    scaler = StandardScaler()
    scaler.fit(ins)
    outs = scaler.transform(ins)
    
    return True