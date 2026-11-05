import os
out=[]
for r,_,fs in os.walk('C:\\'):
    # prune heavy dirs
    if any(x in r for x in ('\\.npm-cache','\\node_modules','\\Windows\\WinSxS')):
        continue
    for f in fs:
        if f=='model.pt':
            out.append(os.path.join(r,f))
with open('modelpt_locations.txt','w') as fh:
    fh.write('\n'.join(out))
print('done', len(out))
