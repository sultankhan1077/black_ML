import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n_groups = 4 
xins = (3.6,2.8,11.8,3) 
ins = (3.8,3,12,3.2)
fig,ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3

g1 = plt.bar(index,xins,bar_width,alpha=0.8,color='b',label='without instrumentation')
g2 = plt.bar(index+bar_width,ins,bar_width,alpha=0.8,color='r',label='with instrumentation')

plt.xlabel('Workload Types')
plt.ylabel('Time Taken(s)')
plt.xticks(index + 0.17, ('Encoding', 'Streaming', 'Compression', 'Graphics Rendering'))
plt.legend()

rng = np.arange(0,14,4)
plt.yticks(rng)

plt.tight_layout()
plt.savefig('img.png')



